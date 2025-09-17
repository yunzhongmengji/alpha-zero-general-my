# dotsandboxes/pytorch/NNet.py  （修改版，支持 AMP、batch predict 与更好 DataLoader）
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils import dotdict
from NeuralNet import NeuralNet
from .DotsAndBoxesNNet import DotsAndBoxesNNet

# ---------- 超参（可按需修改） ----------
args = dotdict({
    'lr': 0.001,
    'epochs': 5,            # 推荐：5（较为优秀档位）
    'batch_size': 128,      # 推荐：128（MX450 显存友好）
    'cuda': True,
    'input_channels': 5,    # 保持原来 5 通道
    'num_workers': 2,       # DataLoader 并行线程（若报错设 0）
    'pin_memory': True,     # DataLoader 加速
    'lr_step_size': 50,     # 每多少轮衰减 lr
    'lr_gamma': 0.5,        # 衰减倍数
})

# ---------- 工具函数（保持原样） ----------
def _score_diff_normalized(board_2d):
    p1, p2 = board_2d[0, -1], board_2d[1, -1]
    diff = p1 - p2
    n = board_2d.shape[1] - 1
    max_score, min_score = n**2, -n**2
    return (diff - min_score) / (max_score - min_score)

def _to_planes(board_2d, C=5):
    H, W = board_2d.shape
    n = W - 1
    planes = np.zeros((H, W, C), dtype=np.float32)

    # H (horizontal edges)
    planes[:n+1, :n, 0] = board_2d[:n+1, :n]
    # V (vertical edges)
    planes[-n:, :, 1]   = board_2d[-n:, :]

    # PASS
    planes[:, :, 2] = board_2d[2, -1]

    # SCORE_DIFF
    planes[:, :, 3] = _score_diff_normalized(board_2d)

    # TURN
    planes[:, :, 4] = 1.0

    return planes

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.game = game
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        # 更稳健的 device 选择
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        self.nnet = DotsAndBoxesNNet(game, in_channels=args.input_channels).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.nnet.parameters(), lr=args.lr, weight_decay=1e-4
        )

        # 学习率调度（按 epoch 衰减）
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=args.lr_step_size,
                                                         gamma=args.lr_gamma)

        # AMP 混合精度，用于加速与显存优化
        self.use_amp = (self.device.type == 'cuda')
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def _boards_to_tensor(self, boards_np):
        planes = np.asarray([_to_planes(b, C=args.input_channels) for b in boards_np], dtype=np.float32)
        x = torch.tensor(planes).permute(0, 3, 1, 2)  # (B,H,W,C)->(B,C,H,W)
        return x.to(self.device)

    def train(self, examples):  # 定义训练函数；examples 是 (board, pi, v) 列表
        """
        examples: list of (board, pi, v)
        使用 AMP（若可用）和 scheduler
        """
        self.nnet.train()  # 将模型切到训练模式（启用BN统计、Dropout等训练行为）

        input_boards, target_pis, target_vs = list(zip(*examples))  # 将样本列表解包成三个元组：棋盘、策略标签π、价值z
        X = np.asarray(input_boards)  # 转为 numpy 数组（便于一次性张量化）
        P = np.asarray(target_pis, dtype=np.float32)  # 策略标签转 float32 的 numpy 数组
        V = np.asarray(target_vs, dtype=np.float32).reshape(-1, 1)  # 价值标签转 float32，并 reshape 成 (B,1)

        ds = TensorDataset(  # 构建 PyTorch 数据集（张量对齐后可被 DataLoader 采样）
            self._boards_to_tensor(X),  # 自定义函数：把 numpy 的棋盘批量转为模型期望的张量格式/形状/类型
            torch.tensor(P, dtype=torch.float32, device=self.device),  # 将策略标签转为 torch 张量（放到指定 device）
            torch.tensor(V, dtype=torch.float32, device=self.device)  # 将价值标签转为 torch 张量（放到指定 device）
        )
        loader = DataLoader(  # 构建 DataLoader：负责分批、打乱、多线程加载
            ds,
            batch_size=args.batch_size,  # 每个 batch 的样本数
            shuffle=True,  # 每个 epoch 打乱数据
            num_workers=args.num_workers,  # 数据加载的子进程数量
            pin_memory=args.pin_memory  # 若为 True，固定内存页以加速拷贝到 GPU
        )

        for ep in range(args.epochs):  # 进行多个 epoch 的训练
            epoch_loss = 0.0  # 记录本 epoch 的损失和（稍后可算平均）
            for xb, pb, vb in loader:  # 从 DataLoader 取一批：棋盘、策略标签、价值标签
                self.optimizer.zero_grad()  # 清空上一轮留下的梯度（避免累积）
                if self.use_amp:  # 若启用自动混合精度（AMP）
                    with torch.cuda.amp.autocast():  # 在 autocast 区域内自动选择 FP16/FP32 混合计算
                        pi_logits, v_out = self.nnet(xb)  # 前向：输出策略logits和价值预测
                        log_probs = F.log_softmax(pi_logits, dim=1)  # 对策略 logits 做 log_softmax 得到 log 概率
                        policy_loss = -(pb * log_probs).sum(dim=1).mean()  # 软标签交叉熵：-sum(p*log q) 后对 batch 求均值
                        value_loss = F.mse_loss(v_out, vb)  # 价值头的均方误差损失 MSE(v_out, z)
                        loss = policy_loss + value_loss  # 总损失：策略损失 + 价值损失（L2 可用优化器 weight_decay 代替）
                    # AMP backward
                    self.scaler.scale(loss).backward()  # 使用 GradScaler 对 loss 进行缩放后反传，避免 FP16 下溢
                    torch.nn.utils.clip_grad_norm_(  # 梯度裁剪，防止梯度爆炸；最大范数=1.0
                        self.nnet.parameters(), 1.0
                    )
                    self.scaler.step(self.optimizer)  # 使用缩放器驱动优化器 step（内部处理溢出/跳步）
                    self.scaler.update()  # 更新缩放因子（自适应扩/缩放以保持数值稳定）
                else:  # 非 AMP 路径（纯 FP32 训练）
                    pi_logits, v_out = self.nnet(xb)  # 前向
                    log_probs = F.log_softmax(pi_logits, dim=1)  # 策略 log 概率
                    policy_loss = -(pb * log_probs).sum(dim=1).mean()  # 策略损失（同上）
                    value_loss = F.mse_loss(v_out, vb)  # 价值损失
                    loss = policy_loss + value_loss  # 总损失
                    loss.backward()  # 反向传播，计算梯度
                    torch.nn.utils.clip_grad_norm_(  # 梯度裁剪
                        self.nnet.parameters(), 1.0
                    )
                    self.optimizer.step()  # 参数更新

                epoch_loss += loss.item()  # 累加本批次的标量损失（用于日志统计）

            # 每个 epoch 更新 lr scheduler
            self.scheduler.step()  # 调用学习率调度器步进（具体策略由 scheduler 类型决定）

            # 可选：打印 epoch 信息（便于监控）
            print(  # 打印：当前 epoch 序号 / 总 epoch、累计损失、当前学习率
                f"[NNet] epoch {ep + 1}/{args.epochs}, "
                f"loss={epoch_loss:.4f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
            )

    @torch.no_grad()
    def predict(self, board):
        """
        单个 board 的预测接口（保持兼容）
        """
        self.nnet.eval()
        b = np.copy(board)[np.newaxis, ...]        # (1,H,W)
        x = self._boards_to_tensor(b)              # (1,5,H,W)
        if self.use_amp:
            with torch.cuda.amp.autocast():
                pi_logits, v = self.nnet(x)
        else:
            pi_logits, v = self.nnet(x)
        pi = torch.softmax(pi_logits, dim=1).cpu().numpy()[0]
        v  = v.cpu().numpy()[0]
        return pi, v

    @torch.no_grad()
    def predict_batch(self, boards_np):
        """
        批量推理接口：boards_np 为 (B, H, W) numpy array
        返回：
            pis: (B, action_size)
            vs:  (B, 1)
        适用于在 MCTS 中把多个 leaf 一起送到 GPU 提高吞吐。
        """
        self.nnet.eval()
        x = self._boards_to_tensor(boards_np)  # (B, C, H, W)
        if self.use_amp:
            with torch.cuda.amp.autocast():
                pi_logits, v = self.nnet(x)
        else:
            pi_logits, v = self.nnet(x)
        pis = torch.softmax(pi_logits, dim=1).cpu().numpy()
        vs  = v.cpu().numpy()
        return pis, vs

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        ckpt = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(ckpt['state_dict'])
        self.optimizer.load_state_dict(ckpt.get('optimizer', {}))
        if 'scheduler' in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler'])
            except Exception:
                pass
        self.nnet.to(self.device)
