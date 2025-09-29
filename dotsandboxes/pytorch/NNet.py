# -*- coding: utf-8 -*-
# dotsandboxes/pytorch/NNet.py  （修复 DataLoader pin_memory 报错 + 显式处理 torch.load FutureWarning）
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
    'epochs': 10,
    'batch_size': 64,
    'cuda': True,
    'input_channels': 5,
    'num_workers': 2,
    'pin_memory': True,
    'lr_step_size': 50,
    'lr_gamma': 0.5,
})


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

        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        self.nnet = DotsAndBoxesNNet(game, in_channels=args.input_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=args.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

        # AMP（保留你之前的设置；如需用新接口，可按我之前给你的版本改）
        self.use_amp = (self.device.type == 'cuda')
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def _boards_to_tensor_cpu(self, boards_np):
        """
        [MOD] 保持在 CPU 的张量，配合 DataLoader(pin_memory=True) + non_blocking=True
        之前的问题是这里直接 .to(self.device) 导致 Dataset 内部是 CUDA 张量，pin_memory 线程崩溃。
        """
        planes = np.asarray([_to_planes(b, C=args.input_channels) for b in boards_np], dtype=np.float32)
        x = torch.tensor(planes).permute(0, 3, 1, 2).contiguous()  # (B,H,W,C)->(B,C,H,W), 保持 CPU
        return x  # 不要 .to(self.device)！

    def train(self, examples):
        """
        examples: list of (board, pi, v)
        修复点：数据集全部留在 CPU；每个 batch 再搬到 GPU（non_blocking=True）
        """
        self.nnet.train()

        input_boards, target_pis, target_vs = list(zip(*examples))
        X = np.asarray(input_boards)
        P = np.asarray(target_pis, dtype=np.float32)
        V = np.asarray(target_vs, dtype=np.float32).reshape(-1, 1)

        s = P.sum(axis=1, keepdims=False)
        print("[DEBUG] pi.sum(): min/mean/max = %.4f / %.4f / %.4f" % (s.min(), s.mean(), s.max()))

        # [MOD] Dataset 内的张量全部是 CPU 的
        x_cpu = self._boards_to_tensor_cpu(X)  # CPU tensor
        p_cpu = torch.tensor(P, dtype=torch.float32)  # CPU tensor
        v_cpu = torch.tensor(V, dtype=torch.float32)  # CPU tensor

        ds = TensorDataset(x_cpu, p_cpu, v_cpu)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(args.pin_memory and self.device.type == "cuda"),  # 仅 CUDA 时有意义
            persistent_workers=True if (self.device.type == "cuda" and args.num_workers > 0) else False,
        )

        for ep in range(args.epochs):
            total_items = 0
            sum_total, sum_pi, sum_v = 0.0, 0.0, 0.0

            for xb_cpu, pb_cpu, vb_cpu in loader:
                xb = xb_cpu.to(self.device, non_blocking=True)
                pb = pb_cpu.to(self.device, non_blocking=True)
                vb = vb_cpu.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                pi_logits, v_out = self.nnet(xb)
                log_probs = F.log_softmax(pi_logits, dim=1)
                policy_loss = -(pb * log_probs).sum(dim=1).mean()  # batch-mean
                value_loss = F.mse_loss(v_out, vb)  # batch-mean
                loss = policy_loss + value_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), 1.0)
                self.optimizer.step()

                bs = xb.size(0)
                # 还原为“样本总和”，最后再除以总样本得到“样本平均”
                sum_pi += policy_loss.item() * bs
                sum_v += value_loss.item() * bs
                sum_total += loss.item() * bs
                total_items += bs

            self.scheduler.step()
            epoch_pi = sum_pi / total_items
            epoch_value = sum_v / total_items
            epoch_total = sum_total / total_items
            print(f"[NNet] epoch {ep + 1}/{args.epochs}, "
                  f"loss={epoch_total:.4f} (pi={epoch_pi:.4f}, v={epoch_value:.4f}), "
                  f"lr={self.optimizer.param_groups[0]['lr']:.6f}, samples={total_items}")

    @torch.no_grad()
    def predict(self, board):
        """
        单例推理；这里直接把 CPU 张量搬到 device 即可
        """
        self.nnet.eval()
        b = np.copy(board)[np.newaxis, ...]
        x_cpu = self._boards_to_tensor_cpu(b)        # CPU
        x = x_cpu.to(self.device, non_blocking=False)  # 体量小，non_blocking 随意

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
        批量推理；同上，先在 CPU 组装，再搬到 device
        """
        self.nnet.eval()
        x_cpu = self._boards_to_tensor_cpu(boards_np)  # CPU
        x = x_cpu.to(self.device, non_blocking=False)

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

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', safe_mode='explicit_false'):
        """
        safe_mode:
          - 'explicit_false'  -> 显式 weights_only=False（你自己的 ckpt，兼容 optimizer/scheduler）
          - 'weights_only'    -> 显式 weights_only=True（只加载权重，适合不受信第三方权重）
        """
        filepath = os.path.join(folder, filename)
        if safe_mode == 'weights_only':
            # 只加载权重；若文件不是纯 state_dict，而是字典，则只取其中名为 'state_dict' 的部分（若存在）
            obj = torch.load(filepath, map_location=self.device, weights_only=True)
            if isinstance(obj, dict):
                # 可能直接就是 state_dict（weights_only=True 时通常如此）
                self.nnet.load_state_dict(obj)
            else:
                # 非预期格式，降级使用常规路径（注意风险）
                ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
                self.nnet.load_state_dict(ckpt['state_dict'])
        else:
            # 显式声明 weights_only=False，消除 FutureWarning，并保持对 optimizer/scheduler 的兼容
            ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
            self.nnet.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt.get('optimizer', {}))
            if 'scheduler' in ckpt:
                try:
                    self.scheduler.load_state_dict(ckpt['scheduler'])
                except Exception:
                    pass
        self.nnet.to(self.device)
