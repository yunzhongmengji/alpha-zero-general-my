# dotsandboxes/pytorch/NNet.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils import dotdict
from NeuralNet import NeuralNet
from .DotsAndBoxesNNet import DotsAndBoxesNNet

args = dotdict({
    'lr': 0.001,
    'epochs': 10,
    'batch_size': 64,
    'cuda': True,
    'input_channels': 5,   # ★ 5通道
})

def _score_diff_normalized(board_2d):
    """
    分差归一化到[0,1]，不改写原棋盘:
    diff = p1 - p2;  p1 = board[0,-1], p2 = board[1,-1]
    """
    p1, p2 = board_2d[0, -1], board_2d[1, -1]
    diff = p1 - p2
    n = board_2d.shape[1] - 1
    max_score, min_score = n**2, -n**2
    return (diff - min_score) / (max_score - min_score)

def _to_planes(board_2d, C=5):
    """
    将( H=2n+1, W=n+1 )的棋盘转为 (H, W, 5)：
    0: H (横边)      -> board[:n+1, :n]
    1: V (竖边)      -> board[-n:, :]
    2: PASS 常数面   -> board[2, -1]
    3: SCORE_DIFF    -> 归一化分差常数面
    4: TURN 常数面   -> 1.0（规范局面视角为当前手）
    """
    H, W = board_2d.shape
    n = W - 1
    planes = np.zeros((H, W, C), dtype=np.float32)

    # H
    planes[:n+1, :n, 0] = board_2d[:n+1, :n]
    # V
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

        self.nnet = DotsAndBoxesNNet(game, in_channels=args.input_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=args.lr)

    def _boards_to_tensor(self, boards_np):
        """
        boards_np: (B, H, W) numpy
        -> (B, 5, H, W) torch.float32 on device
        """
        planes = np.asarray([_to_planes(b, C=args.input_channels) for b in boards_np], dtype=np.float32)
        x = torch.tensor(planes).permute(0, 3, 1, 2)  # (B,H,W,C)->(B,C,H,W)
        return x.to(self.device)

    def train(self, examples):
        """
        examples: list of (board, pi, v)
        """
        self.nnet.train()

        input_boards, target_pis, target_vs = list(zip(*examples))
        X = np.asarray(input_boards)
        P = np.asarray(target_pis, dtype=np.float32)
        V = np.asarray(target_vs,  dtype=np.float32).reshape(-1, 1)

        ds = TensorDataset(self._boards_to_tensor(X),
                           torch.tensor(P, dtype=torch.float32, device=self.device),
                           torch.tensor(V, dtype=torch.float32, device=self.device))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

        for _ in range(args.epochs):
            for xb, pb, vb in loader:
                self.optimizer.zero_grad()
                pi_logits, v_out = self.nnet(xb)

                # policy：soft targets 交叉熵
                log_probs = F.log_softmax(pi_logits, dim=1)
                policy_loss = -(pb * log_probs).sum(dim=1).mean()

                # value：MSE
                value_loss = F.mse_loss(v_out, vb)

                loss = policy_loss + value_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), 1.0)
                self.optimizer.step()

    @torch.no_grad()
    def predict(self, board):
        """
        board: (H, W) numpy
        return: (pi (action_size,), v (1,))
        """
        self.nnet.eval()
        b = np.copy(board)[np.newaxis, ...]        # (1,H,W)
        x = self._boards_to_tensor(b)              # (1,5,H,W)
        pi_logits, v = self.nnet(x)
        pi = torch.softmax(pi_logits, dim=1).cpu().numpy()[0]
        v  = v.cpu().numpy()[0]
        return pi, v

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        torch.save(self.nnet.state_dict(), filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        self.nnet.load_state_dict(torch.load(filepath, map_location=self.device))
        self.nnet.to(self.device)
