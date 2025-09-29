# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Standard ResNet basic block: Conv(3x3)-BN-ReLU-Conv(3x3)-BN + identity, then ReLU.
    这里我们保持通道数不变，结构简单稳定。
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + x)   # residual add + ReLU
        return out


class DotsAndBoxesNNet(nn.Module):
    """
    ResNet-style backbone with dual heads (policy/value).

    Input : (B, C=5, H=2n+1, W=n+1)
    Policy: (B, action_size)    -- raw logits (softmax在loss/预测时做)
    Value : (B, 1) in [-1, 1]
    """
    def __init__(self, game, num_filters=128, in_channels=5, num_res_blocks=6, dropout=0.3):
        super().__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        # residual trunk
        blocks = [ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        self.trunk = nn.Sequential(*blocks)

        # policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * self.board_x * self.board_y, self.action_size)
        self.policy_dropout = nn.Dropout(p=dropout)

        # value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(self.board_x * self.board_y, 256)
        self.value_dropout = nn.Dropout(p=dropout)
        self.value_fc2  = nn.Linear(256, 1)


    def forward(self, x):
        # x: (B, C, H, W)
        x = self.stem(x)
        x = self.trunk(x)

        # policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        # [MOD] GPU/AMP 下可能出现非连续张量，view 会报错；改为安全展平：
        #      torch.flatten(p, 1) 等价于 p.reshape(p.shape[0], -1)，可自动处理非连续内存
        p = torch.flatten(p, 1)  # [MOD]
        # 可选一次性自检（调通后可注释掉）：
        # assert p.dim() == 2, f"[policy] flatten failed: got shape {tuple(p.shape)}"
        p = self.policy_dropout(p)
        pi_logits = self.policy_fc(p)

        # value
        v = F.relu(self.value_bn(self.value_conv(x)))
        # [MOD] 同理，使用安全展平，避免非连续张量用 view：
        v = torch.flatten(v, 1)  # [MOD]
        # assert v.dim() == 2, f"[value] flatten failed: got shape {tuple(v.shape)}"
        v = F.relu(self.value_fc1(v))
        v = self.value_dropout(v)
        v = torch.tanh(self.value_fc2(v))
        return pi_logits, v
