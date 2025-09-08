# dotsandboxes/pytorch/DotsAndBoxesNNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DotsAndBoxesNNet(nn.Module):
    """
    Minimal CNN replacement for the old Flatten+Dense model.
    Input:  (B, 1, H, W) with H = 2n+1, W = n+1  (same board as before, just加了通道维1)
    Output: policy logits (B, action_size) and value (B, 1 in [-1, 1])
    """
    def __init__(self, game, num_filters=64, in_channels=5):
        super().__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        # trunk
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(num_filters)

        # policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * self.board_x * self.board_y, self.action_size)

        # value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(self.board_x * self.board_y, 64)
        self.value_fc2  = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, 1, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        pi_logits = self.policy_fc(p)  # raw logits; softmax在loss/预测时再做

        # value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # [-1,1]

        return pi_logits, v
