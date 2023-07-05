"""
Implement Deep Q-Network using Pytorch: https://arxiv.org/pdf/1312.5602.pdf 

Input: raw Atari 210 x 160 pixel image frames with 128 color palette, 
preprocessed to get the playing area only and cropped to square

Network: 
84 x 84 x 4 image --> Conv Layer (16 8x8 filters with stride 4) --> ReLU
                  --> Conv Layer (32 4 x 4 filters with stride 2) --> ReLU
                  --> Fully Connected Layer (256 rectifier units)
                  --> Fully Connected Layer --> output for valid action (between 4 and 18 games)
"""

import torch
from torch import nn
from collections import deque, namedtuple
import random


class ReplayBuffer:
    """
    Class representing replay buffer, which
    """

    def __init__(self, max_len, batch_size):
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_len)
        self.Transition = namedtuple(
            "Transition", ["state", "action", "reward", "next_state", "done"]
        )

    def __len__(self):
        return len(self.memory)

    def append(self, state, action, reward, next_state, done):
        transition = self.Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self):
        return random.sample(self.memory, k=self.batch_size)


class DeepQNet(nn.Module):
    def __init__(self, input_shape, out_actions):
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0], out_channels=16, kernel_size=8, stride=4
        )
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(
            in_features=32 * input_shape[1] * input_shape[2], out_features=256
        )
        self.fc2 = nn.Linear(in_features=256, out_features=out_actions)

    def forward(self, x):
        x = nn.ReLU(self.conv1(x))
        x = nn.ReLU(self.conv1(x))
        x = torch.flatten(x)
        x = nn.ReLU(self.fc1(x))
        return self.fc2(x)
