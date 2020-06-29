import math
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MazeEnc(nn.Module):

    def __init__(self, num_channels):
        """
        intial code taken from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/model.py

        num_channels: the num of input channels
        hidden_size: the final size of the encoding
        """
        super(MazeEnc, self).__init__()

        # init_ = lambda m: init(m,
        #     nn.init.orthogonal_,
        #     lambda x: nn.init.constant_(x, 0),
        #     nn.init.calculate_gain('relu'))

        # architecture taken from Safe Lyp Appendix F
		#  convolutional neural network with filters of size 3 × 3 × 3 × 32,
        # 32 × 3 × 3 × 64, and 64 × 3 × 3 × 128, with 2 × 2 max-pooling and relu activations after each.


        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3,),
            nn.MaxPool2d(kernel_size=2,),
            nn.ReLU(),
            #
            nn.Conv2d(32, 64, 3,),
            nn.MaxPool2d(kernel_size=2,),
            nn.ReLU(),
            #
            nn.Conv2d(64, 128, 3,),
            torch.nn.MaxPool2d(kernel_size=2,),
            nn.ReLU(),
            Flatten(),
        )

        # in the end it will return [batch_size, 128] shape tensor


        self.train()

    def forward(self, x):

        return self.main(x)



class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        num_channels = num_inputs[0]
        self.enc = MazeEnc(num_channels)

        self.q_layer = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        phi_x = self.enc(x)
        q_vals = self.q_layer(phi_x)
        return q_vals


class OneHotDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(OneHotDQN, self).__init__()

        num_feats = num_inputs[0]

        self.q_layer = nn.Sequential(
            nn.Linear(num_feats, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        q_vals = self.q_layer(x)
        return q_vals


class miniMazeEnc(nn.Module):

    def __init__(self, num_channels):
        """
        sa
        """
        super(miniMazeEnc, self).__init__()


        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3,),
            nn.MaxPool2d(kernel_size=2,),
            nn.ReLU(),
            #
            nn.Conv2d(32, 64, 3,),
            nn.MaxPool2d(kernel_size=2,),
            nn.ReLU(),
            #
            Flatten(),
        )

        # in the end it will return [batch_size, 64] shape tensor
        self.train()

    def forward(self, x):

        return self.main(x)


class miniDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(miniDQN, self).__init__()

        num_channels = num_inputs[0]
        self.enc = miniMazeEnc(num_channels)

        self.q_layer = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        phi_x = self.enc(x)
        q_vals = self.q_layer(phi_x)
        return q_vals



class OneHotValueNetwork(nn.Module):
    def __init__(self, num_inputs):
        super(OneHotValueNetwork, self).__init__()

        num_feats = num_inputs[0]

        self.value = nn.Sequential(
            nn.Linear(num_feats, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        val = self.value(x)
        return val
