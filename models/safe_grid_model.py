import math
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

from models.grid_model import MazeEnc, miniMazeEnc

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class OH_Policy(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(OH_Policy, self).__init__()

        num_feats = num_inputs[0]

        self.pi = nn.Sequential(
            nn.Linear(num_feats, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.Softmax(),
        )



    def forward(self, x):
        probs = self.pi(x)
        dist = torch.distributions.Categorical(probs = probs)
        return probs, dist


class OH_QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(OH_QNetwork, self).__init__()

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
        return self.q_layer(x)


class OH_ValueNetwork(nn.Module):
    def __init__(self, num_inputs):
        super(OH_ValueNetwork, self).__init__()

        num_feats = num_inputs[0]

        self.val_layer = nn.Sequential(
            nn.Linear(num_feats, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.val_layer(x)



#
# class DQN(nn.Module):
#     def __init__(self, num_inputs, num_actions):
#         super(DQN, self).__init__()
#
#         num_channels = num_inputs[0]
#         self.enc = MazeEnc(num_channels)
#
#         self.q_layer = nn.Sequential(
#             nn.Linear(128, 512),
#             nn.ReLU(),
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_actions),
#         )
#
#     def forward(self, x):
#         phi_x = self.enc(x)
#         q_vals = self.q_layer(phi_x)
#         return q_vals
#
#
# class miniDQN(nn.Module):
#     def __init__(self, num_inputs, num_actions):
#         super(miniDQN, self).__init__()
#
#         num_channels = num_inputs[0]
#         self.enc = miniMazeEnc(num_channels)
#
#         self.q_layer = nn.Sequential(
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_actions),
#         )
#
#     def forward(self, x):
#         phi_x = self.enc(x)
#         q_vals = self.q_layer(phi_x)
#         return q_vals
