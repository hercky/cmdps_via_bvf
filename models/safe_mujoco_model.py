import math
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.mujoco_model import ActorCritic


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                        constant_(x, 0), np.sqrt(2))


# ----------------- For DDPG -----------------
class DDPG_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(DDPG_Actor, self).__init__()

        self.max_action = max_action

        self.l1 = nn.Linear(state_dim, 100)
        self.l2 = nn.Linear(100, 50)
        self.l3 = nn.Linear(50, action_dim)




    def forward(self, x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = self.max_action * F.tanh(self.l3(x))
        return x



class DDPG_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPG_Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 200)
        self.l2 = nn.Linear(200 + action_dim, 50)
        self.l3 = nn.Linear(50, 1)


    def forward(self, x, u):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(torch.cat([x, u], 1)))
        x = self.l3(x)
        return x



# ----------------- For PPO -----------------
class SafeActorCritic(ActorCritic):
    # for A2C/PPO models
    def __init__(self, state_dim, action_dim, discrete=False, std=0.0):

        ActorCritic.__init__(self, state_dim, action_dim, discrete, std)


        # Cost dependent estimators

        # reverse value function (the reverse critic)
        # Maybe for later
        # self.reviewer = nn.Sequential(
        #     nn.Linear(state_dim, 200),
        #     nn.Tanh(),
        #     nn.Linear(200, 50),
        #     nn.Tanh(),
        #     nn.Linear(50, 1),
        # )

        self.apply(init_weights)

    def forward(self, x, safe=False, epsilon = None, grad = None):
        value = self.critic(x)
        mu    = self.actor_mu(x)

        # std = self.actor_log_std(x)
        logstd = self.constant_logstd.expand_as(mu)
        std = torch.exp(logstd)

        if not safe:
            # return the regular policy
            dist  = torch.distributions.Normal(mu, std)
        else:
            # compute mu_safe

            # calculate lambda_ here
            lambda_ = F.relu( (-1. * epsilon) / (torch.bmm(grad.unsqueeze(1),
                                                  grad.unsqueeze(2)).squeeze(2)) )

            mu_ = mu - lambda_ * grad

            dist  = torch.distributions.Normal(mu_, std)


        return dist, value, mu

class Cost_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Cost_Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 200)
        self.l2 = nn.Linear(200 + action_dim, 50)
        self.l3 = nn.Linear(50, 1)

        self.apply(init_weights)


    def forward(self, x, u):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(torch.cat([x, u], 1)))
        x = self.l3(x)
        return x


class Cost_Reviewer(nn.Module):
    def __init__(self, state_dim):
        super(Cost_Reviewer, self).__init__()

        self.reviewer = nn.Sequential(
            nn.Linear(state_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.apply(init_weights)


    def forward(self, x):

        return self.reviewer(x)



# ----------------- For PPO -----------------
class SafeUnprojectedActorCritic(ActorCritic):
    # for A2C/PPO models
    def __init__(self, state_dim, action_dim, action_bound, discrete=False, std=0.0):

        ActorCritic.__init__(self, state_dim, action_dim, action_bound, discrete, std)
        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor_mu(x)

        # std = self.actor_log_std(x)
        logstd = self.constant_logstd.expand_as(mu)
        std = torch.exp(logstd)

        return value, mu, std
