import math
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


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

# ------------------ For AC/PPO  ---------------
class ActorCritic(nn.Module):
    # for PPO models
    def __init__(self, state_dim, action_dim, action_bound, discrete=False, std=0.0):
        super(ActorCritic, self).__init__()

        self.discrete = discrete
        self.action_bound = action_bound

        # value function
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        # actor - mean
        self.actor_mu = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh() ,
            nn.Linear(50, action_dim),
            nn.Tanh(),
        )

        # actor - log variance
        # self.actor_log_std = nn.Sequential(
        #     nn.Linear(state_dim, 100),
        #     nn.Tanh(),
        #     nn.Linear(100, 50),
        #     nn.Tanh(),
        #     nn.Linear(50, action_dim),
        # )

        self.constant_logstd = nn.Parameter(torch.zeros(action_dim))

        # constant std deviation for gaussian parameterization of the policy
        # can also be used for learning

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)

        mu    = self.actor_mu(x) * self.action_bound
        # mu    = self.actor_mu(x)

        # std = self.actor_log_std(x)
        logstd = self.constant_logstd.expand_as(mu)
        std = torch.exp(logstd)


        # std = self.actor_log_std(x).exp()

        if self.discrete:
            dist = torch.distributions.Categorical(logits=mu)
        else:
            dist  = torch.distributions.Normal(mu, std)

        return dist, value

# ------------------ For Deterministic-PPO ---------------
class OP_DDPG_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, sigma):
        super(OP_DDPG_Actor, self).__init__()

        self.max_action = max_action
        self.sigma = sigma

        self.mu  = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh() ,
            nn.Linear(50, action_dim),
            # nn.Tanh(),
        )


    def forward(self, x):
        # scaled mu
        mu =  self.max_action * self.mu(x)

        dist = torch.distributions.Normal(mu, self.sigma)

        return mu, dist


class OP_DDPG_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(OP_DDPG_Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 200)
        self.l2 = nn.Linear(200 + action_dim, 50)
        self.l3 = nn.Linear(50, 1)


    def forward(self, x, u):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(torch.cat([x, u], 1)))
        x = self.l3(x)
        return x



class Deterministic_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, sigma):
        super(Deterministic_Actor, self).__init__()

        self.max_action = max_action
        self.sigma = sigma


        self.mu  = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh() ,
            nn.Linear(50, action_dim),
            # nn.Tanh(),
        )

        # self.constant_logstd = nn.Parameter(torch.zeros(action_dim))

        # constant std deviation for gaussian parameterization of the policy
        # can also be used for learning

        self.apply(init_weights)



    def forward(self, x):
        # scaled mu
        mu = self.mu(x)

        # mu =  self.max_action * mu

        # logstd = self.constant_logstd.expand_as(mu)
        # std = torch.exp(logstd)


        dist = torch.distributions.Normal(mu, self.sigma)

        return mu, dist


class Deterministic_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Deterministic_Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 200)
        self.l2 = nn.Linear(200 + action_dim, 50)
        self.l3 = nn.Linear(50, 1)


    def forward(self, x, u):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(torch.cat([x, u], 1)))
        x = self.l3(x)
        return x
