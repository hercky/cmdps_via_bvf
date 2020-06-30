import math
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import gym
import copy


from common.utils import *

from common.multiprocessing_envs import SubprocVecEnv
from torchvision.transforms import ToTensor


from models.grid_model import OneHotDQN


from common.schedules import LinearSchedule, ExponentialSchedule

class SarsaAgent(object):

    def __init__(self,
                 args,
                 env,
                 writer = None):
        """
        init the agent here
        """
        self.eval_env = copy.deepcopy(env)
        self.args = args

        self.state_dim = env.reset().shape

        self.action_dim = env.action_space.n

        self.device = torch.device("cuda" if (torch.cuda.is_available() and  self.args.gpu) else "cpu")

        # set the same random seed in the main launcher
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if self.args.gpu:
            torch.cuda.manual_seed(self.args.seed )

        self.writer = writer

        if self.args.env_name == "grid":
            self.dqn = OneHotDQN(self.state_dim, self.action_dim).to(self.device)
            self.dqn_target = OneHotDQN(self.state_dim, self.action_dim).to(self.device)
        else:
            raise Exception("not implemented yet!")

        # copy parameters
        self.dqn_target.load_state_dict(self.dqn.state_dict())

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.args.lr)

        # for actors
        def make_env():
            def _thunk():
                env = create_env(args)
                return env

            return _thunk

        envs = [make_env() for i in range(self.args.num_envs)]
        self.envs = SubprocVecEnv(envs)


        # create epsilon and beta schedule
        # NOTE: hardcoded for now
        self.eps_decay = LinearSchedule(50000 * 200, 0.01, 1.0)
        # self.eps_decay = LinearSchedule(self.args.num_episodes * 200, 0.01, 1.0)

        self.total_steps = 0
        self.num_episodes = 0

        # for storing resutls
        self.results_dict = {
            "train_rewards" : [],
            "train_constraints" : [],
            "eval_rewards" : [],
            "eval_constraints" : [],
        }

        self.cost_indicator = "none"
        if "grid" in self.args.env_name:
            self.cost_indicator = 'pit'
        else:
            raise Exception("not implemented yet")

        self.eps = self.eps_decay.value(self.total_steps)


    def pi(self, state, greedy_eval=False):
        """
        take the action based on the current policy
        """
        with torch.no_grad():
            # to take random action or not
            if (random.random() > self.eps_decay.value(self.total_steps)) or greedy_eval:
                q_value = self.dqn(state)

                # chose the max/greedy actions
                action = q_value.max(1)[1].cpu().numpy()
            else:
                action = np.random.randint(0, high=self.action_dim, size = (self.args.num_envs, ))

        return action


    def compute_n_step_returns(self, next_value, rewards, masks):
        """
        n-step SARSA returns
        """
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.args.gamma * R * masks[step]
            returns.insert(0, R)

        return returns


    def log_episode_stats(self, ep_reward, ep_constraint):
        """
        log the stats for environment performance
        """
        # log episode statistics
        self.results_dict["train_rewards"].append(ep_reward)
        self.results_dict["train_constraints"].append(ep_constraint)
        if self.writer:
            self.writer.add_scalar("Return", ep_reward, self.num_episodes)
            self.writer.add_scalar("Constraint",  ep_constraint, self.num_episodes)


        log(
            'Num Episode {}\t'.format(self.num_episodes) + \
            'E[R]: {:.2f}\t'.format(ep_reward) +\
            'E[C]: {:.2f}\t'.format(ep_constraint) +\
            'avg_train_reward: {:.2f}\t'.format(np.mean(self.results_dict["train_rewards"][-100:])) +\
            'avg_train_constraint: {:.2f}\t'.format(np.mean(self.results_dict["train_constraints"][-100:]))
            )




    def run(self):
        """
        Learning happens here
        """

        self.total_steps = 0
        self.eval_steps = 0

        # reset state and env
        # reset exploration porcess
        state = self.envs.reset()
        prev_state = state

        ep_reward = 0
        ep_len = 0
        ep_constraint = 0
        start_time = time.time()


        while self.num_episodes < self.args.num_episodes:

            values      = []
            c_q_vals    = []
            c_r_vals    = []

            states      = []
            actions     = []
            mus         = []
            prev_states = []

            rewards     = []
            done_masks  = []
            begin_masks = []
            constraints = []


            # n-step sarsa
            for _ in range(self.args.traj_len):

                state = torch.FloatTensor(state).to(self.device)

                # get the action
                action = self.pi(state)
                next_state, reward, done, info = self.envs.step(action)

                # convert it back to tensor
                action = torch.LongTensor(action).unsqueeze(1).to(self.device)

                q_values = self.dqn(state)
                Q_value = q_values.gather(1, action)

                # logging mode for only agent 1
                ep_reward += reward[0]
                ep_constraint += info[0][self.cost_indicator]

                values.append(Q_value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                done_masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(self.device))
                begin_masks.append(torch.FloatTensor([ci['begin'] for ci in info]).unsqueeze(1).to(self.device))
                constraints.append(torch.FloatTensor([ci[self.cost_indicator] for ci in info]).unsqueeze(1).to(self.device))
                prev_states.append(prev_state)
                states.append(state)
                actions.append(action)

                # update states
                prev_state = state
                state = next_state

                self.total_steps += (1 * self.args.num_envs)


                # hack to reuse the same code
                # iteratively add each done episode, so that can eval at regular interval
                for _ in range(done.sum()):
                    if done[0]:
                        if self.num_episodes % self.args.log_every == 0:
                            self.log_episode_stats(ep_reward, ep_constraint)

                        # reset the rewards anyways
                        ep_reward = 0
                        ep_constraint = 0

                    self.num_episodes += 1

                    # eval the policy here after eval_every steps
                    if self.num_episodes  % self.args.eval_every == 0:
                        eval_reward, eval_constraint = self.eval()
                        self.results_dict["eval_rewards"].append(eval_reward)
                        self.results_dict["eval_constraints"].append(eval_constraint)

                        log('----------------------------------------')
                        log('Eval[R]: {:.2f}\t'.format(eval_reward) +\
                            'Eval[C]: {}\t'.format(eval_constraint) +\
                            'Episode: {}\t'.format(self.num_episodes) +\
                            'avg_eval_reward: {:.2f}\t'.format(np.mean(self.results_dict["eval_rewards"][-10:])) +\
                            'avg_eval_constraint: {:.2f}\t'.format(np.mean(self.results_dict["eval_constraints"][-10:]))
                            )
                        log('----------------------------------------')

                        if self.writer:
                            self.writer.add_scalar("eval_reward", eval_reward, self.eval_steps)
                            self.writer.add_scalar("eval_constraint", eval_constraint, self.eval_steps)

                        self.eval_steps += 1



            # break here
            if self.num_episodes >= self.args.num_episodes:
                break

            # calculate targets here
            next_state = torch.FloatTensor(next_state).to(self.device)

            next_q_values = self.dqn(next_state)
            next_action = self.pi(next_state)
            next_action = torch.LongTensor(next_action).unsqueeze(1).to(self.device)

            next_q_values = next_q_values.gather(1, next_action)


            # calculate targets
            target_Q_vals = self.compute_n_step_returns(next_q_values, rewards, done_masks)
            Q_targets = torch.cat(target_Q_vals).detach()
            Q_values = torch.cat(values)

            # bias corrected loss
            loss  = F.mse_loss(Q_values, Q_targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



        # done with all the training

        # save the models
        self.save_models()



    def eval(self):
        """
        evaluate the current policy and log it
        """
        avg_reward = []
        avg_constraint = []

        with torch.no_grad():
            for _ in range(self.args.eval_n):

                state = self.eval_env.reset()
                done = False
                ep_reward = 0
                ep_constraint = 0
                ep_len = 0
                start_time = time.time()

                while not done:

                    # convert the state to tensor
                    state_tensor =  torch.FloatTensor(state).to(self.device).unsqueeze(0)

                    # get the policy action
                    action = self.pi(state_tensor, greedy_eval=True)[0]

                    next_state, reward, done, info = self.eval_env.step(action)
                    ep_reward += reward
                    ep_len += 1
                    ep_constraint += info[self.cost_indicator]

                    # update the state
                    state = next_state


                avg_reward.append(ep_reward)
                avg_constraint.append(ep_constraint)

        return np.mean(avg_reward), np.mean(avg_constraint)

    def save_models(self):
        """create results dict and save"""
        models = {
            "dqn" : self.dqn.state_dict(),
            "env" : copy.deepcopy(self.eval_env),
        }
        torch.save(models,os.path.join(self.args.out, 'models.pt'))
        torch.save(self.results_dict, os.path.join(self.args.out, 'results_dict.pt'))


    def load_models(self):
        models = torch.load(os.path.join(self.args.out, 'models.pt'))
        self.dqn.load_state_dict(models["dqn"])
        self.eval_env = models["env"]
