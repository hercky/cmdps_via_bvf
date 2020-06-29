# The code is inspired from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

import math
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy

from torch.utils.tensorboard import SummaryWriter

from common.utils import *
from common.multiprocessing_envs import SubprocVecEnv
from agents.base_agent import BasePGAgent


# NOTE: Important to change this corresponding to the architecture
from models.mujoco_model import ActorCritic


class PPOAgent(BasePGAgent):
    """
    """
    def __init__(self, args, env, writer = None):
        """
        the init happens here
        """
        BasePGAgent.__init__(self, args, env)
        self.writer = writer

        self.eval_env = create_env(args)

        self.ub_action = torch.tensor(env.action_space.high, dtype=torch.float, device=self.device)

        self.model = ActorCritic(state_dim=self.state_dim,
                                 action_dim=self.action_dim,
                                 action_bound=self.ub_action,
                                 discrete=self.discrete).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        # create the multiple envs here
        # each with different seed
        def make_env():
            def _thunk():
                env = create_env(args)
                return env

            return _thunk

        envs = [make_env() for i in range(self.args.num_envs)]
        self.envs = SubprocVecEnv(envs)

        #  NOTE: the envs automatically reset when the episode ends...

        self.gamma = self.args.gamma
        self.tau = self.args.gae
        self.mini_batch_size = self.args.batch_size
        self.ppo_epochs = self.args.ppo_updates
        self.clip_param = self.args.clip

        # for results
        self.results_dict = {
            "train_rewards" : [],
            "train_constraints" : [],
            "eval_rewards" : [],
            "eval_constraints" : [],
            "eval_at_step" : [],
        }

        self.cost_indicator = "none"
        if self.args.env_name == "pg":
            self.cost_indicator = 'bombs'
        elif self.args.env_name == "cheetah" or self.args.env_name == "pc":
            self.cost_indicator = 'cost'
        else:
            raise Exception("not implemented yet")

        self.total_steps = 0
        self.num_episodes = 0



    def compute_gae(self, next_value, rewards, masks, values):
        """
        compute the targets with GAE
        """
        values = values + [next_value]
        gae = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.tau * masks[i] * gae
            returns.insert(0, gae + values[i])
        return returns


    def ppo_iter(self, states, actions, log_probs, returns, advantage):
        """
        samples a minibatch from the collected experiren
        """
        batch_size = states.size(0)

        # do updates in minibatches/SGD
        for _ in range(batch_size // self.mini_batch_size):
            # get a minibatch
            rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]



    def ppo_update(self, states, actions, log_probs, returns, advantages, clip_param=0.2):
        """
        does the actual PPO update here
        """
        for _ in range(self.ppo_epochs):

            for state, action, old_log_probs, return_, advantage in self.ppo_iter(states, actions, log_probs, returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = self.args.value_loss_coef * critic_loss + actor_loss - self.args.beta * entropy

                self.optimizer.zero_grad()
                loss.backward()


                self.optimizer.step()


        # For debugging purposes



    def sample_action(self, dist):
        """
        sample an action from the policy such that it stays in bounds
        """
        pass



    def pi(self, state):
        """
        take the action based on the current policy

        For a single state
        """
        with torch.no_grad():

            # convert the state to tensor
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)

            dist, val = self.model(state_tensor)


            action =  dist.sample()

        return action.detach().cpu().squeeze(0).numpy(), val


    def log_episode_stats(self, ep_reward, ep_constraint):
        """
        log the stats for environment[0] performance
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


    def check_termination(self, ):
        """
        if either num of steps or episodes exceed max threshold return the signal to terminate
        :param done:
        :return:
        """
        return self.num_episodes >= self.args.num_episodes

    def episode_based_logger(self, done):
        """

        :return:
        """
        # hack to reuse the same code
        # iteratively add each done episode, so that can eval at regular interval'

        for d_idx in range(done.sum()):

            # do the logging for the first agent only here, only once
            if done[0] and d_idx == 0:
                if self.num_episodes % self.args.log_every == 0:
                    self.log_episode_stats(self.ep_reward, self.ep_constraint)

                # reset the rewards anyways
                self.ep_reward = 0
                self.ep_constraint = 0

            self.num_episodes += 1

            # eval the policy here after eval_every steps
            if self.num_episodes % self.args.eval_every == 0:
                eval_reward, eval_constraint = self.eval()
                self.results_dict["eval_rewards"].append(eval_reward)
                self.results_dict["eval_constraints"].append(eval_constraint)

                log('----------------------------------------')
                log('Eval[R]: {:.2f}\t'.format(eval_reward) + \
                    'Eval[C]: {}\t'.format(eval_constraint) + \
                    'Episode: {}\t'.format(self.num_episodes) + \
                    'avg_eval_reward: {:.2f}\t'.format(np.mean(self.results_dict["eval_rewards"][-10:])) + \
                    'avg_eval_constraint: {:.2f}\t'.format(np.mean(self.results_dict["eval_constraints"][-10:]))
                    )
                log('----------------------------------------')

                if self.writer:
                    self.writer.add_scalar("eval_reward", eval_reward, self.eval_steps)
                    self.writer.add_scalar("eval_constraint", eval_constraint, self.eval_steps)

                self.eval_steps += 1

                # TODO: early stopping prevention is that important?
                # if test_reward > threshold_reward: early_stop = True

            # saving the model
            if self.num_episodes % self.args.checkpoint_interval == 0:
                self.save_models()


    def run(self):
        """
        main PPO algo runs here
        """



        self.num_episodes = 0
        self.eval_steps = 0
        self.ep_reward = 0
        self.ep_constraint = 0
        ep_len = 0
        traj_len = 0

        start_time = time.time()


        # reset
        done = False
        state = self.envs.reset()

        while self.num_episodes < self.args.num_episodes:

            log_probs = []
            values    = []
            states    = []
            actions   = []
            rewards   = []
            masks     = []
            constraints = []
            entropy = 0

            # breakpoint()

            # local_steps_per_epcoh/local_traj = steps_per_epoch/num_actors

            # the lenght of the n-step for updates
            for _ in range(self.args.traj_len):

                state = torch.FloatTensor(state).to(self.device)
                dist, value = self.model(state)
                action = dist.sample()

                next_state, reward, done, info = self.envs.step(action.cpu().numpy())

                # logging mode for only agent 1
                self.ep_reward += reward[0]
                self.ep_constraint += info[0][self.cost_indicator]

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()


                # for actions and log_prob with only 1-dim (mazes)
                if self.args.env_name in ["maze", "door", "apple"]:
                    action = action.unsqueeze(1)
                    log_prob = log_prob.unsqueeze(1)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(self.device))
                constraints.append(torch.FloatTensor([ci[self.cost_indicator] for ci in info]).unsqueeze(1).to(self.device))

                states.append(state)
                actions.append(action)

                state = next_state

                self.total_steps += self.args.num_envs
                # Do logging here depending on step vs episodes
                self.episode_based_logger(done)

                if self.check_termination():
                    break



            # calculate the returns
            next_state = torch.FloatTensor(next_state).to(self.device)
            _, next_value = self.model(next_state)
            returns = self.compute_gae(next_value, rewards, masks, values)

            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            advantage = returns - values

            self.ppo_update(states, actions, log_probs, returns, advantage)


        # done with all the training

        # save the models and results
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

                    # get the policy action
                    action, _  = self.pi(state)

                    next_state, reward, done, info = self.eval_env.step(action)
                    ep_reward += reward
                    ep_len += 1
                    ep_constraint += info[self.cost_indicator]

                    # update the state
                    state = next_state

                    done = done

                avg_reward.append(ep_reward)
                avg_constraint.append(ep_constraint)

        self.eval_env.reset()

        return np.mean(avg_reward), np.mean(avg_constraint)


    def save_models(self):
        """create results dict and save"""
        models = {
            "ac" : self.model.state_dict(),
        }
        torch.save(models,os.path.join(self.args.out, 'models.pt'))
        torch.save(self.results_dict, os.path.join(self.args.out, 'results_dict.pt'))


    def load_models(self):
        models = torch.load(os.path.join(self.args.out, 'models.pt'))
        self.model.load_state_dict(models["ac"])
