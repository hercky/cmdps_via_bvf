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


from models.safe_mujoco_model import Cost_Critic, Cost_Reviewer, SafeUnprojectedActorCritic


class SafePPOAgent(BasePGAgent):
    """
    """
    def __init__(self, args, env, writer = None):
        """
        the init happens here
        """
        BasePGAgent.__init__(self, args, env)
        self.writer = writer

        # self.eval_env = copy.deepcopy(env)
        self.eval_env = create_env(args)

        self.ub_action = torch.tensor(env.action_space.high, dtype=torch.float, device=self.device)

        self.ac_model = SafeUnprojectedActorCritic(state_dim=self.state_dim,
                                 action_dim=self.action_dim,
                                 action_bound=self.ub_action,
                                 discrete=self.discrete).to(self.device)
        self.cost_reviewer = Cost_Reviewer(state_dim=self.state_dim).to(self.device)
        self.cost_critic = Cost_Critic(state_dim=self.state_dim,
                                       action_dim=self.action_dim).to(self.device)

        self.sg_model = Cost_Reviewer(state_dim=self.state_dim).to(self.device)
        self.sg_optimizer = optim.Adam(self.sg_model.parameters(), lr=self.args.lr)


        self.ac_optimizer = optim.Adam(self.ac_model.parameters(), lr=self.args.lr)
        self.review_optimizer = optim.Adam(self.cost_reviewer.parameters(), lr=self.args.cost_reverse_lr)
        self.critic_optimizer = optim.Adam(self.cost_critic.parameters(), lr=self.args.cost_q_lr)

        # create the multiple envs here
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
        }


        self.cost_indicator = "none"
        if self.args.env_name == "pg":
            self.cost_indicator = 'bombs'
        elif self.args.env_name == "pc" or self.args.env_name == "cheetah":
            self.cost_indicator = 'cost'
        elif "torque" in self.args.env_name:
            self.cost_indicator = 'cost'
        else:
            raise Exception("not implemented yet")


        self.total_steps = 0
        self.num_episodes = 0


        # reviewer parameters
        self.r_gamma = self.args.gamma




    def compute_gae(self, next_value, rewards, masks, values):
        """
        compute the targets with GAE

        This is the same for the Q(s,a) too
        """
        values = values + [next_value]
        gae = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.tau * masks[i] * gae
            returns.insert(0, gae + values[i])
        return returns


    def compute_review_gae(self, prev_value, rewards, begin_masks, r_values):
        """
        compute the targets with GAE, to be implemented
        """
        r_values = [prev_value] + r_values
        gae = 0
        returns = []
        for i in range(len(rewards)):
            r_delta = rewards[i] + self.r_gamma * r_values[i-1] * begin_masks[i] - r_values[i]
            gae = r_delta + self.r_gamma * self.tau * begin_masks[i] * gae
            returns.append(gae + r_values[i])
        return returns



    def ppo_iter(self, states, actions, log_probs, returns, advantage,   sg_returns, sg_advantage, c_r_returns, c_q_returns, c_costs):
        """
        samples a minibatch from the collected experiren
        """
        batch_size = states.size(0)

        # do updates in minibatches/SGD
        for _ in range(batch_size // self.mini_batch_size):
            # get a minibatch
            rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :],   sg_returns[rand_ids, :], sg_advantage[rand_ids, :], c_r_returns[rand_ids, :], c_q_returns[rand_ids, :], c_costs[rand_ids, :]



    def safe_ac(self, state, current_cost= 0.0):
        """
        get the projected mean dist and val
        """
        # calculate uncorrected mu and std
        val, mu, std = self.ac_model(state)

        # get the gradient Q_D wrt  mu
        q_D = self.cost_critic(state, mu)

        gradients = torch.autograd.grad(outputs=q_D,
                                        inputs= mu,
                                        grad_outputs=torch.ones(q_D.shape).to(self.device),
                                        only_inputs=True,
                                        create_graph=True, #
                                        retain_graph=True, #
                                        )[0]

        # get  the epsilon here
        epsilon = (self.args.d0 + current_cost - (self.cost_reviewer(state) +  q_D))

        # grad_sq
        grad_sq = torch.bmm(gradients.unsqueeze(1), gradients.unsqueeze(2)).squeeze(2) + 1e-8

        gradients = gradients + 1e-8

        lambda_ = F.relu((-1. * epsilon))/grad_sq

        correction = lambda_ * gradients

        # make sure correction in bounds/ not too large
        correction = F.tanh(correction) * self.ub_action

        mu_safe = mu - correction

        # sample from the new dist here
        dist = torch.distributions.Normal(mu_safe, std)

        return val, mu_safe, dist




    def ppo_update(self, states, actions, log_probs, returns, advantages,  sg_returns, sg_advantage, c_r_returns, c_q_returns, c_costs, clip_param=0.2):
        """
        does the actual PPO update here
        """
        for _ in range(self.ppo_epochs):
            for state, action, old_log_probs, return_, advantage, sg_adv, sg_return_, c_r_return_, c_q_return_, c_cost_ in self.ppo_iter(states, actions, log_probs, returns, advantages,  sg_returns, sg_advantage, c_r_returns, c_q_returns, c_costs):

                val, mu_safe, dist = self.safe_ac(state, current_cost= c_cost_)
                cost_r_val = self.cost_reviewer(state)

                cost_q_val = self.cost_critic(state, mu_safe.detach())

                # for actor
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2)

                if self.args.cost_sg_coeff:
                    # safeguard policy here, without baseline
                    _, sg_mu, sg_std = self.ac_model(state)
                    sg_val = self.sg_model(state)
                    unconst_dist = torch.distributions.Normal(sg_mu, sg_std)
                    sg_new_log_probs = unconst_dist.log_prob(action)

                    sg_ratio = (sg_new_log_probs - old_log_probs).exp()

                    sg_1 = sg_ratio * sg_adv
                    sg_2 = torch.clamp(sg_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * sg_adv
                    sg_loss = - torch.min(sg_1, sg_2)

                    violate_mask = torch.le(c_q_return_ + c_q_return_, self.args.d0).float().detach()

                    actor_loss =  violate_mask * actor_loss + (1. - violate_mask) * self.args.cost_sg_coeff * sg_loss
                # --------------------------------------------------------------
                actor_loss = actor_loss.mean()

                # add to the final  ac loss
                critic_loss = (return_ - val).pow(2).mean()

                ac_loss = (self.args.value_loss_coef * critic_loss) + \
                        (actor_loss) - (self.args.beta * entropy)

                self.ac_optimizer.zero_grad()
                ac_loss.backward()
                self.ac_optimizer.step()

                # for costs
                # for reviewer
                self.cost_reviewer.zero_grad()
                self.cost_critic.zero_grad()

                cost_review_loss = (c_r_return_ - cost_r_val).pow(2).mean()

                self.review_optimizer.zero_grad()
                cost_review_loss.backward()
                self.review_optimizer.step()

                cost_critic_loss = (c_q_return_ - cost_q_val).pow(2).mean()

                self.critic_optimizer.zero_grad()
                cost_critic_loss.backward()

                self.critic_optimizer.step()

                # clean everything just in case
                self.clear_models_grad()

                # extra step
                if self.args.cost_sg_coeff:
                    sg_val_loss = self.args.value_loss_coef * (sg_return_ - sg_val).pow(2).mean()
                    # self.sg_optimizer.zero_grad()
                    sg_val_loss.backward()
                    self.sg_optimizer.step()

                    # clean everything just in case
                    self.clear_models_grad()



    def clear_models_grad(self):
        # clean the grads
        self.ac_model.zero_grad()
        self.cost_critic.zero_grad()
        self.cost_reviewer.zero_grad()

    def safe_pi(self, state, current_cost=0.0, log=False):
        """
        take the action based on the current policy

        For a single state

        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        _, _, dist = self.safe_ac(state, current_cost)

        # sample from the new dist here
        action = dist.sample()

        self.clear_models_grad()

        return action.detach().squeeze(0).cpu().numpy()

    def pi(self, state):
        """
        take the action based on the current policy

        For a single state
        """
        with torch.no_grad():

            # convert the state to tensor
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)

            dist, val, r_val = self.model(state_tensor)

            action =  dist.sample()

        return action.detach().cpu().squeeze(0).numpy(), val, r_val


    def log_episode_stats(self, ep_reward, ep_constraint):
        """
        log the stats for environment 0 performance
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
        main PPO algo runs here
        """



        self.num_episodes = 0
        self.eval_steps = 0

        ep_reward = 0
        ep_constraint = 0
        ep_len = 0
        traj_len = 0

        start_time = time.time()


        # reset
        done = False
        state = self.envs.reset()


        prev_state = torch.FloatTensor(state).to(self.device)
        current_cost = torch.zeros(self.args.num_envs,1).float().to(self.device)


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

            log_probs   = []
            entropy     = 0

            sg_rewards = []
            sg_values = []

            c_costs = []


            for _ in range(self.args.traj_len):

                state = torch.FloatTensor(state).to(self.device)

                # calculate uncorrected mu and std
                val, mu_safe, dist = self.safe_ac(state, current_cost=current_cost)
                action = dist.sample()

                cost_q_val = self.cost_critic(state, mu_safe.detach())
                cost_r_val = self.cost_reviewer(state)

                next_state, reward, done, info = self.envs.step(action.cpu().numpy())

                # logging mode for only agent 1
                ep_reward += reward[0]
                ep_constraint += info[0][self.cost_indicator]

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()


                log_probs.append(log_prob)
                values.append(val)
                c_r_vals.append(cost_r_val)
                c_q_vals.append(cost_q_val)

                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                done_masks.append(torch.FloatTensor(1.0 - done).unsqueeze(1).to(self.device))
                begin_masks.append(torch.FloatTensor([(1.0 - ci['begin']) for ci in info]).unsqueeze(1).to(self.device))
                constraints.append(torch.FloatTensor([ci[self.cost_indicator] for ci in info]).unsqueeze(1).to(self.device))

                sg_rewards.append(torch.FloatTensor([-1.0 * ci[self.cost_indicator] for ci in info]).unsqueeze(1).to(self.device))
                sg_val = self.sg_model(state)
                sg_values.append(sg_val)

                prev_states.append(prev_state)
                states.append(state)
                actions.append(action)
                mus.append(mu_safe)

                prev_state = state
                state = next_state

                # update the current cost
                # if done flag is true for the current env, this implies that the next_state cost = 0.0
                # because the agent starts with 0.0 cost (or doesn't have access to it anyways)
                current_cost = torch.FloatTensor([(ci[self.cost_indicator] * (1.0 - di)) for ci,di in zip(info, done)]).unsqueeze(1).to(self.device)
                c_costs.append(current_cost)


                # hack to reuse the same code
                # iteratively add each done episode, so that can eval at regular interval
                for d_idx in range(done.sum()):

                    # do the logging for the first agent only here, only once
                    if done[0] and d_idx==0:
                        if self.num_episodes % self.args.log_every == 0:
                            self.log_episode_stats(ep_reward, ep_constraint)

                        # reset the rewards anyways
                        ep_reward = 0
                        ep_constraint = 0


                    self.num_episodes += 1

                    # eval the policy here after eval_every steps
                    if self.num_episodes % self.args.eval_every == 0:
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


            # calculate the returns
            next_state = torch.FloatTensor(next_state).to(self.device)

            next_value, next_mu_safe, next_dist = self.safe_ac(next_state, current_cost)
            returns = self.compute_gae(next_value, rewards, done_masks, values)

            next_c_value = self.cost_critic(next_state, next_mu_safe.detach())
            c_q_targets = self.compute_gae(next_c_value, constraints, done_masks, c_q_vals)

            prev_value = self.cost_reviewer(prev_states[0])
            c_r_targets = self.compute_review_gae(prev_value, constraints, begin_masks, c_r_vals)


            returns   = torch.cat(returns).detach()
            c_q_targets = torch.cat(c_q_targets).detach()
            c_r_targets = torch.cat(c_r_targets).detach()
            values = torch.cat(values).detach()
            log_probs = torch.cat(log_probs).detach()

            c_r_vals = torch.cat(c_r_vals).detach()
            c_q_vals = torch.cat(c_q_vals).detach()

            c_costs = torch.cat(c_costs).detach()

            states    = torch.cat(states)
            actions   = torch.cat(actions)


            advantage = returns - values



            # for sg
            sg_returns = self.compute_gae(self.sg_model(next_state), sg_rewards, done_masks, sg_values)
            sg_returns = torch.cat(sg_returns).detach()
            sg_values = torch.cat(sg_values).detach()


            sg_advantage = sg_returns - sg_values

            self.ppo_update(states, actions, log_probs, returns, advantage, sg_returns, sg_advantage, c_r_targets, c_q_targets, c_costs)


        # done with all the training

        # save the models and results
        self.save_models()


    def eval(self):
        """
        evaluate the current policy and log it
        """
        avg_reward = []
        avg_constraint = []


        for _ in range(self.args.eval_n):

            state = self.eval_env.reset()
            done = False

            ep_reward = 0
            ep_constraint = 0
            ep_len = 0
            start_time = time.time()
            current_cost = torch.FloatTensor([0.0]).to(self.device).unsqueeze(0)

            while not done:

                action  = self.safe_pi(state, current_cost=current_cost)

                next_state, reward, done, info = self.eval_env.step(action)
                ep_reward += reward
                ep_len += 1
                ep_constraint += info[self.cost_indicator]

                # update the state
                state = next_state

                current_cost = torch.FloatTensor([info[self.cost_indicator] * (1.0 - done)]).to(self.device).unsqueeze(0)


            avg_reward.append(ep_reward)
            avg_constraint.append(ep_constraint)

        self.eval_env.reset()

        return np.mean(avg_reward), np.mean(avg_constraint)



    def save_models(self):
        """create results dict and save"""
        torch.save(self.results_dict, os.path.join(self.args.out, 'results_dict.pt'))
        models = {
            "ac" : self.ac_model.state_dict(),
            "c_Q" : self.cost_critic.state_dict(),
            "c_R" : self.cost_reviewer.state_dict(),
            "sg" : self.sg_model.state_dict(),
            "eval_env" : copy.deepcopy(self.eval_env),
        }
        torch.save(models,os.path.join(self.args.out, 'models.pt'))



    def load_models(self):
        models = torch.load(os.path.join(self.args.out, 'models.pt'))
        self.ac_model.load_state_dict(models["ac"])
        self.cost_critic.load_state_dict(models["c_Q"])
        self.cost_reviewer.load_state_dict(models["c_R"])
        # self.critic.load_state_dict(models["critic"])
