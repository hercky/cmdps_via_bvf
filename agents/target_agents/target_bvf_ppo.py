# The code is built using code from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

import numpy as np
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import copy

from common.utils import *
from common.multiprocessing_envs import SubprocVecEnv
from agents.base_agent import BasePGAgent


from models.safe_mujoco_model import SafeUnprojectedActorCritic, Cost_Reviewer, Cost_Critic


class TargetBVFPPOAgent(BasePGAgent):
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

        # unconstrainted
        self.ac_model = SafeUnprojectedActorCritic(state_dim=self.state_dim,
                                 action_dim=self.action_dim,
                                 action_bound=self.ub_action,
                                 discrete=self.discrete).to(self.device)


        # Q_D(s,a) and \cev{V}_D(s)
        self.cost_reviewer = Cost_Reviewer(state_dim=self.state_dim).to(self.device)
        self.cost_critic = Cost_Critic(state_dim=self.state_dim,
                                       action_dim=self.action_dim).to(self.device)

        self.ac_optimizer = optim.Adam(self.ac_model.parameters(), lr=self.args.lr)
        self.review_optimizer = optim.Adam(self.cost_reviewer.parameters(), lr=self.args.cost_reverse_lr)
        self.critic_optimizer = optim.Adam(self.cost_critic.parameters(), lr=self.args.cost_q_lr)

        # safe guard policy equivalent
        # V_D(s)
        self.cost_value = Cost_Reviewer(state_dim=self.state_dim).to(self.device)
        self.cost_value_optimizer = optim.Adam(self.cost_value.parameters(), lr=self.args.cost_q_lr)


        # --------------
        # Baselines (or targets for storing the last iterate)
        # ------------

        # for reducing the cost
        self.baseline_ac_model = SafeUnprojectedActorCritic(state_dim=self.state_dim,
                                                            action_dim=self.action_dim,
                                                            action_bound=self.ub_action,
                                                            discrete=self.discrete).to(self.device)
        self.baseline_cost_critic = Cost_Critic(state_dim=self.state_dim,
                                                action_dim=self.action_dim).to(self.device)
        self.baseline_cost_value = Cost_Reviewer(state_dim=self.state_dim).to(self.device)
        self.baseline_cost_reviewer = Cost_Reviewer(state_dim=self.state_dim).to(self.device)



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
        elif self.args.env_name in ["pc", "cheetah"]:
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

    def ppo_iter(self, states, actions, log_probs,
                 returns, advantage,
                 c_q_returns, c_advantages,
                 c_r_returns, c_costs):
        """
        samples a minibatch from the collected experiren
        """
        batch_size = states.size(0)

        for _ in range(batch_size // self.mini_batch_size):
            # get a minibatch
            rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], \
                    advantage[rand_ids, :], c_q_returns[rand_ids, :], c_advantages[rand_ids, :], c_r_returns[rand_ids, :], c_costs[rand_ids, :]


    def safe_ac(self, state: torch.Tensor, current_cost: torch.Tensor):
        """
        :param state:
        :param current_cost: the cost at the current state d(x)
        :return:
        """
        # calculate uncorrected mu and std
        val, mu, std = self.ac_model(state)

        # use the baseline to do the projection
        _, mu_baseline, std_baseline = self.baseline_ac_model(state)

        # get the gradient Q_D wrt  mu
        q_D_baseline = self.baseline_cost_critic(state, mu_baseline)

        # calculate the
        gradients = torch.autograd.grad(outputs=q_D_baseline,
                                        inputs=mu_baseline,
                                        grad_outputs=torch.ones(q_D_baseline.shape).to(self.device),
                                        only_inputs=True,  # we don't want any grad accumulation
                                        )[0]
        # get  the epsilon here
        # TODO: this can be current also
        v_D_baseline = self.baseline_cost_value(state)

        # calculate eps(x)
        eps = (self.args.d0 + current_cost - (v_D_baseline +  q_D_baseline)).detach()

        # grad_sq
        grad_sq = torch.bmm(gradients.unsqueeze(1), gradients.unsqueeze(2)).squeeze(2)

        mu_theta = mu
        g_pi_diff = torch.bmm(gradients.unsqueeze(1), (mu_theta - mu_baseline).unsqueeze(2)).squeeze(2)

        eta = self.args.prob_alpha

        # get the optim lambda
        lambda_ = F.relu(((1 - eta) * g_pi_diff - eps) / grad_sq)

        # This step acts as a correction layer based on the baseline
        proj_mu = ((1.0 - eta) * mu) + (eta * mu_baseline) - lambda_ * gradients

        # project the correction in the range of action
        proj_mu = torch.tanh(proj_mu) * self.ub_action

        # create the new dist for sampling here
        dist = torch.distributions.Normal(proj_mu, std)

        return val, proj_mu, dist



    def ppo_update(self, states, actions, log_probs, returns, advantages,
                   c_q_returns, c_advantages,
                   c_r_returns, c_costs,
                   ):
        """
        does the actual PPO update here
        """
        for _ in range(self.ppo_epochs):

            for state, action, old_log_probs, return_, advantage, c_q_return_, c_advantage, c_r_return_, c_cost_ in self.ppo_iter(
                    states, actions, log_probs, returns, advantages, c_q_returns, c_advantages, c_r_returns, c_costs):

                # ------- Do all the estimations here
                val, mu, dist = self.safe_ac(state, current_cost=c_cost_)

                # Q_D
                cost_q_val = self.cost_critic(state, mu.detach())
                # V_D
                cost_val = self.cost_value(state)
                # \cev{V}_D
                cost_r_val = self.cost_reviewer(state)

                # ------ Losses ------

                # for actor
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                actor_loss = - torch.min(surr1, surr2)

                # ---- to use safe guard policy
                if self.args.cost_sg_coeff > 0:
                    # opposite of rewards (minimize cost)
                    actor_loss += self.args.cost_sg_coeff * (+1.0) * (ratio * c_advantage).mean()

                # regular actor loss
                actor_loss = actor_loss.mean()

                # for critic
                critic_loss = (return_ - val).pow(2).mean()

                # AC loss
                ac_loss = (self.args.value_loss_coef * critic_loss) + \
                          (actor_loss) - (self.args.beta * entropy)


                # Q_D loss
                cost_critic_loss = self.args.value_loss_coef * (c_q_return_ - cost_q_val).pow(2).mean()

                # V_D loss
                cost_val_loss = self.args.value_loss_coef * (c_q_return_ - cost_val).pow(2).mean()

                # \cev{V}_D loss
                cost_review_loss = self.args.value_loss_coef * (c_r_return_ - cost_r_val).pow(2).mean()

                # ------- Optim updates
                self.ac_optimizer.zero_grad()
                ac_loss.backward()
                self.ac_optimizer.step()

                # for Q_D
                self.cost_critic.zero_grad()
                cost_critic_loss.backward()
                self.critic_optimizer.step()

                # for V_D
                self.cost_value.zero_grad()
                cost_val_loss.backward()
                self.cost_value_optimizer.step()

                # for \cev{V}_D
                self.review_optimizer.zero_grad()
                cost_review_loss.backward()
                self.review_optimizer.step()



    def clear_models_grad(self):
        # clean the grads
        self.ac_model.zero_grad()
        self.cost_critic.zero_grad()
        self.cost_reviewer.zero_grad()
        self.cost_value.zero_grad()

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

            # saving the model
            if self.num_episodes % self.args.checkpoint_interval == 0:
                self.save_models()


    def check_termination(self, ):
        """
        if either num of steps or episodes exceed max threshold return the signal to terminate
        :param done:
        :return:
        """
        return self.num_episodes >= self.args.num_episodes


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

        prev_state = torch.FloatTensor(state).to(self.device)
        current_cost = torch.zeros(self.args.num_envs, 1).float().to(self.device)

        envs_traj_len = torch.zeros(self.args.num_envs, 1).float().to(self.device)

        while self.num_episodes < self.args.num_episodes:

            values = []
            c_values = []
            c_r_vals = []
            c_costs = []

            states = []
            actions = []
            prev_states = []
            rewards = []
            done_masks = []
            begin_masks = []
            constraints = []

            log_probs = []
            entropy = 0

            # the length of the n-step for updates
            for _ in range(self.args.traj_len):

                state = torch.FloatTensor(state).to(self.device)
                # calculate uncorrected mu and std
                val, mu, dist = self.safe_ac(state, current_cost=current_cost)
                action = dist.sample()

                # get the estimates regarding the costs
                cost_r_val = self.cost_reviewer(state)
                cost_val = self.cost_value(state)

                next_state, reward, done, info = self.envs.step(action.cpu().numpy())

                # logging mode for only agent 1
                self.ep_reward += reward[0]
                self.ep_constraint += info[0][self.cost_indicator]

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(val)
                c_r_vals.append(cost_r_val)
                c_values.append(cost_val)

                # add 1 to envs_ep_len
                #   if step = 1 ==> begin is True else False
                #   if done, reset the len/step to 0
                envs_traj_len += 1.
                begin_mask_ = (envs_traj_len == 1.).to(dtype=torch.float, device=self.device)
                envs_traj_len *= torch.FloatTensor(1.0 - done).unsqueeze(1).to(self.device)

                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                done_masks.append(torch.FloatTensor(1.0 - done).unsqueeze(1).to(self.device))
                begin_masks.append(begin_mask_)

                # add the costs
                constraints.append(
                    torch.FloatTensor([ci[self.cost_indicator] for ci in info]).unsqueeze(1).to(self.device))

                prev_states.append(prev_state)
                states.append(state)
                actions.append(action)

                # update the states
                prev_state = state
                state = next_state

                # update the current cost d(x)
                # if done flag is true for the current env, this implies that the next_state cost = 0.0
                # because the agent starts with 0.0 cost
                current_cost = torch.FloatTensor(
                    [(ci[self.cost_indicator] * (1.0 - di)) for ci, di in zip(info, done)]).unsqueeze(1).to(self.device)
                c_costs.append(current_cost)

                self.total_steps += self.args.num_envs
                # Do logging here depending on step vs episodes

                self.episode_based_logger(done)

                if self.check_termination():
                    break


            # calculate the returns
            next_state = torch.FloatTensor(next_state).to(self.device)

            # doesn't need safety projection for the value function
            next_value, _, _ = self.ac_model(next_state)
            returns = self.compute_gae(next_value, rewards, done_masks, values)

            # same for constraints
            next_c_value = self.cost_value(next_state)
            c_q_returns = self.compute_gae(next_c_value, constraints, done_masks, c_values)

            # reverse estimates
            prev_value = self.cost_reviewer(prev_states[0])
            c_r_targets = self.compute_review_gae(prev_value, constraints, begin_masks, c_r_vals)

            # regular PPO updates
            returns = torch.cat(returns).detach()
            values = torch.cat(values).detach()
            log_probs = torch.cat(log_probs).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)

            advantages = returns - values

            c_q_returns = torch.cat(c_q_returns).detach()
            c_values = torch.cat(c_values).detach()

            c_advantages = c_q_returns - c_values

            c_r_targets = torch.cat(c_r_targets).detach()
            c_costs = torch.cat(c_costs).detach()

            # NOTE: update the baseline policies
            # for a policy in an env (\theta_k), baseline is (\theta_{k-1})
            #  baseline is the old policy
            self.baseline_ac_model.load_state_dict(self.ac_model.state_dict())
            self.baseline_cost_critic.load_state_dict(self.cost_critic.state_dict())
            self.baseline_cost_value.load_state_dict(self.cost_value.state_dict())
            self.baseline_cost_reviewer.load_state_dict(self.cost_reviewer.state_dict())

            # update the current model
            self.ppo_update(states, actions, log_probs, returns, advantages,
                            c_q_returns, c_advantages,
                            c_r_targets, c_costs)



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
                # get the policy action
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                _, _, dist = self.safe_ac(state, current_cost=current_cost)

                # sample from the new dist here
                action = dist.sample()
                action = action.detach().squeeze(0).cpu().numpy()

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
        models = {
            "ac": self.ac_model.state_dict(),
            "c_Q": self.cost_critic.state_dict(),
            "c_R": self.cost_reviewer.state_dict(),
        }
        torch.save(models,os.path.join(self.args.out, 'models.pt'))
        torch.save(self.results_dict, os.path.join(self.args.out, 'results_dict.pt'))


    def load_models(self):
        models = torch.load(os.path.join(self.args.out, 'models.pt'))
        self.ac_model.load_state_dict(models["ac"])
        self.cost_critic.load_state_dict(models["c_Q"])
        self.cost_reviewer.load_state_dict(models["c_R"])
