import math
import numpy as np
import os
import time

import torch
import random

from torch.utils.tensorboard import SummaryWriter

from common.utils import *


class BasePGAgent(object):
    """
    The base agent class for PG agents
    """
    def __init__(self,
                 args,
                 env,
                 writer_dir = None):
        """
        init agent
        """
        self.env = env
        self.args = args

        self.state_dim = env.observation_space.shape[0]

        # handle the discrete vs continuous case here
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.discrete = True
            self.action_dim = self.env.action_space.n
        else:
            # continuous
            self.discrete = False
            self.action_dim = env.action_space.shape[0]
            self.max_action = env.action_space.high[0]

        self.device = torch.device("cuda" if (torch.cuda.is_available() and  self.args.gpu) else "cpu")

        # set the seed the same as in the main launcher
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if self.args.gpu:
            torch.cuda.manual_seed(self.args.seed )

        self.writer = SummaryWriter(log_dir=writer_dir) if  writer_dir is not None else None
        self.total_steps = 0

        # models and optim go to the respective classes
        self.actor = None
        self.actor_optimizer = None


    def pi(self, state):
        """
        take the action based on the current policy
        """
        self.actor.eval()
        # can do this for faster implementation
        # with torch.no_grad():

        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        dist = self.actor(state).detach()
        action =  dist.sample().cpu().numpy()

        self.actor.train()

        return action



    def run(self):
        """

        """
        results_dict = {
            "train_rewards" : [],
            "eval_rewards" : [],
        }

        eval_steps = 0
        self.total_steps = 0
        num_episodes = 0

        # reset state and env
        # reset exploration porcess
        state = self.env.reset()
        done = False

        ep_reward = 0
        ep_len = 0

        start_time = time.time()


        for step in range(self.args.num_steps):
            # for step in range(self.args.step_per_epoch):
            # convert the state to tensor
            state_tensor = torch.from_numpy(state).float().to(self.device).view(-1, self.state_dim)

            # get the expl action
            action = self.exp_pi(state_tensor)
            # print(action)

            next_state, reward, done, _ = self.env.step(action)
            ep_reward += reward
            ep_len += 1
            self.total_steps += 1

            # hard reset done for rllab envs
            done = done or ep_len >= self.args.max_path_len

            # add the transition in the memory
            transition = Transition(state = state, action = action,
                                       reward = reward, next_state = next_state,
                                       done = float(done))

            self.memory.add(transition)


            # update the state
            state = next_state

            if done :

                # log
                results_dict["train_rewards"].append(ep_reward)
                self.writer.add_scalar("ep_reward", ep_reward, num_episodes)
                self.writer.add_scalar("ep_len", ep_len, num_episodes)
                self.writer.add_scalar("reward_step", ep_reward, self.total_steps)

                log(
                    'Num Episode {}\t'.format(num_episodes) + \
                    'Time: {:.2f}\t'.format(time.time() - start_time) + \
                    'E[R]: {:.2f}\t'.format(ep_reward) +\
                    'E[t]: {}\t'.format(ep_len) +\
                    'Step: {}\t'.format(self.total_steps) +\
                    'Epoch: {}\t'.format(self.total_steps // 10000) +\
                    'avg_train_reward: {:.2f}\t'.format(np.mean(results_dict["train_rewards"][-100:]))
                    )


                if self.exploration is not None and (
                        isinstance(self.exploration, PolyNoise) or isinstance(self.exploration, GyroPolyNoise) ):
                    rand_or_poly = np.asarray(self.exploration.rand_or_poly)
                    self.writer.add_scalar("poly_action_begin", np.sum(rand_or_poly[:100]), num_episodes)
                    self.writer.add_scalar("poly_action_end", np.sum(rand_or_poly[-100:]), num_episodes)
                    self.writer.add_scalar("poly_actions", np.sum(rand_or_poly), num_episodes)
                    results_dict["poly_or_rand"].append(rand_or_poly)

                if isinstance(self.exploration, GyroPolyNoise):
                    # get the radius of gyration per episode
                    g_history = np.asarray(self.exploration.g_history)
                    g_delta = float(self.exploration.avg_delta_g) / ep_len
                    results_dict["g_history"].append(g_history)
                    self.writer.add_scalar("avg_delta_g", g_delta, num_episodes)


                # plot the trajectory to the

                # converting the image to tensor here
                # convert the plot to image here and then to tensor
                # image = PIL.Image.open(plot_buf)
                # image = ToTensor()(image).unsqueeze(0)
                # add image to writer here
                # self.writer.add_image('Image', image, n_iter)


                # reset
                state = self.env.reset()
                done = False
                if self.exploration is not None:
                    self.exploration.reset()
                ep_reward = 0
                ep_len = 0
                start_time = time.time()

                # update counters
                num_episodes += 1


            # update here
            if self.memory.count > self.args.batch_size * 5:
                self.off_policy_update(update_steps)
                update_steps += 1


            # eval the policy here after 10k steps
            if self.total_steps % self.args.eval_every == 0:
                eval_ep, eval_len = self.eval()
                results_dict["eval_rewards"].append(eval_ep)

                log('----------------------------------------')
                log('Eval[R]: {:.2f}\t'.format(eval_ep) +\
                    'Eval[t]: {}\t'.format(eval_len) +\
                    'avg_eval_reward: {:.2f}\t'.format(np.mean(results_dict["eval_rewards"][-10:]))
                    )
                log('----------------------------------------')

                self.writer.add_scalar("eval_reward", eval_ep, eval_steps)

                eval_steps += 1


            if self.total_steps % self.args.checkpoint_interval == 0:
                self.save_models()

        # done with all the training

        # save the models
        self.save_models()

        # save the results
        torch.save(results_dict, os.path.join(self.args.out, 'results_dict.pt'))


    def eval(self):
        """
        evaluate the current policy and log it
        """
        avg_reward = []
        avg_len = []

        for _ in range(self.args.eval_n):

            state = self.env.reset()
            done = False

            ep_reward = 0
            ep_len = 0
            start_time = time.time()

            while not done:

                # convert the state to tensor
                state_tensor = torch.from_numpy(state).float().to(self.device).view(-1, self.state_dim)

                # get the policy action
                action = self.pi(state_tensor)

                next_state, reward, done, _ = self.env.step(action)
                ep_reward += reward
                ep_len += 1

                # update the state
                state = next_state

                done = done or ep_len >= self.args.max_path_len

            avg_reward.append(ep_reward)
            avg_len.append(ep_len)

        return np.mean(avg_reward), np.mean(avg_len)


    def save_models(self):
        """create results dict and save"""
        models = {
            "actor" : self.actor.state_dict(),
            "critic" : self.critic.state_dict(),
        }
        torch.save(models,os.path.join(self.args.out, 'models.pt'))


    def load_models(self):
        models = torch.load(os.path.join(self.args.out, 'models.pt'))
        self.actor.load_state_dict(models["actor"])
        self.critic.load_state_dict(models["critic"])
