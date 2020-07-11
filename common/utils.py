import gym

from datetime import datetime
import sys

from collections import namedtuple

from rllab.misc import ext
# from envs.mujoco.gather.point_gather_env import PointGatherEnv
# from envs.mujoco.speeding.halfcheetah_speedlimit import SafeCheetahEnv

from envs.custom_mujoco.point_gather import PointGatherEnv
from envs.custom_mujoco.halfcheetah_speedlimit import SafeCheetahEnv
from envs.rllib_mujoco.circle.point_env_safe import SafePointEnv
from envs.grid.safety_gridworld import PitWorld


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state',
                                       'done',))

SarsaTransition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state',
                                       'next_action', 'done',))


FullTransition = namedtuple('FullTransition', ('state', 'action', 'reward', 'cost',
                                               'next_state', 'done', 'next_action',
                                               'prev_state', 'begin'))



def log(msg):
    print("[%s]\t%s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg))
    sys.stdout.flush()


def soft_update(target, source, tau):
    """
    do the soft parameter update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def arr_hash(arr):
    """
    takes a numpy array and returns it as bytes (immutable)
    """
    return arr.tobytes()



def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def create_gym_env(env_name = 'Swimmer-v1', init_seed=0):
    """
    create a copy of the environment and set seed
    """
    env = gym.make(env_name)
    env.seed(init_seed)

    return env


def create_safety_gym_env(env_name = "Safexp-PointGoal1-v0", init_seed=0):
    """
    create the safety_gym env
    https://github.com/openai/safety-gym

    :param env_name:
    :return:
    """
    env = gym.make(env_name)
    return env


def create_rllab_env(env_name, init_seed):
    """
    create the rllab env
    """
    env = eval(env_name)()
    ext.set_seed(init_seed)
    return env

def create_PointGatherEnv():
    """
    create the rllab env
    """
    env = PointGatherEnv(n_apples=2, n_bombs=8, bomb_cost=10.0)
    env._max_episode_steps = 15
    return env

def get_filename(args):
    """
    Filter what to print based on agent type, and return the filename for the models and logs

    :param args:
    :return: name (string)
    """
    # all the params that go in for the logging go over her
    toprint = ['agent', 'lr', 'batch_size', ]
    args_param = vars(args)

    if args.agent == "ppo":
        toprint += ['num_envs', 'ppo_updates', 'gae', 'clip', 'traj_len', 'beta', 'value_loss_coef', ]
    elif args.agent == "a2c":
        toprint += ['num_envs', 'traj_len', 'critic_lr', 'beta']
    elif args.agent == "sarsa":
        toprint += ['num_envs', 'traj_len', ]
    # bvf agents
    elif args.agent == "bvf-sarsa":
        toprint += ['num_envs', 'traj_len', 'cost_reverse_lr', 'cost_q_lr', ]
    # elif args.agent == "safe-ppo":
    #     toprint += ['num_envs', 'cost_reverse_lr', 'cost_q_lr', 'traj_len', 'beta',
    #                 'ppo_updates', 'gae', 'clip', 'value_loss_coef', ]
    elif args.agent == "bvf-ppo":
        toprint += ['num_envs', 'cost_reverse_lr', 'cost_q_lr', 'traj_len', 'beta', 'gae', 'clip',
                    'ppo_updates', 'd0', 'cost_sg_coeff', 'prob_alpha']
    elif args.agent == "safe-a2c":
        toprint += ['num_envs', 'cost_reverse_lr', 'cost_q_lr', 'traj_len', 'beta']
    # lyapunov agents
    elif args.agent == "lyp-a2c":
        toprint += ['num_envs', 'cost_q_lr', 'traj_len', 'beta', 'd0', 'cost_sg_coeff']
    elif args.agent == "lyp-sarsa":
        toprint += ['num_envs', 'traj_len', 'cost_q_lr', 'd0', 'cost_sg_coeff']
    elif args.agent == "lyp-ppo":
        toprint += ['num_envs', 'cost_q_lr', 'ppo_updates', 'traj_len', 'value_loss_coef', 'd0',
                    'cost_sg_coeff', 'prob_alpha']
    else:
        raise Exception("Not implemented yet!!")

    # for every safe agent
    if "safe" or "bvf" in args.agent:
        toprint += ['d0', 'cost_sg_coeff']

    # if early stopping for ppo
    if args.early_stop:
        toprint += ['early_stop']

    name = ''
    for arg in toprint:
        name += '_{}_{}'.format(arg, args_param[arg])


    return name


def create_env(args):
    """
    the main method which creates any environment
    """
    env = None

    if args.env_name == "pg":
        # create point gather envrionment
        # env = PointGatherEnv(n_apples=2,
        #                      n_bombs=8,
        #                      apple_reward=+10,
        #                      bomb_cost=10, #bomb inherently negative
        #                      max_ep_len = 15,
        #                      )
        env = create_PointGatherEnv()
    elif args.env_name == "pc":
        # create Point Circle
        env = SafePointEnv(circle_mode=True,
                           xlim=2.5,
                           abs_lim=True,
                           target_dist=15,
                           max_ep_len = 65,
                           )
    elif args.env_name == "cheetah":
        # create Point Circle as per Lyp PG
        env = SafeCheetahEnv(limit=1,
                             max_ep_len=200)
    elif args.env_name == "grid":
        # create the grid with pits env
        env = PitWorld(size = 14,
                       max_step = 200,
                       per_step_penalty = -1.0,
                       goal_reward = 1000.0,
                       obstace_density = 0.3,
                       constraint_cost = 10.0,
                       random_action_prob = 0.005,
                       one_hot_features=True,
                       rand_goal=True, # for testing purposes
                       )
    else:
        raise Exception("Not implemented yet")

    return env
