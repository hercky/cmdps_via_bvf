import argparse


def get_args():
    """
    Utility for getting the arguments from the user for running the experiment

    :return: parsed arguments
    """

    # Env
    parser = argparse.ArgumentParser(description='collect arguments')

    parser.add_argument('--env-name', default='pg',
                        help="pg: point gather env\n"\
                             "cheetah: safe-cheetah env\n"\
                             "grid: grid world env\n"\
                            "pc: point circle env\n"\
                        )

    parser.add_argument('--agent', default='ppo',
                        help="the RL algo to use\n"\
                             "ppo: for ppo\n"\
                             "lyp-ppo: for Lyapnunov based ppo\n" \
                             "bvf-ppo: for Backward value function based ppo\n" \
                             "sarsa: for n-step sarsa\n" \
                             "lyp-sarsa: for Lyapnunov based sarsa\n"\
                             "bvf-sarsa: for Backward Value Function based sarsa"\
                        )
    parser.add_argument('--gamma', type=float, default=0.99, help="discount factor")
    parser.add_argument('--d0', type=float, default=5.0, help="the threshold for safety")

    # Actor Critic arguments goes here
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                            help="learning rate")
    parser.add_argument('--target-update-steps', type=int, default=int(1e4),
                        help="number of steps after to train the agent")
    parser.add_argument('--beta', type=float, default=0.001, help='entropy regularization')
    parser.add_argument('--critic-lr', type=float, default=1e-3, help="critic learning rate")
    parser.add_argument('--updates-per-step', type=int, default=1, help='model updates per simulator step (default: 1)')
    parser.add_argument('--tau', type=float, default=0.001, help='soft update rule for target netwrok(default: 0.001)')

    # PPO arguments go here
    parser.add_argument('--num-envs', type=int, default=10, help='the num of envs to gather data in parallel')
    parser.add_argument('--ppo-updates', type=int, default=1, help='num of ppo updates to do')
    parser.add_argument('--gae', type=float, default=0.95, help='GAE coefficient')
    parser.add_argument('--clip', type=float, default=0.2, help='clipping param for PPO')
    parser.add_argument('--traj-len', type=int, default= 10, help="the maximum length of the trajectory for an update")
    parser.add_argument('--early-stop', action='store_true',
                        help="early stop pi training based on target KL ")

    # Optmization arguments
    parser.add_argument('--lr', type=float, default=1e-2,
                            help="learning rate")
    parser.add_argument('--adam-eps', type=float, default=0.95, help="momenturm for RMSProp")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='size of minibatch for ppo/ ddpg update')

    # Safety params
    parser.add_argument('--cost-reverse-lr', type=float, default=5e-4,
                            help="reverse learning rate for reviewer")
    parser.add_argument('--cost-q-lr', type=float, default=5e-4,
                            help="reverse learning rate for critic")
    parser.add_argument('--cost-sg-coeff', type=float, default=0.0,
                            help="the coeeficient for the safe guard policy, minimizes the cost")
    parser.add_argument('--prob-alpha', type=float, default=0.6,
                        help="the kappa parameter for the target networks")
    parser.add_argument('--target', action='store_true',
                        help="use the target network based implementation")

    # Training arguments
    parser.add_argument('--num-steps', type=int, default=int(1e4),
                        help="number of steps to train the agent")
    parser.add_argument('--num-episodes', type=int, default=int(1e4),
                        help="number of episodes to train the agetn")
    parser.add_argument('--max-ep-len', type=int, default=int(15),
                        help="number of steps in an episode")

    # Evaluation arguments
    parser.add_argument('--eval-every', type=float, default=1000,
                        help="eval after these many steps")
    parser.add_argument('--eval-n', type=int, default=1,
                        help="average eval results over these many episodes")

    # Experiment specific
    parser.add_argument('--gpu', action='store_true', help="use the gpu and CUDA")
    parser.add_argument('--log-mode-steps', action='store_true',
                            help="changes the mode of logging w.r.r num of steps instead of episodes")
    parser.add_argument('--log-every', type=int, default=100,
                        help="logging schedule for training")
    parser.add_argument('--checkpoint-interval', type=int, default=1e5,
                        help="when to save the models")
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', type=str, default='/tmp/safe/models/')
    parser.add_argument('--log-dir', type=str, default="/tmp/safe/logs/")
    parser.add_argument('--reset-dir', action='store_true',
                        help="give this argument to delete the existing logs for the current set of parameters")

    args = parser.parse_args()

    return args

    # DQN specific arguments
    # parser.add_argument('--eps-decay-steps',type=int, default=10000,
    #                     help="eps decay rate in num of episodes (1/decay_rate)")
    # parser.add_argument('--prioritized', action='store_true',
    #                         help="If true use the prioritized buffer")
    # parser.add_argument('--beta-decay-steps',type=int, default=100,
    #                     help="eps decay rate in num of episodes (1/decay_rate)")
    # parser.add_argument('--beta-start', type=float, default=0.4,
    #                         help="the intial beta for the IS correction")

    # parser.add_argument('--dqn-target-update',type=int, default=1000,
    #                     help="number of steps after which to update the target dqn")
    # Safe_DQN stuff
    # parser.add_argument('--pi-update-steps',type=int, default=10,
    #                     help="number of times to run the inner optimization loop")

    # parser.add_argument('--max-grad-norm', type=float, default=5.0, help='max norm of gradients (default: 0.5)')
    # parser.add_argument('--ou-sigma', type=float, default=0.2, help="std for ou noise")
    # parser.add_argument('--replay-size', type=int, default=10000, help='size of replay buffer (default: 10000)')
