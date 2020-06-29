from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


class SafeCheetahEnv(HalfCheetahEnv):
    def __init__(self, limit=1, max_ep_len=200):
        """
        low_limit, high_limit: define the optimal range in which the agent receives the maximum reward

        slack: the agent dies after high_limit + slack
        """
        # super().__init__()
        self._time_step = 0
        self.limit = limit
        self.max_ep_len = max_ep_len

        HalfCheetahEnv.__init__(self)


    def step(self, a):

        (s, r, done, info) = super(SafeCheetahEnv, self).step(a)

        # breakpoint()
        self._time_step += 1


        # xvelocity is in the info as reward_run
        cost = 0.0
        if abs(info['reward_run']) > 1.0:
            cost = 1.0
        info['cost'] = cost

        if self._time_step >= self.max_ep_len:
            done = True

        if self._time_step == 1:
            info['begin'] = True
        else:
            info['begin'] = False

        return (s, r, done, info)

    def reset(self):
        """
        """
        s = super(SafeCheetahEnv, self).reset()
        self._time_step = 0

        return s
