"""
Creates the map as in Safe Lyp RL paper

Main code from here: https://github.com/junhyukoh/value-prediction-network/blob/master/maze.py
And visualization inspired from: https://github.com/WojciechMormul/rl-grid-world
"""


from PIL import Image
import numpy as np
import gym
from gym import spaces
import copy

import torch


# constants
BLOCK = 0
AGENT = 1
GOAL = 2
PIT = 3

# movemnent, can only move in 4 directions
DX = [0, 1, 0, -1]
DY = [-1, 0, 1, 0]


# for generation purposes
COLOR = [
        [44, 42, 60], # block - black
        [91, 255, 123], # agent - green
        [52, 152, 219], # goal - blue
        [255, 0, 0], # pit - red
        ]


def generate_maze(size=27, obstacle_density=0.3, gauss_placement=False, rand_goal=True):
    """
    returns the 4 rooms maze, start and goal
    """
    mx = size-2; my = size-2 # width and height of the maze
    maze = np.zeros((my, mx))

    #NOTE: padding here
    dx = DX
    dy = DY


    # define the start and the goal
    # start in (0, alpha)
    start_y, start_x =  my-1, mx-1

    # goal position   (24,24)
    if rand_goal:
        goal_y, goal_x = np.random.randint(0,my), 0
    else:
        goal_y, goal_x = my//2, 0


    # create the actual maze here
    # maze_tensor = np.zeros((size, size, len(COLOR)))
    maze_tensor = np.zeros((len(COLOR), size, size))

    # fill everything with blocks
    # maze_tensor[:,:,BLOCK] = 1.0
    maze_tensor[BLOCK,:,:] = 1.0

    # fit the generated maze
    # maze_tensor[1:-1, 1:-1, BLOCK] = maze
    maze_tensor[BLOCK, 1:-1, 1:-1] = maze

    # put the agent
    # maze_tensor[start_y+1][start_x+1][AGENT] = 1.0
    maze_tensor[AGENT][start_y+1][start_x+1]= 1.0

    # put the goal
    # maze_tensor[goal_y+1][goal_x+1][GOAL] = 1.0
    maze_tensor[GOAL][goal_y+1][goal_x+1] = 1.0


    # put the pits

    # create the the pits here
    for i in range(0, mx):
        for j in range(0, my):
            # pass if start or goal state
            if (i==start_x and j==start_y) or (i==goal_x and j==goal_y):
                pass

            # with prob p place the pit
            if np.random.rand() < obstacle_density:
                # maze_tensor[j+1][i+1][PIT] = 1.0
                maze_tensor[PIT][j+1][i+1] = 1.0


    return maze_tensor, [start_y+1, start_x+1], [goal_y+1, goal_x+1]




class PitWorld(gym.Env):
    """
    the env from safe lyp RL
    """
    def __init__(self,
                 size = 27,
                 max_step = 200,
                 per_step_penalty = -1.0,
                 goal_reward = 1000.0,
                 obstace_density = 0.3,
                 constraint_cost = 1.0,
                 random_action_prob = 0.005,
                 rand_goal = True,
                 one_hot_features=False):
        """
        create maze here
        """

        self.size = size
        self.dy = DY
        self.dx = DX
        self.random_action_prob = random_action_prob
        self.per_step_penalty = per_step_penalty
        self.goal_reward = goal_reward
        self.obstace_density = obstace_density
        self.max_step = max_step
        self.constraint_cost = constraint_cost
        self.one_hot = one_hot_features
        self.rand_goal = rand_goal

        # 4 possible actions
        self.action_space = spaces.Discrete(4)

        # create the maze
        self.init_maze, self.start_pos, self.goal_pos = generate_maze(size=self.size,
                                                                      obstacle_density=self.obstace_density,
                                                                      rand_goal = self.rand_goal)


        # observation space
        # TODO: 4d tensor or 3d image

        if self.one_hot is False:
            self.observation_space = spaces.Box(low=0, high=1,
                                            shape=self.init_maze.shape)
        else:
            self.observation_space = spaces.Box(low=0, high=1,
                                            shape=self.init_maze[AGENT].shape)


        self.reset()

    def reset(self):
        """
        """
        self.maze = copy.deepcopy(self.init_maze)
        self.agent_pos = copy.deepcopy(self.start_pos)

        self.t = 0
        self.episode_reward = 0
        self.done = False

        return self.observation()


    def observation(self):
        obs = np.array(self.maze, copy=True)

        if self.one_hot is False:
            # returns in the (channel, height, width) format
            obs = np.reshape(obs, (-1, self.size, self.size))
        else:
            obs = obs[AGENT].flatten()

        return obs


    def visualize(self, img_size=320):
        """
        create an image
        """
        img_maze = np.array(self.maze, copy=True).reshape(self.size, self.size, -1)
        #         currently for maze[y][x][color]
        my = self.maze.shape[0]
        mx = self.maze.shape[1]
        colors = np.array(COLOR, np.uint8)
        num_channel = self.maze.shape[2]
        vis_maze = np.matmul(self.maze, colors[:num_channel])
        vis_maze = vis_maze.astype(np.uint8)
        for i in range(vis_maze.shape[0]):
            for j in range(vis_maze.shape[1]):
                if self.maze[i][j].sum() == 0.0:
                    vis_maze[i][j][:] = int(255)
        image = Image.fromarray(vis_maze)
        return image.resize((int(float(img_size) * mx / my), img_size), Image.NEAREST)



    def to_string(self):
        my = self.maze.shape[1]
        mx = self.maze.shape[2]
        str = ''
        for y in range(my):
            for x in range(mx):
                if self.maze[BLOCK][y][x] == 1:
                    str += '#'
                elif self.maze[AGENT][y][x] == 1:
                    str += 'A'
                elif self.maze[GOAL][y][x] == 1:
                    str += 'G'
                elif self.maze[PIT][y][x] == 1:
                    str += 'x'
                else:
                    str += ' '
            str += '\n'
        return str


    def is_reachable(self, y, x):

        # if there is no block
        return self.maze[BLOCK][y][x] == 0

    def move_agent(self, direction):
        """
        part of forward model responsible for moving
        """

        #print("before:", self.agent_pos, self.maze[self.agent_pos[0]][self.agent_pos[1]][AGENT])

        y = self.agent_pos[0] + self.dy[direction]
        x = self.agent_pos[1] + self.dx[direction]

        if not self.is_reachable(y, x):
            return False

        # else move the agent

        self.maze[AGENT][self.agent_pos[0]][self.agent_pos[1]] = 0.0

        self.maze[AGENT][y][x] = 1.0
        self.agent_pos = [y, x]

        # moved the agent
        return True

    def step(self, action):
        assert self.action_space.contains(action)
        # assert self.done is False

        constraint = 0
        info = {}

        # increment
        self.t += 1

        # for stochasticity, overwrite random action
        if self.random_action_prob > 0 and np.random.rand() < self.random_action_prob:
            action = np.random.choice(range(len(DX)))

        # move the agent
        moved = self.move_agent(action)

        # default reward
        reward = self.per_step_penalty

        # if reached GOAL
        if self.maze[GOAL][self.agent_pos[0]][self.agent_pos[1]] == 1.0:
            reward = self.goal_reward
            self.done = True

        # if reached PIT
        if self.maze[PIT][self.agent_pos[0]][self.agent_pos[1]] == 1.0:
            constraint = self.constraint_cost

        # if max time steps reached
        if self.t >= self.max_step:
            self.done = True


        if self.t == 1:
            info['begin'] = True
        else:
            info['begin'] = False

        info['pit'] = constraint


        return self.observation(), reward, self.done, info




if __name__ == "__main__":

    env = PitWorld(size=14, one_hot_features=True, rand_goal=False)

    s = env.reset()

    print(env.maze.shape)
    print(env.maze[AGENT].shape)

    print("shape of obs:", s.shape)
    print("shape of obs:", env.reset().shape)
    print(env.to_string())
    print(env.agent_pos)

    # for a in range(4):
    for _ in range(50):

        print("0->u, 1->r, 2->d, 3->l")
        a = int(input())
        if a not in range(4):
            # go down
            if a == -1:
                s = env.reset()

            a = 0


        # print( DY[a], DX[a])
        s, r, d, info = env.step(a)

        # print(s)
        # print(r)
        print(env.to_string())
        print(env.agent_pos)
        print(env.t)
        print(env.done)
        print(info)
        # print(env.observation()[:,:,AGENT])
        # print(d)

    print("--------------------------------------------")
    s = env.reset()
    # print(env.to_string())
    print("--------------------------------------------")
