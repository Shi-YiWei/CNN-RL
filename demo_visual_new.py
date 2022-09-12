import numpy as np
import os
import sys
import pygame
import random
from sge.mazeenv import MazeEnv
from sge.utils import KEY

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm
import argparse

from arguments import get_args
from dqn import  DQN
from plot import  plot
from train_DQN import train_DQN





if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--game_name', default='mining',    help='MazeEnv/')
    parse.add_argument('--graph_param', default='train_1', help='difficulty of subtask graph')
    parse.add_argument('--seed', default=1, type=int,  help='random seed')
    parse.add_argument('--gamma', default=0.99, type=float,    help='discount factor')
    parse.add_argument('--nsteps', type=int, default=70, help='the steps to update the network')               # change





    # ========== ==========
    parse.add_argument('--lr', type=float, default=1e-2, help='learning rate of the algorithm')
    parse.add_argument('--num-episodes', type=int, default=20000, help='the number of episodes')
    parse.add_argument('--hidden-dim', type=int, default=128, help='the hidden dim')
    parse.add_argument('--epsilon', type=int, default=0.01, help='epsilon')
    parse.add_argument('--target-update', type=int, default=50, help='target_update')
    parse.add_argument('--buffer-size', type=int, default=5000, help='buffer-size')
    parse.add_argument('--minimal-size', type=int, default=1000, help='minimal_size')
    parse.add_argument('--batch-size', type=int, default=64, help='batch_size')
    # ========== ==========


    args = parse.parse_args()



    render_config = { 'vis': True, 'save': True, 'key_cheatsheet': False }

    env = MazeEnv(args.game_name, args.graph_param, args.nsteps, args.gamma, render_config)
    #env.reset(args.seed)

    action_set = env.get_actions()
    action_set = list(action_set)
    len_action_set = len(action_set)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



    #env = gym.make(args.env_name)
    #state_dim = env.observation_space.shape[0]
    action_dim = len_action_set  # 将连续动作分成11个离散动作

    random.seed(0)
    np.random.seed(0)
    #env.seed(0)
    torch.manual_seed(0)

    replay_buffer = rl_utils.ReplayBuffer(args.buffer_size)
    agent = DQN(action_dim, device, args)
    return_list, max_q_value_list = train_DQN(agent, env, replay_buffer, args)


    episodes_list = list(range(len(return_list)))
    mv_return = rl_utils.moving_average(return_list, 5)
    plot(episodes_list,mv_return,max_q_value_list,args.env_name)










