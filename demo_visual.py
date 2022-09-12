import numpy as np
import os
import sys
import pygame
import random
from sge.mazeenv import MazeEnv
from sge.utils import KEY

###################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default='mining',    help='MazeEnv/')
    parser.add_argument('--graph_param', default='train_1', help='difficulty of subtask graph')
    parser.add_argument('--game_len', default=70,   type=int, help='episode length')
    parser.add_argument('--seed', default=1, type=int,  help='random seed')
    parser.add_argument('--gamma', default=0.99, type=float,    help='discount factor')
    args = parser.parse_args()

    render_config = { 'vis': True, 'save': True, 'key_cheatsheet': False }
    env = MazeEnv(args.game_name, args.graph_param, args.game_len, args.gamma, render_config)

    env.reset(args.seed)

    action_set = env.get_actions()
    action_set = list(action_set)


    step = 0
    done = False

    while not done:
        print("action_set:",type(action_set))

        a_number = random.randint(0,len(action_set)-1)

        action = action_set[a_number]
        # action = random.sample(list(action_set), 1)[0]
        print("action:",action)

        state, rew, done, info = env.step(action)

        string = 'Step={:02d}, Action={}, Reward={:.2f}, Done={}'
        print(string.format(step, action, rew, done))
        step += 1
        # pygame.time.wait(100)
