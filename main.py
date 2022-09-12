import random
import gym
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


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# num_episodes = 200
# hidden_dim = 128

# epsilon = 0.01
# target_update = 50
# buffer_size = 5000
# minimal_size = 1000



args = get_args()

env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]

action_set = env.get_actions()


action_dim =   len(action_set) # 

random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)

replay_buffer = rl_utils.ReplayBuffer(args.buffer_size)
agent = DQN(action_dim, device, args)
return_list, max_q_value_list = train_DQN(agent, env, replay_buffer, args)


episodes_list = list(range(len(return_list)))
mv_return = rl_utils.moving_average(return_list, 5)
plot(episodes_list,mv_return,max_q_value_list,args.env_name)

