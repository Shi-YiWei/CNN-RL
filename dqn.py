import random
import numpy as np
import torch
import torch.nn.functional as F
import rl_utils

from qnet import ConvolutionalQnet

class DQN:
    ''' DQN算法,包括Double DQN '''
    def __init__(self, action_dim, device, args,  dqn_type='VanillaDQN'):

        self.action_dim = action_dim
        # self.q_net = Qnet(state_dim, args.hidden_dim, self.action_dim).to(device)
        self.q_net = ConvolutionalQnet( self.action_dim).to(device)

        # self.target_q_net = Qnet(state_dim, args.hidden_dim, self.action_dim).to(device)
        self.target_q_net = ConvolutionalQnet( self.action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.target_update = args.target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self, state):

        if np.random.random() < self.epsilon:
            
            action = np.random.randint(self.action_dim-1)
        else:
            
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            #####
            action = self.q_net(state).argmax().item()         
            #####
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to( self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1) 
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else: # DQN的情况
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

        #print("update() completed!")







