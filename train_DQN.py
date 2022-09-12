import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm




def train_DQN(agent, env, replay_buffer, args):
    num_episodes = args.num_episodes
    minimal_size = args.minimal_size
    batch_size = args.batch_size

    return_list = []
    max_q_value_list = []
    max_q_value = 0

    action_set = env.get_actions()
    action_set = list(action_set)
    # print("action_set:",action_set)

    # the number of the round
    for i in range(10):
        with tqdm(total=int(num_episodes / 100), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 100)):
                episode_return = 0
                # change
                state = env.reset(args.seed)
                done = False
                while not done:

                    action_index = agent.take_action(state) # number index  not string

                    action = action_set[action_index]

                    # print("action index:",action_index,", action:", action)

                    max_q_value = agent.max_q_value( state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值

                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action_index, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list, max_q_value_list


