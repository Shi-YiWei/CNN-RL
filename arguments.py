import argparse


def get_args():
    parse = argparse.ArgumentParser()


    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL')
    parse.add_argument('--env-name', type=str, default='Pendulum-v1', help='the environment name')
    parse.add_argument('--lr', type=float, default=1e-2, help='learning rate of the algorithm')
    parse.add_argument('--num-episodes', type=int, default=200, help='the number of episodes')
    parse.add_argument('--hidden-dim', type=int, default=128, help='the hidden dim')
    parse.add_argument('--epsilon', type=int, default=0.01, help='epsilon')
    parse.add_argument('--target-update', type=int, default=50, help='target_update')
    parse.add_argument('--buffer-size', type=int, default=5000, help='buffer-size')
    parse.add_argument('--minimal-size', type=int, default=1000, help='minimal_size')
    parse.add_argument('--batch-size', type=int, default=64, help='batch_size')
    args = parse.parse_args()

    return args





