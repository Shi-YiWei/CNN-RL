import matplotlib.pyplot as plt


def plot(episodes_list,mv_return,max_q_value_list,env_name):


	
	plt.plot(episodes_list, mv_return)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('DQN on {}'.format(env_name))
	plt.show()

	frames_list = list(range(len(max_q_value_list)))
	plt.plot(frames_list, max_q_value_list)
	plt.axhline(0, c='orange', ls='--')
	plt.axhline(10, c='red', ls='--')
	plt.xlabel('Frames')
	plt.ylabel('Q value')
	plt.title('DQN on {}'.format(env_name))
	plt.show()