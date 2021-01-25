import torch
if torch.cuda.is_available():
	import torch.backends.cudnn as cudnn
	cudnn.benchmark = True

if __name__ == '__main__':
	from my_agent import MyAgent
	from learner import Learner
	from tensorboardX import SummaryWriter
	from model_nav import Net
	import torch.multiprocessing as mp
	import numpy as np
	import random

	torch.manual_seed(0)
	torch.cuda.manual_seed(0)
	np.random.seed(0)
	random.seed(0)
	if torch.cuda.is_available():
		torch.backends.cudnn.deterministic = True

	N = 16
	GAMMA = 0.99
	UP_STEP = 100
	BS = 8
	LR = 0.0005
	ENTROPY_COST = 0.00025
	BASELINE_COST = 0.5
	N_ACTIONS = 3

	mp.set_start_method('spawn')

	# writer = SummaryWriter()

	gnet = Net(N_ACTIONS)

	global_ep, wins, tot_rewards = mp.Value('i', 0), mp.Value('i', 0), mp.Value('d', 0.)
	res_queue, queue, g_que = mp.Queue(), mp.Queue(), mp.Queue()

	learner = Learner(gnet, queue, g_que, N, global_ep, GAMMA, LR, UP_STEP, 1000000000, BS, ENTROPY_COST, BASELINE_COST)

	agents = [MyAgent(gnet, i, global_ep, wins, tot_rewards, res_queue, queue, g_que, GAMMA, UP_STEP, BS, N_ACTIONS) for i in range(N)]

	learner.start()

	[agent.start() for agent in agents]

	while 0:
		r = res_queue.get()
		if r is not None:
			writer.add_scalar('global_ep_r', r[0], r[1])
			writer.add_scalar('loss', r[2], r[1])
			writer.add_scalar('val_loss', r[3], r[1])
			writer.add_scalar('pol_loss', r[4], r[1])
			writer.add_scalar('H_loss', r[5], r[1])
			writer.add_scalar('depth_loss', r[6], r[1])
			writer.add_scalar('grad_norm', r[7], r[1])
			for name, param in r[8].named_parameters():
					writer.add_histogram(name, param, r[1])
			writer.add_scalar('total_reward', r[9], r[1])
			writer.add_scalar('wins', r[10], r[1])
			writer.add_scalar('lr', r[11], r[1])
			writer.add_scalar('personal_reward_%i' % r[13], r[12], r[1])
		else:
			break

	[agent.join() for agent in agents]
	learner.join()
