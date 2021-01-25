from environment import Environment
import torch.multiprocessing as mp
import numpy as np
import random
import torch
import copy
import time


def init_hidden():
	init_h = torch.nn.Parameter(torch.zeros(1, 1, 256).type(torch.FloatTensor), requires_grad=False)
	init_c = torch.nn.Parameter(torch.zeros(1, 1, 256).type(torch.FloatTensor), requires_grad=False)
	return init_h, init_c


class MyAgent(mp.Process):

	def __init__(self, gnet, idx, global_ep, wins, total_rewards, res_queue, queue, g_que, gamma, up_step, bs, n_actions):
		super(MyAgent, self).__init__()
		self.daemon = True
		self.idx = idx
		self.global_ep, self.res_queue, self.queue, self.g_que, self.gamma, self.up_step, self.wins = global_ep, res_queue, queue, g_que, gamma, up_step, wins
		self.loss, self.vl, self.pl, self.cl, self.dl, self.grad_norm = 0, 0, 0, 0, 0, 0
		self.lnet = copy.deepcopy(gnet)
		self.rewards, self.personal_reward = 0, 0
		self.bs = bs
		self.n_actions = n_actions
		self.total_rewards = total_rewards
		self.lr = 0

	def step(self, reward, image, hc, vis_match):
		with self.total_rewards.get_lock():
			self.total_rewards.value += reward
		with self.global_ep.get_lock():
			self.global_ep.value += 1
		action, hc, logits, _, _ = self.lnet.choose_action(image, hc, vis_match)
		self.rewards += reward
		self.personal_reward += reward
		return action, hc, logits

	def push_and_pull(self, bd, s_, bs, ba, br, hc, bl, b_depth, b_match, vis_match_):
		self.queue.put([torch.cat(bs), torch.tensor(ba), s_, bd, hc, torch.tensor(br).unsqueeze(1), torch.stack(bl), torch.cat(b_depth), torch.stack(b_match), vis_match_])
		g_dict, self.loss, self.vl, self.pl, self.cl, self.dl, self.grad_norm, self.lr = self.g_que.get()
		self.lnet.load_state_dict(g_dict)

	def run(self):

		torch.manual_seed(self.idx)
		torch.cuda.manual_seed(self.idx)
		np.random.seed(self.idx)
		random.seed(self.idx)
		torch.backends.cudnn.deterministic = True

		env = Environment(9734 + self.idx)

		reward = 0
		sample_count = 0
		d = 0
		buffer_a, buffer_r, buffer_l, buffer_d, buffer_obs, buffer_i, buffer_hc, buffer_depth, buffer_match = (), (), (), (), (), (), (), (), ()
		(h, c) = init_hidden()
		hc = (h, c)
		n_step = 0
		obs, depth, vis_match = env.reset()		# RGB image, Depth image, visibility one-hot vector

		for p in self.lnet.parameters():
			p.requires_grad = False

		while self.global_ep.value < 1000000000:
			n_step += 1
			sample_count += 1

			action, hc, logits = self.step(reward, obs, hc, vis_match)
			reward, obs_, depth_, vis_match_ = env.env_step(action)		# reward, RGB image, Depth image, visibility one-hot vector

			if n_step % 900 == 0:
				d = True
				obs_, depth_, vis_match_ = env.reset()		# RGB image, Depth image, visibility one-hot vector

			if len(buffer_obs) < 500:
				buffer_obs += (obs,)
				buffer_depth += (depth,)
				buffer_a += (action,)
				buffer_r += (reward,)
				buffer_match += (vis_match,)
				buffer_l += (logits,)
				buffer_d += (d,)
				buffer_hc += (hc,)
			else:
				buffer_obs = buffer_obs[1:] + (obs,)
				buffer_depth = buffer_depth[1:] + (depth,)
				buffer_match = buffer_match[1:] + (vis_match,)
				buffer_a = buffer_a[1:] + (action,)
				buffer_r = buffer_r[1:] + (reward,)
				buffer_l = buffer_l[1:] + (logits,)
				buffer_d = buffer_d[1:] + (d,)
				buffer_hc = buffer_hc[1:] + (hc,)

			if sample_count == self.up_step or d:
				for _ in range(2):
					if len(buffer_obs) == 100:
						self.queue.put([torch.cat(buffer_obs), torch.tensor(buffer_a), obs_, buffer_d, buffer_hc[-100], torch.tensor(buffer_r).unsqueeze(1), torch.stack(buffer_l), torch.cat(buffer_depth), torch.stack(buffer_match), vis_match_])
					else:
						replay_index = torch.randint(101, len(buffer_obs), (1,))
						self.queue.put([torch.cat(buffer_obs[-replay_index: -replay_index + 100]), torch.tensor(buffer_a[-replay_index: -replay_index + 100]), buffer_obs[-replay_index + 101], buffer_d[-replay_index: -replay_index + 100], buffer_hc[-replay_index], torch.tensor(buffer_r[-replay_index: -replay_index + 100]).unsqueeze(1), torch.stack(buffer_l[-replay_index: -replay_index + 100]), torch.cat(buffer_depth[-replay_index: -replay_index + 100]), torch.stack(buffer_match[-replay_index: -replay_index + 100]), buffer_match[-replay_index + 101]])
				self.push_and_pull(buffer_d[-100:], obs_, buffer_obs[-100:], buffer_a[-100:], buffer_r[-100:], (h, c), buffer_l[-100:], buffer_depth[-100:], buffer_match[-100:], vis_match_)
				sample_count = 0
				if d:
					print('Agent %i, step %i' % (self.idx, n_step))
					self.res_queue.put([self.rewards, self.global_ep.value, self.loss / self.bs, self.vl / self.bs, self.pl / self.bs, self.cl / (self.bs * self.n_actions * self.up_step), self.dl / self.bs, self.grad_norm, self.lnet, self.total_rewards.value, self.wins.value, self.lr, self.personal_reward, self.idx])
					self.rewards, self.personal_reward = 0, 0
					hc = init_hidden()
					d = 0
				(h, c) = hc

			obs = obs_
			vis_match = vis_match_
			depth = depth_

		self.res_queue.put(None)
		self.queue.put(None)
		time.sleep(1)
		env.close_connection()
		print('Agent %i finished after %i steps.' % (self.idx, n_step))
