from torchvision import transforms
import torch


NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class Environment:

	def __init__(self, port, test=False):
		super(Environment, self).__init__()
		self.test = test
		self.x = 224
		self.y = 224

	def normalize(self, obs):
		if not self.test:
			obs = NORMALIZE(obs).view(1, 3, self.x, self.y)
		return obs

	def reset(self):
		vis_match = torch.zeros((5))

		obs_rgb = torch.zeros((3, self.x, self.y))
		obs_rgb = self.normalize(obs_rgb)
		obs_depth = torch.zeros((1, 1, 40, 80))

		return obs_rgb, obs_depth, vis_match

	def env_step(self, action):
		obs_rgb = torch.zeros((3, self.x, self.y))
		obs_rgb = self.normalize(obs_rgb)
		obs_depth = torch.zeros((1, 1, 40, 80))
		vis_match = torch.zeros((5))
		reward = torch.zeros((5))

		return reward[0], obs_rgb, obs_depth, vis_match

	def close_connection(self):
		pass
