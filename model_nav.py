import torch.nn.functional as F
import torch.nn as nn
import torch


CHANNELS = 3

D1 = 16
D2 = 32

DL = 256
DR = 256

NEW_SIZE = 81


class Net(nn.Module):
	def __init__(self, a_dim):
		super(Net, self).__init__()

		self.a_dim = a_dim
		self.goal = None

		self.conv1 = nn.Conv2d(in_channels=CHANNELS, out_channels=D1, kernel_size=8, stride=4, padding=0)
		self.bnc1 = torch.nn.GroupNorm(int(D1 / 2), D1)

		self.conv2 = nn.Conv2d(in_channels=D1, out_channels=D2, kernel_size=4, stride=2, padding=0)
		self.bnc2 = torch.nn.GroupNorm(int(D2 / 2), D2)

		self.deconv1 = nn.ConvTranspose2d(in_channels=D2, out_channels=D1, kernel_size=4, stride=2, padding=0)
		self.debnc1 = torch.nn.GroupNorm(int(D1 / 2), D1)

		self.deconv2 = nn.ConvTranspose2d(in_channels=D1, out_channels=1, kernel_size=8, stride=4, padding=0)

		self.lin = nn.Linear(NEW_SIZE * D2 + 5, DL)

		self.lstm = nn.LSTM(DL, DR)

		self.p = nn.Linear(DR, a_dim)
		self.v = nn.Linear(DR, 1)

		self.distribution = torch.distributions.Categorical

	def forward(self, x, hc, vis_match):

		x_84 = F.adaptive_avg_pool2d(x.view(-1, CHANNELS, x.shape[-2], x.shape[-1]), 84)

		x1 = self.bnc1(F.relu(self.conv1(x_84)))

		x2 = self.bnc2(F.relu(self.conv2(x1)))

		x2_ = x2.view(-1, D2 * NEW_SIZE)

		x3 = F.relu(self.lin(torch.cat([x2_, vis_match.view(-1, 5)], dim=1)))

		x4, hc = self.lstm(x3.view(-1, x.shape[-4], DL), hc)

		s0 = x4.shape[0]
		s1 = x4.shape[1]

		x4 = F.relu(x4.view(-1, DR))

		logits = self.p(x4).view(s0, s1, self.a_dim)
		values = self.v(x4).view(s0, s1)

		x1_depth = self.debnc1(F.relu(self.deconv1(x2)))
		x2_depth = F.relu(self.deconv2(x1_depth))
		depth_pred = torch.clamp(x2_depth[:, :, 22:62, 2:82], min=0, max=1)

		return logits.squeeze(), values, hc, depth_pred

	def set_goal(self, goal):
		self.goal = goal

	def choose_action(self, s, hc, vis_match, train=False):
		if not train:
			self.eval()
		logits, values, hc, depth_pred = self.forward(s, hc, vis_match)
		probs = torch.clamp(F.softmax(logits, dim=-1), 0.00001, 0.99999).data
		m = self.distribution(probs)
		action = m.sample().type(torch.IntTensor)

		return action, (hc[0].data, hc[1].data), logits, values, depth_pred

	def choose_action1(self, s, hc, vis_match):
		self.eval()
		logits, values, hc, _ = self.forward(s, hc, vis_match)
		probs = torch.clamp(F.softmax(logits, dim=-1), 0.00001, 0.99999).data
		return torch.argmax(probs, -1), (hc[0].data, hc[1].data), logits, values

	def get_weights(self):
		layers = [self.conv1, self.bnc1, self.conv2, self.bnc2, self.deconv1, self.debnc1, self.deconv2, self.lin_match, self.match_softmax, self.lin, self.lstm, self.p, self.v]
		weigths = []
		for layer in layers:
			tot = 0
			for p in layer.parameters():
				tot += p.sum()
			weigths.append(tot.item())
		return weigths
