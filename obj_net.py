import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch


CHANNELS = 3
DL = 256

D1 = 512
D2 = 128
D3 = 16

RESNET_SIZE = 2048


class ObjNet(nn.Module):
	def __init__(self):
		super(ObjNet, self).__init__()

		self.resnet = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
		for p in self.resnet.parameters():
			p.requires_grad = False

		self.conv_1 = nn.Conv2d(in_channels=RESNET_SIZE, out_channels=D1, kernel_size=3, padding=1)
		self.bnc1 = nn.GroupNorm(int(D1 / 2), D1)
		self.conv_2 = nn.Conv2d(in_channels=D1, out_channels=D2, kernel_size=3, padding=1)
		self.bnc2 = nn.GroupNorm(int(D2 / 2), D2)
		self.conv_3 = nn.Conv2d(in_channels=D2, out_channels=D3, kernel_size=3, padding=1)
		self.bnc3 = nn.GroupNorm(int(D3 / 2), D3)

		self.conv_4 = nn.Conv2d(in_channels=D3, out_channels=D3, kernel_size=3, padding=1)
		self.bnc4 = nn.GroupNorm(int(D3 / 2), D3)
		self.conv_5 = nn.Conv2d(in_channels=D3, out_channels=D3, kernel_size=3, padding=0)
		self.bnc5 = nn.GroupNorm(int(D3 / 2), D3)

		self.lin_match = nn.Linear(D3 * 5 * 5 * 2, DL)
		self.match_softmax = nn.Linear(DL, 6)

	def forward(self, x1, goal):

		self.resnet.eval()

		x_ = torch.cat([x1, goal], dim=0)

		x3_r = self.resnet(x_)
		x3_act = self.bnc1(F.relu(self.conv_1(x3_r)))
		x3_act = self.bnc2(F.relu(self.conv_2(x3_act)))
		x3_act = self.bnc3(F.relu(self.conv_3(x3_act)))

		x_1g = x3_act[:x1.shape[0] + goal.shape[0]]
		x_1g = self.bnc4(F.relu(self.conv_4(x_1g)))
		x_1g = self.bnc5(F.relu(self.conv_5(x_1g)))

		x_1 = x_1g[:x1.shape[0]].view(x1.shape[0], D3 * 5 * 5)
		x_g = x_1g[x1.shape[0]:].view(goal.shape[0], D3 * 5 * 5)
		x_1g = torch.cat([x_1, x_g], dim=1)

		x_1g = F.relu(self.lin_match(x_1g))

		vis_match_1g = torch.clamp(F.softmax(self.match_softmax(x_1g), dim=-1), 0.00001, 0.99999)

		return vis_match_1g

	@staticmethod
	def get_weights(layer):
		tot = 0
		for p in layer.parameters():
			tot += p.sum()
		return tot.item()
