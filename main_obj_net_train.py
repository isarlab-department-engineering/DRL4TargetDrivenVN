# from tensorboardX import SummaryWriter
from dataset import SimulationDataset
import torch.backends.cudnn as cudnn
from torch.utils import data
from obj_net import ObjNet
import numpy as np
import random
import torch
import os


cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
	model = ObjNet().cuda()
else:
	model = ObjNet()

batch_size = 128

dataloader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 16}

# writer = SummaryWriter()

model_params = model.parameters()
opt = torch.optim.SGD(model_params, momentum=0, lr=0.005, weight_decay=0)

training_set = SimulationDataset('train')
training_generator = data.DataLoader(training_set, **dataloader_params)
validation_set = SimulationDataset('val')
validation_generator = data.DataLoader(validation_set, **dataloader_params)

n_epoch = 0
tot_batch = 0
max_epochs = 50
val_loss_old = 9999

for epoch in range(max_epochs):
	n_batch = 0
	n_epoch += 1
	train_loss = 0
	model.train()
	print('Epoch: ' + str(n_epoch))
	for real_batch, unreal_batch_sim, unreal_batch_div, vis_match_batch_sim, vis_match_batch_div, _, _, _ in training_generator:
		n_batch += 1
		tot_batch += 1

		opt.zero_grad()

		if torch.cuda.is_available():
			real_batch, unreal_batch_sim, unreal_batch_div, vis_match_batch_sim, vis_match_batch_div = real_batch.cuda(), unreal_batch_sim.cuda(), unreal_batch_div.cuda(), vis_match_batch_sim.cuda(), vis_match_batch_div.cuda()
		else:
			real_batch, unreal_batch_sim, unreal_batch_div, vis_match_batch_sim, vis_match_batch_div = real_batch, unreal_batch_sim, unreal_batch_div, vis_match_batch_sim, vis_match_batch_div

		vis_match_1g_probs, vis_match_2g_probs, x3_act = model(unreal_batch_sim, real_batch, unreal_batch_div)

		vis_match_1g_pred = torch.argmax(vis_match_1g_probs, -1)
		m1g = torch.distributions.Categorical(vis_match_1g_probs)

		match_ids = np.where(vis_match_1g_pred.cpu() != 0)[0]
		if torch.cuda.is_available():
			match = torch.zeros_like(vis_match_1g_pred).type(torch.FloatTensor).cuda()
		else:
			match = torch.zeros_like(vis_match_1g_pred).type(torch.FloatTensor)
		match[match_ids] = 1

		if torch.cuda.is_available():
			loss_1g = (-m1g.log_prob(vis_match_batch_sim.cuda()) * (1 + abs(vis_match_1g_pred.type(torch.FloatTensor).cuda() - vis_match_batch_sim.type(torch.FloatTensor).cuda()) * match)).mean()
		else:
			loss_1g = (-m1g.log_prob(vis_match_batch_sim) * (1 + abs(vis_match_1g_pred.type(torch.FloatTensor) - vis_match_batch_sim.type(torch.FloatTensor)) * match)).mean()

		m2g = torch.distributions.Categorical(vis_match_2g_probs)

		loss_2g = -m2g.log_prob(vis_match_batch_div).mean()

		bs = int(x3_act.shape[0] / 3)
		true_batch = x3_act[:bs]
		goal_batch = x3_act[bs:2*bs]
		zero_batch = x3_act[2*bs:3*bs]
		if torch.cuda.is_available():
			loss3 = 1/2 * max(torch.tensor(0).type(torch.FloatTensor).cuda(), 0.1 + ((goal_batch - true_batch)**2).mean() - ((goal_batch - zero_batch)**2).mean())
		else:
			loss3 = 1/2 * max(torch.tensor(0).type(torch.FloatTensor), 0.1 + ((goal_batch - true_batch)**2).mean() - ((goal_batch - zero_batch)**2).mean())

		loss = loss_1g + loss_2g + loss3

		loss.backward()
		opt.step()

		train_loss += loss.cpu().item()

		# writer.add_scalar('loss_visual_match_train', loss.cpu(), tot_batch)
		# writer.add_scalar('loss_1g', loss_1g.cpu(), tot_batch)
		# writer.add_scalar('loss_2g', loss_2g.cpu(), tot_batch)
		# writer.add_scalar('loss_3', loss3.cpu(), tot_batch)

	# torch.save(model.state_dict(), 'path_to_model')
	print('train_loss = ' + str(train_loss / n_batch))

	val_loss = 0
	val_loss_1g = 0
	val_loss_2g = 0
	val_loss3 = 0
	val_accuracy = 0
	n_val_batch = 0

	for real_batch, unreal_batch_sim, unreal_batch_div, vis_match_batch_sim, vis_match_batch_div, _, _, _ in validation_generator:
		n_val_batch += 1

		if torch.cuda.is_available():
			real_batch, unreal_batch_sim, unreal_batch_div, vis_match_batch_sim, vis_match_batch_div = real_batch.cuda(), unreal_batch_sim.cuda(), unreal_batch_div.cuda(), vis_match_batch_sim.cuda(), vis_match_batch_div.cuda()
		else:
			real_batch, unreal_batch_sim, unreal_batch_div, vis_match_batch_sim, vis_match_batch_div = real_batch, unreal_batch_sim, unreal_batch_div, vis_match_batch_sim, vis_match_batch_div

		vis_match_1g_probs, vis_match_2g_probs, x3_act = model(unreal_batch_sim, real_batch, unreal_batch_div)

		vis_match_1g_pred = torch.argmax(vis_match_1g_probs, -1)
		m1g = torch.distributions.Categorical(vis_match_1g_probs)

		match_ids = np.where(vis_match_1g_pred.cpu() != 0)[0]
		if torch.cuda.is_available():
			match = torch.zeros_like(vis_match_1g_pred).type(torch.FloatTensor).cuda()
		else:
			match = torch.zeros_like(vis_match_1g_pred).type(torch.FloatTensor)
		match[match_ids] = 1

		if torch.cuda.is_available():
			loss_1g = (-m1g.log_prob(vis_match_batch_sim.cuda()) * (1 + abs(vis_match_1g_pred.type(torch.FloatTensor).cuda() - vis_match_batch_sim.type(torch.FloatTensor).cuda()) * match)).mean()
		else:
			loss_1g = (-m1g.log_prob(vis_match_batch_sim) * (1 + abs(vis_match_1g_pred.type(torch.FloatTensor) - vis_match_batch_sim.type(torch.FloatTensor)) * match)).mean()

		vis_match_2g_pred = torch.argmax(vis_match_2g_probs, -1)
		m2g = torch.distributions.Categorical(vis_match_2g_probs)

		if torch.cuda.is_available():
			loss_2g = -m2g.log_prob(vis_match_batch_div.cuda()).mean()
		else:
			loss_2g = -m2g.log_prob(vis_match_batch_div).mean()

		bs = int(x3_act.shape[0] / 3)
		true_batch = x3_act[:bs]
		goal_batch = x3_act[bs:2 * bs]
		zero_batch = x3_act[2 * bs:3 * bs]
		if torch.cuda.is_available():
			loss3 = 1/2 * max(torch.tensor(0).type(torch.FloatTensor).cuda(), 0.1 + ((goal_batch - true_batch)**2).mean() - ((goal_batch - zero_batch)**2).mean())
		else:
			loss3 = 1/2 * max(torch.tensor(0).type(torch.FloatTensor), 0.1 + ((goal_batch - true_batch)**2).mean() - ((goal_batch - zero_batch)**2).mean())

		loss = loss_1g + loss_2g + loss3

		for i in range(len(vis_match_1g_pred)):
			if vis_match_1g_pred[i] == vis_match_batch_sim[i]:
				val_accuracy += 1 / len(validation_set.real_img_ids) / 2
		for i in range(len(vis_match_2g_pred)):
			if vis_match_2g_pred[i] == vis_match_batch_div[i]:
				val_accuracy += 1 / len(validation_set.real_img_ids) / 2

		val_loss += loss.cpu().item()
		val_loss_1g += loss_1g.cpu().item()
		val_loss_2g += loss_2g.cpu().item()
		val_loss3 += loss3.cpu().item()

	print('val_loss = ' + str(val_loss / n_val_batch))
	print('val_acc = ' + str(round(val_accuracy, 4) * 100) + '%')
	print()

	# writer.add_scalar('loss_visual_match_val', val_loss / n_val_batch, n_epoch)
	# writer.add_scalar('accuracy_visual_match_val', val_accuracy, n_epoch)
	# writer.add_scalar('loss_1g_val', val_loss_1g, tot_batch)
	# writer.add_scalar('loss_2g_val', val_loss_2g, tot_batch)
	# writer.add_scalar('loss_3_val', val_loss3, tot_batch)

	if val_loss < val_loss_old:
		# torch.save(model.state_dict(), 'path_to_model')
		val_loss_old = val_loss
