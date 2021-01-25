from environment import Environment
from torchvision import transforms
import torch.nn.functional as F
from obj_net import ObjNet
from model_nav import Net
from PIL import Image
import numpy as np
import torch
import time
import cv2


def init_hidden():
	if torch.cuda.is_available():
		init_h = torch.nn.Parameter(torch.zeros(1, 1, 256).type(torch.FloatTensor), requires_grad=False).cuda()
		init_c = torch.nn.Parameter(torch.zeros(1, 1, 256).type(torch.FloatTensor), requires_grad=False).cuda()
	else:
		init_h = torch.nn.Parameter(torch.zeros(1, 1, 256).type(torch.FloatTensor), requires_grad=False)
		init_c = torch.nn.Parameter(torch.zeros(1, 1, 256).type(torch.FloatTensor), requires_grad=False)
	return init_h, init_c


def get_random_goal(object_type):
	goal = Image.open('goals/goal_%i.jpg' % object_type)
	goal = np.moveaxis(np.array(goal), -1, 0)
	goal = torch.as_tensor(goal).type(torch.FloatTensor).view(3, 224, 224) / 255
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	if torch.cuda.is_available():
		goal = normalize(goal).cuda()
	else:
		goal = normalize(goal)
	return goal


def get_vis_match(vis_match_probs, vis_match_gt, mode):
	match = vis_match_gt
	if mode != 'gt':
		vis_match = torch.zeros_like(vis_match_gt)
		if mode == 'obj':
			vis_match_probs = vis_match_probs.view(6)
			vis_match_argmax = torch.argmax(vis_match_probs, dim=-1)
			if vis_match_argmax != 0:
				vis_match[vis_match_argmax - 1] = 1
		match = vis_match
	return match


def update_image(obs, cam):

	frame = np.moveaxis(obs.numpy(), 0, -1).astype(np.float64)

	obs = NORMALIZE(obs).view(1, 3, 224, 224)

	frame_norm = np.moveaxis(obs.view(3, 224, 224).numpy(), 0, -1).astype(np.float64)

	if torch.max(cam) != 0:
		cam = transforms.ToPILImage()(cam / torch.max(cam))
		cam = cam.resize((224, 224))
	else:
		cam = transforms.ToPILImage()(cam)
		cam = cam.resize((224, 224))

	alpha = np.zeros((224, 224, 3))
	alpha[:, :, 0] = cam
	alpha[:, :, 1] = cam
	alpha[:, :, 2] = cam

	frame_cam = (cv2.multiply(alpha / 255, frame)[..., ::-1] * 255).astype(np.uint8)
	frame = (frame * 255).astype(np.uint8)[..., ::-1]
	frame_norm = (frame_norm * 255).astype(np.uint8)[..., ::-1]

	return obs, frame, frame_cam, frame_norm


def load_networks(nav_path, obj_path):
	if torch.cuda.is_available():
		nav_net = Net(3).cuda()
		nav_net.load_state_dict(torch.load(nav_path))
		nav_net.eval()

		obj_net = ObjNet().cuda()
		obj_net.load_state_dict(torch.load(obj_path))
		obj_net.eval()
	else:
		nav_net = Net(3)
		nav_net.load_state_dict(torch.load(nav_path))
		nav_net.eval()

		obj_net = ObjNet()
		obj_net.load_state_dict(torch.load(obj_path))
		obj_net.eval()
	return nav_net, obj_net


TIME = 300
MAX_STEP = 1000000000000

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

nav_net, obj_net = load_networks(nav_path='path_to_nav_model', obj_path='path_to_obj_model')

env = Environment(9733, test=True)

ep_reward = 0
reward = 0
hc = init_hidden()
obs, depth, vis_match_gt = env.reset()

cam = torch.zeros((1, 9, 9))

obs, frame, frame_cam, frame_norm = update_image(obs, cam)

# object types = 0:Chair, 1:Monitor, 2:Trash, 3:Microwave, 4:Bottle, 5:Ball, 6:Lamp, 7:Plant, 8:Jar, 9:Can, 10:Extinguisher, 11:Boot
goal = get_random_goal(0).view(1, 3, 224, 224)

start = time.time()

step = 0
while ((time.time() - start) < TIME) and (reward != 1) and step < MAX_STEP:
	step += 1
	if torch.cuda.is_available():
		vis_match_probs = obj_net(obs.cuda(), goal)
		vis_match = get_vis_match(vis_match_probs, vis_match_gt, mode='obj')
		action, hc, logits, v, depth_pred = nav_net.choose_action(obs.cuda(), hc, vis_match.cuda(), train=False)
	else:
		vis_match_probs = obj_net(obs, goal)
		vis_match = get_vis_match(vis_match_probs, vis_match_gt, mode='obj')
		action, hc, logits, v, depth_pred = nav_net.choose_action(obs, hc, vis_match, train=False)
	probs = torch.clamp(F.softmax(logits, dim=-1), 0.000001, 0.999999)

	reward, obs_, depth_, vis_match_gt = env.env_step(action)
	ep_reward += reward

	if reward == 1:
		obs_, depth_, vis_match_ = env.reset()

	obs = obs_.cpu()
	depth = depth_

	obs, frame, frame_cam, frame_norm = update_image(obs, cam)

	cv2.imshow('frame', frame)
	cv2.waitKey(1)

env.close_connection()
cv2.destroyAllWindows()
