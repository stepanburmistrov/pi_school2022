import time
import sys
import argparse
import math

import cv2
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv
from pyglet.window import key
import torch
from torch import nn, optim
import torch.nn.functional as F






DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='loop_pedestrians')
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
args = parser.parse_args()


def solve(img):
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb_image, (256, 256))
    input_tensor = torch.from_numpy(rgb).unsqueeze(0)
    input_tensor = input_tensor/255
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    input_tensor = input_tensor.to(DEVICE,dtype=torch.float32)
    pred = model(input_tensor)
    st = (pred.detach().cpu().numpy()[0][0])*10-0.5
    sp = 0.1
    return sp, st



class Mod(nn.Module):
    def __init__(self):
        super(Mod, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(119072, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = torch.load('model.t')
model.to(DEVICE)
model.eval()


if args.env_name is None:
    env = DuckietownEnv(
        map_name='loop_empty',
        domain_rand=False,
        draw_bbox=False,
        user_tile_start=[1, 2]
    )
else:
    env = gym.make(args.env_name)

obs = env.reset()
env.render(mode="top_down")
total_reward = 0





while True:
    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad

    speed, steering = solve(obs)

    obs, reward, done, info = env.step([speed, steering])
    if done or env.step_count > 1000:
        env.reset()
        obs, reward, done, info = env.step([speed, steering])
        print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))
    total_reward += reward

    env.render(mode="top_down")
