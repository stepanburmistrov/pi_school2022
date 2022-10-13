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

        # return F.log_softmax(x, dim=1)


model = Mod()
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-2)


def getData(bgr_image):
    rgb_image = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, (256, 256))
    h, w, _ = rgb_image.shape
    w_min = (26, 0, 118)
    w_max = (146, 32, 221)
    y_min = (87, 74, 158)
    y_max = (96, 255, 229)
    region_vertices = [(0, h), (0, h * 3 / 5), (w / 8, h / 2), (w * 7 / 8, h / 2), (w, h * 3 / 5), (w, h)]
    mask_cropping = np.zeros_like(rgb_image)
    cv2.fillPoly(mask_cropping, np.array([region_vertices], np.int32), (255, 255, 255))
    cropped_image = cv2.bitwise_and(rgb_image, mask_cropping)
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)

    kernel = np.ones((5, 5), 'uint8')

    img_bin_white = cv2.inRange(hsv_image, w_min, w_max)
    img_bin_white = cv2.dilate(img_bin_white, kernel, iterations=3)
    img_bin_white = cv2.erode(img_bin_white, kernel, iterations=3)
    img_bin_white = cv2.erode(img_bin_white, kernel, iterations=1)
    img_bin_white = cv2.dilate(img_bin_white, kernel, iterations=1)

    img_bin_yellow = cv2.inRange(hsv_image, y_min, y_max)
    img_bin_yellow = cv2.dilate(img_bin_yellow, kernel, iterations=3)
    img_bin_yellow = cv2.erode(img_bin_yellow, kernel, iterations=3)
    img_bin_yellow = cv2.erode(img_bin_yellow, kernel, iterations=1)
    img_bin_yellow = cv2.dilate(img_bin_yellow, kernel, iterations=1)

    dual_mask = np.zeros((h, w, 3), dtype='uint8')
    dual_mask[img_bin_yellow > 0] = [0, 255, 255]
    dual_mask[img_bin_white > 0] = [255, 255, 255]

    ready_to_tensor = np.zeros((h, w), dtype='double')
    ready_to_tensor[img_bin_yellow > 0] = 0.5
    ready_to_tensor[img_bin_white > 0] = 1
    # ready_to_tensor = cv2.resize(ready_to_tensor,(32,32))

    return cropped_image, img_bin_white, img_bin_yellow, dual_mask, ready_to_tensor


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
rgb, bin_white, bin_yellow, mask, to_tensor = getData(obs)
env.render(mode="top_down")
total_reward = 0
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-8)
reward = 0.0




@env.unwrapped.window.event
def on_key_press(symbol, modifiers):

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()

    if symbol == key.SPACE:
        print('Model saved')
        torch.save(model, 'model.t')

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


while True:
    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad

    mask_prev = mask.copy()  # save step

    rgb, bin_white, bin_yellow, mask, to_tensor = getData(obs)  # get new data

    sum_img1 = np.hstack([bin_yellow, bin_white])
    sum_img2 = np.hstack([rgb, mask])
    sum_img1 = cv2.cvtColor(sum_img1, cv2.COLOR_GRAY2RGB)
    sum_img = np.vstack([sum_img2, sum_img1])
    cv2.imshow('bin', sum_img)

    # difference = cv2.subtract(mask_prev, mask)
    # difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    # ret, m = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # cv2.imshow('diff', difference)

    input_tensor = torch.from_numpy(to_tensor).unsqueeze(0).unsqueeze(0)
    print(input_tensor.shape)

    input_tensor = torch.from_numpy(rgb).unsqueeze(0)
    print(input_tensor.shape)
    input_tensor = input_tensor/255
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    input_tensor = input_tensor.to(DEVICE,dtype=torch.float32)

    pred = model(input_tensor)
    steering = pred.detach().cpu().numpy()[0][0]
    #speed, steering = pred.detach().cpu().numpy()[0]
    criterion = nn.SmoothL1Loss()
    reward = [map_range(reward, -3, 1, 0, 1)]
    reward_agent = torch.from_numpy(np.array(reward)).unsqueeze(0).to(dtype=torch.float32)
    reward_agent.requires_grad = True
    reward_target = torch.from_numpy(np.array(1)).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)

    loss = criterion(reward_agent, reward_target)
    print(loss,reward_agent,reward_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    k_p = 20
    k_d = 5
    speed = 0.1
    # steering = k_p * distance_to_road_center + k_d * angle_from_straight_in_rads

    obs, reward, done, info = env.step([speed, steering*10-0.5])
    if done or env.step_count > 1000:
        env.reset()
        obs, reward, done, info = env.step([speed, steering])
        print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))
    total_reward += reward

    env.render(mode="top_down")
