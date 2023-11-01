import random

import numpy as np
import torch
from tensorboardX import SummaryWriter

import torch.optim as optim

from DRL.DQN import algorithm
from config import replay_memory_capacity, batch_size, lr
from DRL.DQN.memory import Memory
from model import QNet

# parameters setting
channel_num = 8

# action_parameters setting


def get_action(state, target_net, esplion):
    if np.random.rand() <= esplion:
        # 这里应当从动作集合中随机选择action
        return np.random.randint(0, 111)
    else:
        # 选择概率最大的action
        return target_net.get_action(state)


def update_target_model(online_net, target_net):
    target_net.load_state_dict(online_net.state_dict())
    #  通过调用 load_state_dict() 函数，将 online_net 的参数状态加载到 target_net 中，从而使得 target_net 的参数与 online_net 的参数相同。
    #  state_dict() 函数返回了 online_net 的当前参数状态，它是一个字典，其中键是参数的名称，值是对应参数的张量。


def reward(S, K, W):
    pass



torch.manual_seed(500)

num_input = 14
num_output = 112

online_net = QNet(num_input, num_output)
target_net = QNet(num_input, num_output)
update_target_model(online_net, target_net)

online_net.train()
target_net.train()

memory = Memory(replay_memory_capacity)
writer = SummaryWriter('logs')
optimizer = optim.Adam(online_net.parameters(), lr=lr)

steps = 0
loss = 0
epsilon = 1.0
inition_explort = 10000
observation = algorithm.init_state()
state = torch.Tensor(observation)
for i in range(10):
    for j in range(10000):
        steps += 1
        action = get_action(state, target_net, epsilon)
        next_state, reward = algorithm.step(observation, action)
        next_state = torch.Tensor(next_state)
        action_one_hot = np.zeros(112)
        action_one_hot[action] = 1
        memory.push(state, next_state, action_one_hot, reward)
        state = next_state
        if steps >= inition_explort:
            epsilon -= 0.0001
            epsilon = max(epsilon, 0.01)
            batch = memory.sample(batch_size)
            loss = QNet.train_model(online_net, target_net, optimizer, batch)
            if steps % 10 == 0:
                update_target_model(online_net, target_net)
                writer.add_scalar('log/loss', float(loss), steps)
                writer.add_scalar('log/reward', float(reward), steps)
        if steps % 1000 == 0:
            print("#循环次数i={},steps={}".format(i, steps))
