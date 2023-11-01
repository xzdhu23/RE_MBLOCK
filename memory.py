import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward'))


class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)  #定义一个collections格式的元组 memory用于存储训练结果
        self.capacity = capacity    # 定义memory的大小 i.e. 决定了接下来的learn过程agent可以参照的历史经验的size

    def push(self, state, next_state, action, reward):
        self.memory.append(Transition(state, next_state, action, reward))  # 将每次训练的结果接续保存下来

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)#随机采样
        batch = Transition(*zip(*transitions))#按照Transition的格式压缩 多个Transition合成1个，e.g. (Transition(1),Transition(2),Transition(3),Transition(4))  ——> Transition(1,2,3,4)
        return batch

    def __len__(self):
        return len(self.memory)
