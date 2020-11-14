# -*- coding: utf-8 -*-
from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    r"""记忆单元，用于存放经历过的状态、动作、价值"""

    def __init__(self, capacity):
        self.capacity = capacity  # 容量
        self.memory = []
        self.position = 0  # 记忆的位置

    def push(self, *args):
        r"""save a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        r"""sample memory"""
        return random.sample(self.memory, batch_size)

    def random_clean_memory(self, clear_size):
        r"""随机清除记忆"""
        if clear_size < len(self.memory):
            self.memory = random.sample(self.memory, clear_size)
            self.position = clear_size
