# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    r"""DQN神经网络"""
    def __init__(self, h, w, outputs):
        r"""自定义神经网络结构

        :param h: 输入图片的高
        :param w: 输入图片的宽
        :param outputs: 输出维度
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6(inplace=True)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - kernel_size) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        r"""前向传播"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.shape[0], -1))
