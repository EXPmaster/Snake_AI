# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch


class GrayScale:
    """RGB转灰度图"""

    def __call__(self, img):
        assert isinstance(img, np.ndarray), 'Img type unknown'
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img_gray


class Resize:
    """改变图片大小"""

    def __init__(self, img_size):
        assert isinstance(img_size, int), 'Img size must be an integer'
        self.output_size = img_size

    def __call__(self, img):
        # 使用双线性插值对输入image做resize
        img_resized = cv2.resize(img, (self.output_size, self.output_size),
                                 interpolation=cv2.INTER_LINEAR)
        return img_resized


class Normalize:
    """图片像素归一化"""

    def __call__(self, img):
        # 归一化到0～1之间
        img_normalized = img / 255
        return img_normalized


class ToTensor:
    """图像numpy数组转Tensor"""

    def __call__(self, img):
        # numpy image: H x W x C ->
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32)
        return torch.from_numpy(img)

