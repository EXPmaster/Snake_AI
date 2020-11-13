# -*- coding: utf-8 -*-
""" Config File"""

""" ---------------- basic configs -----------------"""
KEY = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4}  # 键位
WIDTH, HEIGHT = 150, 150  # 画布宽、高
PIXEL_SIZE = 10  # 单个网格大小
SNAKE_POS_X, SNAKE_POS_Y = 70, 70  # 蛇初始位置
SNAKE_INIT_LEN = 3  # 蛇初始长度
PLAY_FPS = 1  # 帧率
FOOD_VALID_STEPS = 20  # 食物消失的步数

""" ---------------- colors -----------------"""
BLACK = [0, 0, 0]
GRAY = [128, 128, 128]
WHITE = [255, 255, 255]
GREEN = [34, 139, 34]
RED = [255, 0, 0]
