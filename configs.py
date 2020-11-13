# -*- coding: utf-8 -*-
""" Config File"""

""" ---------------- basic configs -----------------"""
KEY = {'UP': 0, 'LEFT': 1, 'DOWN': 2, 'RIGHT': 3}  # 键位
WIDTH, HEIGHT = 150, 150  # 画布宽、高
PIXEL_SIZE = 10  # 单个网格大小
SNAKE_POS_X, SNAKE_POS_Y = 70, 70  # 蛇初始位置
SNAKE_INIT_LEN = 3  # 蛇初始长度
PLAY_FPS = 1  # 帧率
FOOD_VALID_STEPS = 20  # 食物消失的步数
MODEL_PATH = './weights/dqn.pt'  # 模型位置

""" ---------------- training configs -----------------"""
BATCH_SIZE = 128  # 批大小
MEMORY_SIZE = 10000  # 记忆容量
MEM_CLEAN_SIZE = 5000  # 随机清除记忆的数量
LEARNING_RATE = 1e-4  # 学习率
MOMENTUM = 0.9  # 动量

EAT_FOOD_REWARD = 1.0  # 吃食物的奖励
DIE_REWARD = -10.0  # 死亡惩罚
SNAKE_ALIVE_REWARD = -1e-3  # 苟活惩罚

# 采用epsilon贪心策略时，用于确定探索概率的参数
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

GAMMA = 0.999  # 用于计算折扣回报的折扣参数
TARGET_UPDATE = 10  # 更新target net 的频率
NUM_EPISODES = 2000  # 总共进行的游戏轮数
""" ---------------- colors -----------------"""
BLACK = [0, 0, 0]
GRAY = [128, 128, 128]
WHITE = [255, 255, 255]
GREEN = [34, 139, 34]
RED = [255, 0, 0]
