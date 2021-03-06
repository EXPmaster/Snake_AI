# -*- coding: utf-8 -*-
""" Config File"""

""" ---------------- basic configs -----------------"""
KEY = {'UP': 0, 'LEFT': 1, 'DOWN': 2, 'RIGHT': 3}  # 键位
WIDTH, HEIGHT = 200, 200  # 画布宽、高
PIXEL_SIZE = 20  # 单个网格大小
SNAKE_POS_X, SNAKE_POS_Y = 80, 100  # 蛇初始位置
SNAKE_INIT_LEN = 3  # 蛇初始长度
PLAY_FPS = 10  # 帧率
FOOD_VALID_STEPS = 50  # 食物消失的步数
MODEL_PATH = './weights/dqn3.pt'  # 用于训练的模型位置
PLAY_MODEL_PATH = './weights/dqn3_reduced.pt'  # 用来玩的模型的位置

""" ---------------- training configs -----------------"""
BATCH_SIZE = 500  # 批大小
MEMORY_SIZE = 10000  # 记忆容量
MEM_CLEAN_SIZE = 2000  # 随机清除记忆的数量
BEGIN_LR = 1e-4  # 初始化大学习率
LEARNING_RATE = 1e-6  # 学习率
MOMENTUM = 0.9  # 动量
MAX_STEP = 1000  # 每一幕的最大行动步数

EAT_FOOD_REWARD = 1.0  # 吃食物的奖励
DIE_REWARD = -1.0  # 死亡惩罚
SNAKE_CLOSE_TO_FOOD_REWARD = 0.1  # 蛇接近食物奖励
SNAKE_AWAY_FROM_FOOD_REWARD = -0.1  # 蛇远离食物惩罚
FOOD_DISAPPEAR_REWARD = -0.5  # 蛇在限定时间内没有吃到食物的惩罚

# 采用epsilon贪心策略时，用于确定探索概率的参数
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50

GAMMA = 0.95  # 用于计算折扣回报的折扣参数
TARGET_UPDATE = 10  # 更新target net 的频率
NUM_EPISODES = 200_0000  # 总共进行的游戏轮数
SAVE_MODEL_STEPS = 50  # 保存模型的频率
""" ---------------- colors -----------------"""
BLACK = [0, 0, 0]
GRAY = [128, 128, 128]
WHITE = [255, 255, 255]
GREEN = [34, 139, 34]
RED = [255, 0, 0]
