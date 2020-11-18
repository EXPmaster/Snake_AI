# -*- coding: utf-8 -*-

from configs import *
import pygame as pyg
from utils.memory import Transition
import torch
from utils.utils import init_game_state, gen_walls, gen_food, \
    get_screen, draw_scene, load_model, save_model, save_model_only
import torch.nn as nn
import random
import math
from itertools import count
import numpy as np
import logging
import time
import os


def select_action(state):
    r"""根据策略网络选择策略，采用epsilon贪心，使用概率控制采用贪心方式
    还是进行探索"""
    steps_done = episode
    sample = random.random()
    # 设置阈值，随着训练轮次增大，agent的行动会愈加趋向保守，即进行探索的概率会降低
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1 * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        # 若随机数大于阈值，则采用贪心方案，采取当前回报最大的策略
        with torch.no_grad():
            # 选出回报最大的策略对应的索引，代表采取的行动
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # 进行随机探索，等概率从当前策略集中选出一种策略
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long, device=device)


def optimize_model():
    r"""优化模型"""
    # 当记忆不足batch size时，不更新策略网络
    transitions = []
    for memory in [short_memory, good_memory, bad_memory]:
        transitions += memory.sample(BATCH_SIZE)
    size = len(transitions)
    if size < BATCH_SIZE:
        return
    transitions = random.sample(transitions, BATCH_SIZE)  # 从记忆中采样一组数据
    batch = Transition(*zip(*transitions))  # 将一个批次的Transition变成一个Transition批次，每个key有一个batch size的数据

    # 选出下一个状态不是游戏结束状态
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 使用策略网络计算Q(s,a)，代表采取行动a获得的实际累计回报
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 计算Q(s')，为根据经验（target net）获得的下一状态（即采取行动a之后跳转到的状态）的累计回报
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_state).max(1)[0].detach()

    # 计算Q(s,a)值，为r + gamma * Q(s',a')，gamma表示衰减因子
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # 根据当前的实际Q来更新经验
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    acc_loss.append(loss.item())

    # 更新网络
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train():
    r"""训练"""
    global screen, food
    food_down_count = FOOD_VALID_STEPS
    t_reward = []
    game_over = False
    eaten = False
    for t in count():
        reward = 0
        # 获取当前状态
        draw_scene(screen, snake, food, walls, needs_lines=False)
        state = get_screen(screen, device=device)
        # 计算蛇头与食物之间的欧式距离（未开方）
        distance = (snake.head().x - food.x) ** 2 + (snake.head().y - food.y) ** 2

        # snake move
        action = select_action(state)
        key = action.item()
        snake.choose_movement(key)
        snake.move()

        # get reward from next state
        # 蛇撞墙了
        if snake.hits_wall(walls):
            reward = DIE_REWARD
            game_over = True
        # 蛇吃了自己
        elif snake.eats_itself():
            reward = DIE_REWARD
            game_over = True
        # 蛇生长
        elif snake.eats_food(food):
            eaten = True
            reward = EAT_FOOD_REWARD
            snake.grow()
            food = None

        # 蛇没吃到食物且没有死亡
        if reward == 0:
            if food_down_count == 0:
                # 超时惩罚
                reward = FOOD_DISAPPEAR_REWARD
                food_down_count = FOOD_VALID_STEPS
            else:
                # 计算蛇头与食物之间的欧式距离（未开方）
                next_distance = (snake.head().x - food.x) ** 2 + (snake.head().y - food.y) ** 2
                if next_distance < distance:
                    # 距离缩短
                    reward = SNAKE_CLOSE_TO_FOOD_REWARD
                elif next_distance > distance:
                    # 距离拉长
                    # reward = -SNAKE_CLOSE_TO_FOOD_REWARD + \
                    #          (SNAKE_AWAY_FROM_FOOD_REWARD + SNAKE_CLOSE_TO_FOOD_REWARD) * \
                    #          math.exp(-(len(snake) - 3) / 5)
                    reward = SNAKE_AWAY_FROM_FOOD_REWARD
                else:
                    reward = 0.0

        t_reward.append(reward)
        reward_tensor = torch.tensor([reward], dtype=torch.float, device=device)

        # 生成新食物
        if not food:  # or food_down_count == 0:
            food = gen_food(snake)
            food_down_count = FOOD_VALID_STEPS

        if game_over or eaten:
            next_state = None
            if eaten:
                eaten = False
        else:
            draw_scene(screen, snake, food, walls, needs_lines=False)
            next_state = get_screen(screen, device=device)

        # 保存状态
        if not game_over:
            if reward > 0:
                good_memory.push(state, action, next_state, reward_tensor)
            else:
                short_memory.push(state, action, next_state, reward_tensor)
        else:
            bad_memory.push(state, action, next_state, reward_tensor)

        # 更新policy net
        optimize_model()
        # 超时退出
        if t > MAX_STEP:
            game_over = True

        if game_over:
            snake_len.append(len(snake))
            acc_reward.append(np.sum(t_reward))
            return
        food_down_count -= 1


if __name__ == '__main__':
    # 配置logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime(r'%Y%m%d%H%M', time.localtime(time.time()))
    log_path = './logs/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_name = log_path + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, 'w')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)

    show_screen = False
    screen_size = (WIDTH, HEIGHT)
    # n_actions = 3  # 对应4种策略，0表示蛇头左转，1表示不动，2表示蛇头右转
    n_actions = 4  # 对应4种策略，上下左右
    acc_loss = []  # 累计loss
    acc_reward = []  # 累计reward
    snake_len = []  # 蛇长
    # 配置
    options = {
        'restart_mem': False,
        'restart_models': False,
        'restart_optim': False,
        'random_clean_memory': False,
        'opt': 'adam'
    }

    pyg.init()
    screen = pyg.Surface(screen_size)

    if show_screen:
        screen = pyg.display.set_mode(screen_size)
        pyg.display.set_caption('Snake AI')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 初始化生成蛇、食物、墙
    snake, food = init_game_state()
    walls = gen_walls()

    clock = pyg.time.Clock()
    init_screen = get_screen(screen, device)
    _, _, screen_height, screen_width = init_screen.shape
    # 初始化神经网络
    policy_net, target_net, optimizer, memories = load_model(MODEL_PATH, screen_height,
                                                             screen_width, n_actions, device,
                                                             **options)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    criterion = nn.SmoothL1Loss()
    # 初始化记忆单元
    short_memory = memories['short']
    good_memory = memories['good']
    bad_memory = memories['bad']

    # 初始化学习率
    for param_group in optimizer.param_groups:
        if param_group['lr'] != BEGIN_LR:
            param_group['lr'] = BEGIN_LR
            break

    # 训练模型主循环
    for episode in range(NUM_EPISODES):
        if (episode + 1) % 200 == 0:
            # Decay learning rate
            for param_group in optimizer.param_groups:
                if param_group['lr'] > LEARNING_RATE:
                    param_group['lr'] = np.round(param_group['lr'] * 0.5, 10)
                    break

        train()
        snake, food = init_game_state()

        # 更新target net
        if (episode + 1) % TARGET_UPDATE == 0:

            logger.info('*' * 20)
            logger.info(f'Episodes done {episode + 1}')
            logger.info(f'Batch size: {BATCH_SIZE}')
            logger.info(f'Average snake length per episode: {np.mean(snake_len)}')
            logger.info(f'Average reward per episode {np.mean(acc_reward)}')
            logger.info(f'Average loss: {np.mean(acc_loss)}')
            logger.info('Memories:')
            logger.info('  - short: {}'.format(len(short_memory)))
            logger.info('  - good: {}'.format(len(good_memory)))
            logger.info('  - bad: {}'.format(len(bad_memory)))
            for param_group in optimizer.param_groups:
                logger.info(f"learning rate={param_group['lr']}")
                break
            logger.info('Optimizer: {}'.format(optimizer.__class__.__name__))
            acc_loss = []  # 累计loss
            acc_reward = []  # 累计reward
            snake_len = []  # 蛇长

            print('updating target net')
            target_net.load_state_dict(policy_net.state_dict())

        # 保存模型参数及记忆
        if (episode + 1) % SAVE_MODEL_STEPS == 0:
            memories = {
                'short': short_memory,
                'good': good_memory,
                'bad': bad_memory
            }
            save_model(MODEL_PATH, policy_net, target_net, optimizer, memories)
            save_model_only(MODEL_PATH, PLAY_MODEL_PATH)
