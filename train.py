# -*- coding: utf-8 -*-

from configs import *
import pygame as pyg
from utils.memory import Transition, ReplayMemory
import torch
from utils.utils import init_game_state, gen_walls, gen_food, \
    get_screen, draw_scene, load_model, save_model
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
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        # 若随机数大于阈值，则采用贪心方案，采取当前回报最大的策略
        with torch.no_grad():
            # 选出回报最大的策略对应的索引
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # 进行随机探索，等概率从当前策略集中选出一种策略
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long, device=device)


def optimize_model():
    r"""优化模型"""
    # 当记忆不足batch size时，不更新策略网络
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)  # 从记忆中采样一组数据
    batch = Transition(*zip(*transitions))  # 将一个批次的Transition变成一个Transition批次，每个key有一个batch size的数据

    # 选出下一个状态不是游戏结束状态
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 计算Q(s,a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 计算V(s')
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_state).max(1)[0].detach()

    # 计算Q值（带衰减因子gamma）
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
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
    global screen, food, food_down_count, game_over
    draw_scene(screen, snake, food, walls, needs_lines=False)
    t_reward = []
    for t in count():
        # 获取当前状态
        state = get_screen(screen, device=device)
        reward = 0
        action = select_action(state)
        # snake move
        key = action.item()
        snake.choose_movement(key)
        snake.move()
        # 蛇撞墙了
        if snake.hits_wall(walls):
            reward = DIE_REWARD
            game_over = True
        # 蛇吃了自己
        if snake.eats_itself():
            reward = DIE_REWARD
            game_over = True
        # 蛇生长
        if snake.eats_food(food):
            reward = EAT_FOOD_REWARD
            snake.grow()
            food = None
        # 生成新食物
        if not food or food_down_count == 0:
            food = gen_food(snake)
            food_down_count = FOOD_VALID_STEPS
        # 蛇没吃到食物且没有死亡
        if reward == 0:
            reward = SNAKE_ALIVE_REWARD

        t_reward.append(reward)
        reward = torch.tensor([reward], device=device)

        if game_over:
            snake_len.append(len(snake))
            acc_reward.append(np.sum(t_reward))
            t_reward = []
            next_state = None
        else:
            draw_scene(screen, snake, food, walls, needs_lines=False)
            next_state = get_screen(screen, device=device)

        # 保存状态
        memory.push(state, action, next_state, reward)
        # 更新policy net
        optimize_model()

        if game_over:
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
    n_actions = 3  # 对应3种策略，0表示蛇头左转，1表示不动，2表示蛇头右转
    acc_loss = []  # 累计loss
    acc_reward = []  # 累计reward
    snake_len = []  # 蛇长
    # 配置
    options = {
        'restart_mem': False,
        'restart_models': False,
        'restart_optim': False,
        'random_clean_memory': True,
        'opt': 'rmsprop'
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
    game_over = False
    food_down_count = FOOD_VALID_STEPS

    clock = pyg.time.Clock()

    init_screen = get_screen(screen, device)
    _, _, screen_height, screen_width = init_screen.shape
    # 初始化神经网络
    policy_net, target_net, optimizer, memory = load_model(MODEL_PATH, screen_height,
                                                           screen_width, n_actions, device,
                                                           **options)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    criterion = nn.SmoothL1Loss()

    # 初始化学习率
    for param_group in optimizer.param_groups:
        if param_group['lr'] != BEGIN_LR:
            param_group['lr'] = BEGIN_LR
            break

    steps_done = 0

    # 训练模型主循环
    for episode in range(NUM_EPISODES):
        if (steps_done + 1) % 5000 == 0:
            # Decay learning rate
            for param_group in optimizer.param_groups:
                if param_group['lr'] > LEARNING_RATE:
                    param_group['lr'] = np.round(param_group['lr'] * 0.97, 10)
                    break

        train()
        if game_over:
            game_over = False
            snake, food = init_game_state()
            food_down_count = FOOD_VALID_STEPS

        # 更新target net
        if (episode + 1) % TARGET_UPDATE == 0:

            logger.info('*' * 20)
            logger.info(f'Episodes done {episode + 1}')
            logger.info(f'Batch size: {BATCH_SIZE}')
            logger.info(f'Average snake length per episode: {np.mean(snake_len)}')
            logger.info(f'Average reward per episode {np.mean(acc_reward)}')
            logger.info(f'Average loss: {np.mean(acc_loss)}')
            logger.info(f'Memory length: {len(memory)}')
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
            save_model(MODEL_PATH, policy_net, target_net, optimizer, memory)
