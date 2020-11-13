# -*- coding: utf-8 -*-

from configs import *
import pygame as pyg
from utils.memory import Transition, ReplayMemory
import torch
from utils.utils import init_game_state, gen_walls, gen_food, \
    get_screen, draw_scene, load_model, save_model
from model.dqn import DQN
import torch.optim as optim
import torch.nn as nn
import random
import math
from itertools import count


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
        return torch.LongTensor([[random.randrange(n_actions)]], device=device)


def optimize_model():
    r"""优化模型"""
    # 当记忆不足batch size时，不更新策略网络
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)  # 从记忆中采样一组数据
    batch = Transition(*zip(transitions))  # 将一个批次的Transition变成一个Transition批次，每个key有一个batch size的数据

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
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(0))

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

        reward = torch.tensor([reward], device=device)
        if game_over:
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
    show_screen = True
    screen_size = (WIDTH, HEIGHT)
    n_actions = 3  # 对应3种策略，0表示蛇头左转，1表示不动，2表示蛇头右转
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
        if param_group['lr'] != LEARNING_RATE:
            param_group['lr'] = LEARNING_RATE
            break

    steps_done = 0
    for episode in range(NUM_EPISODES):
        food_down_count = FOOD_VALID_STEPS
        game_over = False
        train()
        snake, food = init_game_state()
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

