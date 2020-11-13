# -*- coding: utf-8 -*-

import pygame as pyg
from configs import *
import numpy as np
from objects import Food, Wall, Snake
import cv2
import torchvision.transforms as T
from .image_transform import GrayScale, Resize, Normalize, ToTensor


def get_snake_position(snake):
    r"""获取蛇身体的坐标

    :param snake: snake对象
    :return: 蛇位置坐标数组
    """
    snake_pos = [(seg.x, seg.y) for seg in snake.snake_queue]
    return snake_pos


def check_collision_by_coords(Ax, Ay, Bx, By, size=PIXEL_SIZE):
    r"""通过坐标检查两物体是否有重叠

    :param Ax: A的x坐标
    :param Ay: A的y坐标
    :param Bx: B的x坐标
    :param By: B的y坐标
    :param size: 物体大小
    :return: 重叠返回True，不重叠返回False
    """
    if Ax < Bx + size and Ax + size > Bx and Ay < By + size and Ay + size > By:
        return True
    else:
        return False


def gen_random_pos():
    r""" 生成在墙以内的随机坐标

    :return: x坐标，y坐标
    """
    x = np.random.choice(np.arange(PIXEL_SIZE, WIDTH - PIXEL_SIZE, PIXEL_SIZE))
    y = np.random.choice(np.arange(PIXEL_SIZE, WIDTH - PIXEL_SIZE, PIXEL_SIZE))
    return x, y


def gen_food(snake):
    r"""生成食物

    :param snake: 蛇对象
    :return: 食物对象
    """
    snake_pos = get_snake_position(snake)
    x, y = gen_random_pos()
    collision = True
    # 食物不能与蛇重叠
    while collision:
        has_collision = False
        for pos in snake_pos:
            if check_collision_by_coords(pos[0], pos[1], x, y):
                has_collision = True
                break

        if has_collision:
            x, y = gen_random_pos()
        else:
            collision = False

    return Food(x, y)


def gen_walls():
    r"""生成墙

    :return: 墙数组
    """
    wall_upper = [Wall(x, 0) for x in range(0, WIDTH, PIXEL_SIZE)]
    wall_down = [Wall(x, HEIGHT - PIXEL_SIZE) for x in range(0, WIDTH, PIXEL_SIZE)]
    wall_left = [Wall(0, y) for y in range(PIXEL_SIZE, HEIGHT - PIXEL_SIZE, PIXEL_SIZE)]
    wall_right = [Wall(WIDTH - PIXEL_SIZE, y) for y in range(PIXEL_SIZE, HEIGHT - PIXEL_SIZE, PIXEL_SIZE)]
    return wall_upper + wall_down + wall_left + wall_right


def init_game_state():
    r"""初始化游戏

    :return: 蛇对象，食物对象
    """
    snake = Snake(SNAKE_POS_X, SNAKE_POS_Y, SNAKE_INIT_LEN)
    food = gen_food(snake)

    return snake, food


def draw_scene(screen, snake, food, walls, needs_lines=True):
    r"""绘制场景

    :param screen: screen
    :param snake: 蛇
    :param food: 食物
    :param walls: 墙
    :param needs_lines: 是否需要画线
    :return: None
    """
    screen.fill(BLACK)

    # 画墙
    for wall in walls:
        wall.draw_object(screen)

    # 画蛇
    for seg in snake.snake_queue:
        seg.draw_object(screen)

    # 画食物
    food.draw_object(screen)

    if needs_lines:
        # 画垂直线
        for x in range(PIXEL_SIZE, WIDTH, PIXEL_SIZE):
            pyg.draw.line(screen, BLACK, (x, 0), (x, HEIGHT), 1)

        # 画水平线
        for y in range(0, HEIGHT, PIXEL_SIZE):
            pyg.draw.line(screen, BLACK, (0, y), (WIDTH, y), 1)

    pyg.display.update()


def get_screen(screen, show_img=False):
    r"""

    :param screen: 当前screen
    :param show_img: 是否显示图片
    :return: 图片numpy数组，size: WIDTH*HEIGHT*3
    """
    screen_img = np.rot90(pyg.surfarray.array3d(screen))[::-1]  # 调换使图片显示正确
    # image transformation
    tsfm = T.Compose([
        Resize(84),
        Normalize(),
        ToTensor()
    ])
    ret_img = tsfm(screen_img)

    if show_img:
        img = cv2.cvtColor(screen_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('test', img)
    return ret_img
