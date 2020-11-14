# -*- coding: utf-8 -*-
""" Human play file """

import pygame as pyg
import sys
from pygame.locals import *
from configs import *
from utils.utils import init_game_state, gen_walls, draw_scene, get_screen, gen_food
import time


def playgame():
    pyg.init()
    screen = pyg.display.set_mode((WIDTH, HEIGHT))
    pyg.display.set_caption('Snake Human Play')
    # 初始化生成蛇、食物、墙
    snake, food = init_game_state()
    walls = gen_walls()
    game_over = False
    food_down_count = FOOD_VALID_STEPS

    clock = pyg.time.Clock()

    while True:
        clock.tick(PLAY_FPS)  # 延时
        for event in pyg.event.get():
            if event.type == QUIT:
                pyg.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                # 键盘控制蛇行动
                if event.key == K_UP and snake.head().direction != KEY['DOWN']:
                    snake.head().direction = KEY['UP']
                elif event.key == K_DOWN and snake.head().direction != KEY['UP']:
                    snake.head().direction = KEY['DOWN']
                elif event.key == K_LEFT and snake.head().direction != KEY['RIGHT']:
                    snake.head().direction = KEY['LEFT']
                elif event.key == K_RIGHT and snake.head().direction != KEY['LEFT']:
                    snake.head().direction = KEY['RIGHT']
        snake.move()
        # 蛇撞墙了
        if snake.hits_wall(walls):
            game_over = True
        # 蛇吃了自己
        if snake.eats_itself():
            game_over = True

        # 蛇生长
        if snake.eats_food(food):
            snake.grow()
            food = None

        # 生成新食物
        if not food or food_down_count == 0:
            food = gen_food(snake)
            food_down_count = FOOD_VALID_STEPS
        # 绘制场景
        draw_scene(screen, snake, food, walls, needs_lines=True, play_game=True)

        # 旧食物存在时间减1
        food_down_count -= 1
        # get_screen(screen, device='cpu', show_img=False)
        # 游戏结束
        if game_over:
            time.sleep(3)
            snake, food = init_game_state()
            food_down_count = FOOD_VALID_STEPS
            game_over = False


if __name__ == '__main__':
    playgame()