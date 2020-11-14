import pygame as pyg
from pygame.locals import *
import sys
import torch
from configs import *
from utils.utils import init_game_state, gen_walls, gen_food, get_screen,\
    load_model_only, save_model_only


def main():
    # 初始化设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pyg.init()
    screen = pyg.display.set_mode((WIDTH, HEIGHT))
    pyg.display.set_caption('Snake AI Play')
    clock = pyg.time.Clock()
    font = pyg.font.Font(None, 20)
    n_actions = 4

    # 初始化生成蛇、食物、墙
    snake, food = init_game_state()
    walls = gen_walls()
    game_over = False

    init_screen = get_screen(screen, device)
    _, _, screen_height, screen_width = init_screen.shape

    # 读模型
    try:
        policy_net = load_model_only(PLAY_MODEL_PATH, screen_height, screen_width, n_actions, device)
    except Exception as e:
        print(e)
        save_model_only(MODEL_PATH, PLAY_MODEL_PATH)
        policy_net = load_model_only(PLAY_MODEL_PATH, screen_height, screen_width, n_actions, device)

    while True:
        score = 0
        clock.tick(PLAY_FPS)  # 延时
        for event in pyg.event.get():
            if event.type == QUIT:
                pyg.quit()
                sys.exit()

        if game_over:
            clock.tick(1000)
            game_over = False
            print(score)
            score = 0

