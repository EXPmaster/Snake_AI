import pygame as pyg
from pygame.locals import *
import sys
import torch
from configs import *
from utils.utils import init_game_state, gen_walls, gen_food, get_screen,\
    load_model_only, save_model_only, draw_scene


def main():
    # 初始化设置
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    pyg.init()
    screen = pyg.display.set_mode((WIDTH, HEIGHT))
    # screen = pyg.Surface((WIDTH, HEIGHT))
    # pyg.display.set_caption('Snake AI Play')
    clock = pyg.time.Clock()
    font = pyg.font.Font(None, 20)
    n_actions = 4

    # 初始化生成蛇、食物、墙
    snake, food = init_game_state()
    walls = gen_walls()
    game_over = False
    food_down_count = FOOD_VALID_STEPS

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
            clock.tick(1)
            game_over = False
            print(score)
            score = 0
            snake, food = init_game_state()
            food_down_count = FOOD_VALID_STEPS

        # 生成新食物
        if not food or food_down_count == 0:
            food = gen_food(snake)
            food_down_count = FOOD_VALID_STEPS

        draw_scene(screen, snake, food, walls, needs_lines=False, play_game=True)
        state = get_screen(screen, device)
        action = policy_net(state).max(1)[1].view(1, 1)

        key = action.item()
        snake.choose_movement(key)
        snake.move()

        # 蛇撞墙了
        if snake.hits_wall(walls):
            game_over = True
        # 蛇吃了自己
        elif snake.eats_itself():
            game_over = True
        # 蛇生长
        elif snake.eats_food(food):
            score += 1
            snake.grow()
            food = None

        food_down_count -= 1


if __name__ == '__main__':
    main()
