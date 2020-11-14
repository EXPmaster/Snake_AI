# -*- coding: utf-8 -*-
from configs import *
import pygame as pyg


def check_obj_collision(objA, objB, size=PIXEL_SIZE):
    r"""检查两网格对象是否重叠

    :param objA: 对象A
    :param objB: 对象B
    :param size: 物体大小
    :return: 重叠返回True，不重叠返回False
    """
    if objA.x < objB.x + size and objA.x + size > objB.x and\
            objA.y < objB.y + size and objA.y + size > objB.y:
        return True
    else:
        return False


class BasicPixel:
    r"""网格基类"""
    __slots__ = ['x', 'y', 'color', 'direction', 'position']

    def __init__(self, x, y, color, direction=None):
        r"""

        :param x: x 坐标
        :param y: y 坐标
        :param color: 颜色
        :param direction: 若不为None则表示蛇头方向
        """
        self.x = x
        self.y = y
        self.color = color
        self.direction = direction
        self.position = (self.x, self.y, PIXEL_SIZE, PIXEL_SIZE)

    def change_pos(self):
        r"""改变position"""
        self.position = (self.x, self.y, PIXEL_SIZE, PIXEL_SIZE)

    def draw_object(self, screen):
        r"""在画布上画出当前块

        :param screen: 当前画布
        :return: None
        """
        pyg.draw.rect(screen, self.color, self.position)


class Wall(BasicPixel):
    r"""墙类"""

    def __init__(self, x, y, color=GRAY):
        super().__init__(x, y, color)


class Food(BasicPixel):
    r"""食物类"""

    def __init__(self, x, y, color=RED):
        super().__init__(x, y, color)


class Snake:
    r"""蛇类"""

    def __init__(self, x, y, init_len=3):
        r"""

        :param x: x坐标
        :param y: y坐标
        :param init_len: 初始蛇长
        """
        assert init_len > 0, 'snake must be longer than 0'
        self.head_color = GREEN
        self.body_color = WHITE
        # 初始化蛇头
        head = BasicPixel(x, y, self.head_color, direction=KEY['RIGHT'])
        self.snake_queue = [head]  # 用于存放蛇身的队列
        next_x = x
        # 初始化蛇身
        for i in range(1, init_len):
            next_x = next_x - PIXEL_SIZE
            self.snake_queue.insert(0, BasicPixel(next_x, y, self.body_color, direction=KEY['RIGHT']))
        self.movement = PIXEL_SIZE  # 运动步长

    def __len__(self):
        return len(self.snake_queue)

    def head(self):
        return self.snake_queue[-1]

    def choose_movement(self, key):
        r"""通过神经网络输出的action来选择移动方向，为0
        则向左走，为1向上走，为2向右走，为3向下走

        :param key: 策略网络输出的action index
        :return: None
        """
        # if key == 0:
        #     # 蛇头左转
        #     self.head().direction = (self.head().direction + 1) % 4
        # elif key == 2:
        #     # 蛇头右转
        #     self.head().direction = (self.head().direction - 1) % 4
        if key == 0 and self.head().direction != KEY['RIGHT']:
            self.head().direction = KEY['LEFT']
        elif key == 1 and self.head().direction != KEY['DOWN']:
            self.head().direction = KEY['UP']
        elif key == 2 and self.head().direction != KEY['LEFT']:
            self.head().direction = KEY['RIGHT']
        elif key == 3 and self.head().direction != KEY['UP']:
            self.head().direction = KEY['DOWN']

    def move(self):
        r"""移动"""
        snake_tail = self.snake_queue.pop(0)
        snake_head = self.head()
        snake_head.color = self.body_color
        if snake_head.direction == KEY['UP']:
            snake_tail.y = snake_head.y - self.movement
            snake_tail.x = snake_head.x
        elif snake_head.direction == KEY['DOWN']:
            snake_tail.y = snake_head.y + self.movement
            snake_tail.x = snake_head.x
        elif snake_head.direction == KEY['LEFT']:
            snake_tail.y = snake_head.y
            snake_tail.x = snake_head.x - self.movement
        elif snake_head.direction == KEY['RIGHT']:
            snake_tail.y = snake_head.y
            snake_tail.x = snake_head.x + self.movement

        snake_tail.direction = snake_head.direction
        snake_tail.color = self.head_color
        snake_tail.change_pos()
        self.snake_queue.append(snake_tail)

    def grow(self):
        r"""蛇吃食物长长"""
        x = self.snake_queue[0].x
        y = self.snake_queue[0].y
        color = self.body_color
        self.snake_queue.insert(0, BasicPixel(x, y, color))

    def eats_itself(self):
        r"""判断蛇是否撞上自己

        :return: 若撞上返回True，否则返回False
        """
        head = self.head()
        for body in self.snake_queue[:-1]:
            if check_obj_collision(body, head):
                return True
        return False

    def hits_wall(self, walls):
        r"""判断蛇是否撞墙

        :param walls: 墙数组
        :return: 若撞上返回True，否则返回False
        """
        head = self.head()
        for wall in walls:
            if check_obj_collision(head, wall):
                return True
        return False

    def eats_food(self, food):
        r"""判断蛇吃到食物

        :param food: 食物对象
        :return: 若吃到返回True，否则返回False
        """
        head = self.head()
        if check_obj_collision(head, food):
            return True
        else:
            return False


if __name__ == '__main__':
    wall = Wall(1, 2)