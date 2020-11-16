import cv2
import torch
from configs import *
import numpy as np


def visualize():
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    memories = checkpoint['memories']
    short_memory = memories['short'].memory
    good_memory = memories['good'].memory
    bad_memory = memories['bad'].memory
    print(len(good_memory))
    cur_state = good_memory[265].state
    next_state = good_memory[265].next_state
    print(good_memory[265].reward)
    cur_img = tensor2img(cur_state)
    next_img = tensor2img(next_state)
    if cur_img is not None:
        cv2.imwrite('./img/tmp_cur.jpg', cur_img)
    if next_img is not None:
        cv2.imwrite('./img/tmp_next.jpg', next_img)


def tensor2img(raw):
    if isinstance(raw, torch.Tensor):
        tmp = raw * 255
        tmp = tmp.squeeze(0).permute(1, 2, 0)
        img = tmp.numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    else:
        if raw is None:
            print('no next state!')


if __name__ == '__main__':
    visualize()
