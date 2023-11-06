# -*- coding:utf-8 -*-
"""
@Editor: Ether
@Time: 2023/11/6 22:02
@File: visual
@Contact: 211307040003@hhu.edu.cn
@Version: 1.0
@Description: None
"""
import os.path

from torch import *
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import transforms

t = transforms.ToPILImage()


def judge(_input: Tensor, path_root):
    _dim = len(_input.shape)
    if _dim == 5:
        count = 0
        for i in _input:
            judge(i, "{}/{}".format(path_root, count))
            count = count + 1
    elif _dim == 4:
        count = 0
        for i in _input:
            judge(i, "{}/{}".format(path_root, count))
            count = count + 1
    elif _dim == 3:
        if _input.shape[0] in [1, 3]:
            save_img(_input, "{}.png".format(path_root))
        else:
            count = 0
            for i in _input:
                save_img(i, "{}/{}.png".format(path_root, count))
                count = count + 1
    elif _dim == 2:
        save_img(_input, "{}.png".format(path_root))


def save_img(_input: Tensor, path_png):
    max_val = _input.max()
    min_val = _input.min()
    _one = (_input - min_val) / (max_val - min_val)

    par_path = os.path.dirname(path_png)
    if not os.path.exists(par_path):
        os.makedirs(par_path)
    img: Image = t(_input)
    img.save(path_png)

