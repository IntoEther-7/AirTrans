# -*- coding:utf-8 -*-
"""
@Editor: Ether
@Time: 2023/9/3 9:50
@File: test_voc
@Contact: 211307040003@hhu.edu.cn
@Version: 1.0
@Description: None
"""
import os.path

import torch

from curr_model import get_model
from util.dataset import *

from model.air_trans import AirTrans
from util.tester import tester_for_air_trans

torch.set_printoptions(sci_mode=False)
root = '/home/chenzh/code/FRNOD/datasets/fsod'
json_path = 'annotations/fsod_train.json'
img_path = 'images'
continue_weight = None
save_root = None


def way_shot_test(way, shot, lr, index, continue_weight, save_root):
    # result_voc_r50_2way_5shot_lr2e-06_loss_weight_0
    model = get_model(way, shot, True)

    tester_for_air_trans(
        # 基础参数
        way=way, shot=shot, query_batch=1, is_cuda=True,
        # 设备参数
        random_seed=None, gpu_index=1,
        # 数据集参数
        root=root,
        json_path=json_path,
        img_path=img_path,
        split_cats=base_ids_voc1,
        # 模型
        model=model,
        # 权重文件
        continue_weight=continue_weight,
        # 保存相关的参数
        save_root=save_root)


if __name__ == '__main__':
    continue_weight = 'AirTrans_25_10572.pth'
    save_root = os.path.join('results',
                             '20230910_flatten',
                             'fsod_5way_5shot_lr0.002')
    way_shot_test(5, 5, 2e-03, 1, continue_weight, save_root)

