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
from util.dataset import *

from model.air_trans import AirTrans
from util.tester import tester_for_air_trans

torch.set_printoptions(sci_mode=False)
root = '/home/chenzh/code/FRNOD/datasets/voc/VOCdevkit/VOC2012'
json_path = 'cocoformatJson/voc_2012_train.json'
img_path = 'JPEGImages'
continue_weight = None
save_root = None


def way_shot_test(way, shot, lr, index, continue_weight, save_root):
    # result_voc_r50_2way_5shot_lr2e-06_loss_weight_0
    model = AirTrans(
        # box_predictor params
        way, shot, roi_size=5, is_flatten=False, num_classes=way + 1,
        # backbone
        backbone_name='resnet50', pretrained=True,
        returned_layers=None, trainable_layers=4,
        # transform parameters
        min_size=600, max_size=1000,
        image_mean=None, image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=2000,
        rpn_post_nms_top_n_train=1000, rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=32, rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None, box_head=None, box_predictor=None,
        box_score_thresh=0.05, box_nms_thresh=0.7, box_detections_per_img=20,
        box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
        box_batch_size_per_image=32, box_positive_fraction=0.25,
        bbox_reg_weights=(10., 10., 5., 5.),
        rpn_focal=False, head_focal=False
    )

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
    continue_weight = 'AirTrans_60_1305.pth'
    save_root = os.path.join('results',
                             'air_trans_20230905_decoder_aux',
                             'result_voc_r50_5way_5shot_lr0.002')
    way_shot_test(5, 5, 2e-04, 0, continue_weight, save_root)

    continue_weight = 'AirTrans_10_246.pth'
    save_root = os.path.join('results',
                             'air_trans_20230905_decoder_aux_novel',
                             'result_voc_r50_5way_5shot_lr0.0002')
    way_shot_test(5, 5, 2e-04, 0, continue_weight, save_root)
