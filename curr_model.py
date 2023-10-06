# -*- coding:utf-8 -*-
"""
@Editor: Ether
@Time: 2023/9/9 10:01
@File: curr_model
@Contact: 211307040003@hhu.edu.cn
@Version: 1.0
@Description: None
"""
from model.air_trans import AirTrans


def get_model(way, shot, is_flatten):
    return AirTrans(
        # box_predictor params
        way, shot, is_flatten=is_flatten, roi_size=5, num_classes=way + 1,
        # backbone
        backbone_name='resnet50', pretrained=True,
        returned_layers=None, trainable_layers=3,
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


prefix = '20230915'


def get_save_root(surfix, way, shot, dataset, lr):
    return '/data/chenzh/AirTrans/results/{}_{}/{}_{}way_{}shot' \
        .format(prefix,
                surfix,
                dataset,
                way,
                shot,
                lr)
