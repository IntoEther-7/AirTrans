# -*- coding:utf-8 -*-
"""
@Editor: Ether
@Time: 2023/9/2 19:08
@File: train_voc
@Contact: 211307040003@hhu.edu.cn
@Version: 1.0
@Description: None
"""

from model.air_trans import AirTrans
from util.dataset import *
from util.trainer import trainer_for_air_trans

torch.set_printoptions(sci_mode=False)
root = '/home/chenzh/code/FRNOD/datasets/voc/VOCdevkit/VOC2012'
json_path = 'cocoformatJson/voc_2012_train.json'
img_path = 'JPEGImages'
loss_weights0 = {'loss_classifier': 1, 'loss_box_reg': 1,
                 'loss_objectness': 1, 'loss_rpn_box_reg': 1,
                 'loss_attention': 1, 'loss_aux': 1}
loss_weights1 = {'loss_classifier': 1, 'loss_box_reg': 1,
                 'loss_objectness': 1, 'loss_rpn_box_reg': 1,
                 'loss_attention': 0.03, 'loss_aux': 0.03}
loss_weights无监督attention = {'loss_classifier': 1, 'loss_box_reg': 1,
                               'loss_objectness': 1, 'loss_rpn_box_reg': 1,
                               'loss_attention': 0, 'loss_aux': 1}


def way_shot_train(way, shot, lr, loss_weights, gpu_index, loss_weights_index, split_cats):
    save_root = '/data/chenzh/AirTrans/results/air_trans_{}/result_voc_r50_{}way_{}shot_lr{}' \
        .format(loss_weights_index, way, shot, lr)
    model = AirTrans(
        # box_predictor params
        way, shot, is_flatten=True, roi_size=5, num_classes=way + 1,
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

    trainer_for_air_trans(way=way, shot=shot, query_batch=4, is_cuda=True, lr=lr, gpu_index=gpu_index,
                          root=root, json_path=json_path, img_path=img_path, split_cats=split_cats, model=model,
                          max_epoch=10, continue_epoch=None, continue_iteration=None, continue_weight=weight,
                          save_root=save_root, loss_weights=loss_weights)


if __name__ == '__main__':
    random.seed(4096)
    weight = "/data/chenzh/AirTrans/results/air_trans_20230905_flatten_aux/result_voc_r50_5way_5shot_lr0.002/weights/AirTrans_60_1305.pth"
    way_shot_train(5, 5, 2e-04, loss_weights0, 1, '20230905_flatten_aux_novel', split_cats=novel_ids_voc1)
