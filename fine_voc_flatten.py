# -*- coding:utf-8 -*-
"""
@Editor: Ether
@Time: 2023/9/2 19:08
@File: train_voc
@Contact: 211307040003@hhu.edu.cn
@Version: 1.0
@Description: None
"""
from curr_model import *
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
    save_root = get_save_root(loss_weights_index, way, shot, 'voc', lr)
    model = get_model(way, shot, True)

    trainer_for_air_trans(way=way, shot=shot, query_batch=4, is_cuda=True, lr=lr, gpu_index=gpu_index,
                          root=root, json_path=json_path, img_path=img_path, split_cats=split_cats, model=model,
                          max_epoch=10, continue_epoch=None, continue_iteration=None, continue_weight=weight,
                          save_root=save_root, loss_weights=loss_weights)


if __name__ == '__main__':
    random.seed(4096)
    weight = "/data/chenzh/AirTrans/results/air_trans_20230905_flatten_32->100/result_voc_r50_5way_5shot_lr0.002/weights/AirTrans_60_1305.pth"
    way_shot_train(5, 5, 2e-04, loss_weights0, 1, 'fine_flatten', split_cats=novel_ids_voc1)
