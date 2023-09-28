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
root = '/home/chenzh/code/FRNOD/datasets/fsod'
json_path = 'annotations/fsod_train.json'
img_path = 'images'
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
    save_root = get_save_root(loss_weights_index, way, shot, 'fsod', lr)

    model = get_model(way, shot, False)

    trainer_for_air_trans(way=way, shot=shot, query_batch=4, is_cuda=True, lr=lr, gpu_index=gpu_index,
                          root=root, json_path=json_path, img_path=img_path, split_cats=split_cats, model=model,
                          max_epoch=25, continue_epoch=None, continue_iteration=None, continue_weight=None,
                          save_root=save_root, loss_weights=loss_weights)


if __name__ == '__main__':
    random.seed(4096)
    way_shot_train(5, 5, 2e-03, loss_weights0, 0, 'conv', split_cats=None)
