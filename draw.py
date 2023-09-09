# -*- coding:utf-8 -*-
"""
@Editor: Ether
@Time: 2023/9/3 23:45
@File: draw
@Contact: 211307040003@hhu.edu.cn
@Version: 1.0
@Description: None
"""
import json
import os.path

import numpy as np
from matplotlib import pyplot as plt

max_epoch = 40
loss_weights = {'loss_classifier': 1, 'loss_box_reg': 1,
                            'loss_objectness': 1, 'loss_rpn_box_reg': 1,
                            'loss_attention': 1, 'loss_aux': 1,
                            'loss_sum': 1}


def epoch_loss_process(loss_dict: dict):
    statistics_loss = {'loss_sum': []}
    loss_mean_this_epoch = {}
    for iteration, losses in loss_dict.items():
        loss_sum_iteration = 0
        for k, v in losses.items():
            if k not in statistics_loss.keys():
                statistics_loss[k] = []
            statistics_loss[k].append(v)
            loss_sum_iteration += v * loss_weights[k]
        # loss_sum_iteration /= len(statistics_loss)
        statistics_loss['loss_sum'].append(loss_sum_iteration)
    for k, v in statistics_loss.items():
        mean = np.array(v).mean()
        loss_mean_this_epoch.update({k: mean})
    statistics_loss.update({'x': [int(i) for i in list(loss_dict.keys())]})
    return statistics_loss, loss_mean_this_epoch


def draw(root):
    train_all_losses = {}
    val_all_losses = {}
    train_epoch_losses_mean = {}
    val_epoch_losses_mean = {}
    for epoch in range(1, max_epoch + 1):
        train_loss_path = os.path.join(root, 'train_loss_{}.json'.format(epoch))
        val_loss_path = os.path.join(root, 'val_loss_{}.json'.format(epoch))
        with open(train_loss_path, 'r') as f:
            train_loss_dict = json.load(f)
        with open(val_loss_path, 'r') as f:
            val_loss_dict = json.load(f)
        train_statistics_loss, train_loss_mean_this_epoch = epoch_loss_process(train_loss_dict)
        val_statistics_loss, val_loss_mean_this_epoch = epoch_loss_process(val_loss_dict)
        keys = list(train_loss_mean_this_epoch.keys())
        if epoch == 1:
            num = 0
        else:
            num += len(train_statistics_loss[k])
        x_val = [(x + num) for x in val_statistics_loss['x']]
        if 'x' not in val_all_losses.keys():
            val_all_losses['x'] = []
        val_all_losses['x'].extend(x_val)
        for k in keys:
            if k not in train_all_losses.keys():
                train_all_losses[k] = []
            train_all_losses[k].extend(train_statistics_loss[k])
            if k not in val_all_losses.keys():
                val_all_losses[k] = []
            val_all_losses[k].extend(val_statistics_loss[k])
            if k not in train_epoch_losses_mean.keys():
                train_epoch_losses_mean[k] = []
            train_epoch_losses_mean[k].append(train_loss_mean_this_epoch[k])
            if k not in val_epoch_losses_mean.keys():
                val_epoch_losses_mean[k] = []
            val_epoch_losses_mean[k].append(val_loss_mean_this_epoch[k])
    x_val = val_all_losses['x']
    for k in keys:
        y_train = train_all_losses[k]
        x_train = [i for i in range(1, len(y_train) + 1)]
        y_val = val_all_losses[k]
        fig = plt.figure(figsize=(12, 3), dpi=320.)
        plt.plot(x_train, y_train, linewidth=1)
        plt.plot(x_val, y_val, linewidth=1)
        plt.legend(['train_loss', 'val_loss'], loc='upper right')
        plt.draw()
        plt.savefig(os.path.join(root, '{}.png'.format(k)))
        plt.close(fig)
    x = [i for i in range(1, epoch + 1)]
    for k in keys:
        y_train = train_epoch_losses_mean[k]
        y_val = val_epoch_losses_mean[k]
        fig = plt.figure(figsize=(12, 3), dpi=320.)
        plt.plot(x, y_train, linewidth=1)
        plt.plot(x, y_val, linewidth=1)
        plt.legend(['train_loss', 'val_loss'], loc='upper right')
        plt.draw()
        plt.savefig(os.path.join(root, '{}_epoch.png'.format(k)))
        plt.close(fig)

if __name__ == '__main__':
    max_epoch = 10
    draw('/data/chenzh/AirTrans/results/air_trans_20230905_decoder_aux_novel/result_voc_r50_5way_5shot_lr0.0002/results')
