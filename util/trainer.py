# -*- coding:utf-8 -*-
"""
@Editor: Ether
@Time: 2023/9/2 19:09
@File: trainer
@Contact: 211307040003@hhu.edu.cn
@Version: 1.0
@Description: None
"""
import json
import os
import sys
import random

import torch
from tqdm import tqdm

from model.air_trans import AirTrans
from util.dataset import CocoDataset


def trainer_for_air_trans(
        # 基础参数
        way=5, shot=2, query_batch=16, is_cuda=True, lr=2e-02,
        # 设备参数
        gpu_index=0,
        # 数据集参数
        root=None, json_path=None, img_path=None, split_cats=None,
        # 模型
        model: AirTrans = None,
        # 训练轮数
        max_epoch=20,
        # 继续训练, 如果没有seed可能很难完美续上之前的训练, 不过整个流程随机, 可能也可以
        continue_epoch=None, continue_iteration=None, continue_weight=None,
        # 保存相关的参数
        save_root=None,
        # loss权重
        loss_weights=None
):

    # 检查参数
    # if loss_weights is None:
    #     loss_weights = {'loss_classifier': 0.995, 'loss_box_reg': 0.005,
    #                     'loss_objectness': 0.995, 'loss_rpn_box_reg': 0.005,
    #                     'loss_attention': 0.95, 'loss_aux': 0.05}
    assert root is not None, "root is None"
    assert json_path is not None, "json_path is none"
    assert img_path is not None, "img_path is none"
    assert (continue_iteration is None and continue_epoch is None) \
           or (continue_iteration is not None and continue_epoch is not None), \
        "continue_iteration and continue_epoch should be all None, or all not None"
    # 设置
    torch.set_printoptions(sci_mode=False)

    # 设置参数
    torch.cuda.set_device(gpu_index)

    # 生成数据集
    dataset = CocoDataset(root=root, ann_path=json_path, img_path=img_path,
                          way=way, shot=shot, query_batch=query_batch, is_cuda=is_cuda, catIds=split_cats)

    # 模型
    if model is None:
        model = AirTrans(
            # box_predictor params
            way, shot, roi_size=7, num_classes=way + 1,
            # backbone
            backbone_name='resnet50', pretrained=True,
            returned_layers=None, trainable_layers=3,
            # transform parameters
            min_size=600, max_size=1000,
            image_mean=None, image_std=None,
            # RPN parameters
            rpn_anchor_generator=None, rpn_head=None,
            rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=6000,
            rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=500,
            rpn_nms_thresh=0.7,
            rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
            rpn_score_thresh=0.0,
            # Box parameters
            box_roi_pool=None, box_head=None, box_predictor=None,
            box_score_thresh=0.05, box_nms_thresh=0.3, box_detections_per_img=100,
            box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
            box_batch_size_per_image=100, box_positive_fraction=0.25,
            bbox_reg_weights=(10., 10., 5., 5.)
        )

    if is_cuda:
        model.cuda()

    # 保存相关的参数
    save_weights = os.path.join(save_root, 'weights')
    save_results = os.path.join(save_root, 'results')

    # 创建文件夹保存此次训练
    if not os.path.exists(save_weights):
        os.makedirs(save_weights)
    if not os.path.exists(save_results):
        os.makedirs(save_results)

    # # 日志
    # now = datetime.now()
    # dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
    # log_print = open(os.path.join(save_root, '{}way_{}shot-{}.log'.format(way, shot, dt_string)), 'w')
    # sys.stdout = log_print

    # 训练轮数
    if continue_epoch is not None and continue_iteration is not None:
        continue_weight = os.path.join(save_root, 'weights',
                                       'AirTrans_{}_{}.pth'.format(continue_epoch, continue_iteration))
        weight = torch.load(continue_weight)
        model.load_state_dict(weight['models'])
        continue_done = False
    else:
        continue_epoch = 0
        continue_iteration = 0
        continue_done = True


    fine_epoch = int(max_epoch * 0.7)
    val_losses = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=fine_epoch, gamma=0.1)
    for epoch in range(max_epoch):
        if epoch + 1 < continue_epoch and continue_done is False:
            continue

        save_train_loss = os.path.join(save_results, 'train_loss_{}.json'.format(epoch + 1))
        save_val_loss = os.path.join(save_results, 'val_loss_{}.json'.format(epoch + 1))
        # 保存loss
        if not os.path.exists(save_train_loss):
            with open(save_train_loss, 'w') as f:
                json.dump({}, f)
        if not os.path.exists(save_val_loss):
            with open(save_val_loss, 'w') as f:
                json.dump({}, f)

        # 训练一个轮回
        dataset.initial()
        model.train()
        loss_dict_train = {}
        loss_dict_val = {}
        dataset.set_mode(is_training=True)
        pbar = tqdm(dataset)

        for index, item in enumerate(pbar):

            iteration = index + 1
            if iteration < continue_iteration and continue_done is False:
                continue
            elif iteration == continue_iteration:
                continue_done = True
            loss_this_iteration = {}
            val_loss_this_iteration = {}
            support, bg, query, query_anns, cat_ids = item

            # 训练
            result = model.forward(support, query, targets=query_anns)
            losses = 0
            sum = 0
            for k, v in result.items():
                w = loss_weights[k]
                losses += v * w
                sum += v
                loss_this_iteration.update({k: float(v)})
            scale = model.roi_heads.box_predictor.scale.exp()[0]
            tqdm.write(
                '{:2} / {:3} / {:.6f} / {:.6f} / {}'.format(epoch + 1, iteration, (float(sum)), float(scale), result))

            loss_this_iteration = {iteration: loss_this_iteration}
            loss_dict_train.update(loss_this_iteration)

            if torch.isnan(losses).any() or torch.isinf(losses).any() or losses > 50000:
                print('梯度炸了!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                sys.exit(0)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # 验证
            if (index + 1) % 5 == 0:
                support, bg, query, query_anns, cat_ids = dataset.get_val(
                    random.randint(1, len(dataset.val_iteration)) - 1)
                result = model.forward(support, query, targets=query_anns)
                val_losses = 0
                for k, v in result.items():
                    val_losses += v
                    val_loss_this_iteration.update({k: float(v)})
                loss_this_epoch = {index + 1: val_loss_this_iteration}
                loss_dict_val.update(loss_this_epoch)
                # 信息展示
            postfix = {'epoch': '{:2}/{:2}'.format(epoch + 1, max_epoch),
                       'mission': '{:4}/{:4}'.format(index + 1, len(pbar)),
                       'catIds': cat_ids,
                       '模式': 'train',
                       'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                       'train_loss': "%.6f" % float(losses),
                       'val_loss': "%.6f" % float(val_losses)}

            pbar.set_postfix(postfix)

            # 保存loss与权重
            if iteration % 100 == 0 or (index + 1) == len(dataset):  # 记得改
                with open(save_train_loss, 'r') as f:
                    tmp_loss_dict = json.load(f)
                with open(save_train_loss, 'w') as f:
                    tmp_loss_dict.update(loss_dict_train)
                    loss_dict_train = {}
                    json.dump(tmp_loss_dict, f)
                torch.save({'models': model.state_dict()},
                           os.path.join(save_weights, 'AirTrans_{}_{}.pth'.format(epoch + 1, iteration)))

                with open(save_val_loss, 'r') as f:
                    tmp_loss_dict = json.load(f)
                with open(save_val_loss, 'w') as f:
                    tmp_loss_dict.update(loss_dict_val)
                    loss_dict_val = {}
                    json.dump(tmp_loss_dict, f)

            iteration += 1

        lr_scheduler.step()
