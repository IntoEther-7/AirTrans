# -*- coding:utf-8 -*-
"""
@Editor: Ether
@Time: 2023/9/3 11:47
@File: voc_eval
@Contact: 211307040003@hhu.edu.cn
@Version: 1.0
@Description: None
"""
import os.path

from pycocotools.cocoeval import COCOeval
import json
from util.dataset import *

novel = [i for i in range(1, 801)]


def eval_voc(category_list, gt_path, dt_path):
    print(dt_path)
    # 处理阶段START-----------------------------------
    img_ids = []
    with open(dt_path, 'r') as f:
        j = json.load(f)
        for obj in j:
            if not obj['image_id'] in img_ids:
                img_ids.append(obj['image_id'])
    with open(gt_path, 'r') as f:
        j = json.load(f)
        images, type_value, annotations, categories = j.values()
        del_image = []
        del_ann = []
        for img in images:
            if not img['id'] in img_ids:
                del_image.append(img)
        for ann in annotations:
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # 修改这里
            if not ann['image_id'] in img_ids or not ann['category_id'] in category_list:
                del_ann.append(ann)
        for img in del_image:
            images.remove(img)
        for ann in del_ann:
            annotations.remove(ann)
    gt_path = 'tmp.json'
    with open('tmp.json', 'w') as f:
        json.dump(j, f)
    # 处理阶段END--------------------------------------
    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(dt_path)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


gt_path = "/home/chenzh/code/FRNOD/datasets/fsod/annotations/fsod_train.json"  # 存放真实标签的路径

if __name__ == "__main__":
    eval_voc(
        novel,
        gt_path,
        "/data/chenzh/AirTrans/results/20230910_conv/fsod_5way_5shot_lr0.002/validations/prediction.json")
    eval_voc(
        novel,
        gt_path,
        "/data/chenzh/AirTrans/results/20230915_flatten/fsod_5way_5shot/validations/prediction.json")
