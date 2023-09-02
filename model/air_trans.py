# -*- coding:utf-8 -*-
"""
@Editor: Ether
@Time: 2023/9/2 19:03
@File: air_trans
@Contact: 211307040003@hhu.edu.cn
@Version: 1.0
@Description: None
"""
import warnings
from collections import OrderedDict
from typing import List, Tuple

import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from model.air_trans_head import AirTransBoxHead, AirTransPredictHead
from model.air_trans_rpn import AirTransRPN
from model.attention import FullCrossAttentionModule
from model.backbone import FeatureExtractor
from model.roi_head import AirTransRoIHeads


class AirTrans(GeneralizedRCNN):
    def __init__(self,
                 # box_predictor params
                 way, shot, roi_size,
                 num_classes=None,
                 # backbone
                 backbone_name='resnet50', pretrained=False,
                 returned_layers=None, trainable_layers=4,
                 # transform parameters
                 min_size=600, max_size=1000,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=64, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 rpn_focal=False, head_focal=False):
        r"""

        :param way:
        :param shot:
        :param roi_size:
        :param num_classes:
        :param backbone_name:
        :param pretrained:
        :param returned_layers: 默认[3, 4]
        :param trainable_layers:
        :param min_size:
        :param max_size:
        :param image_mean:
        :param image_std:
        :param rpn_anchor_generator:
        :param rpn_head:
        :param rpn_pre_nms_top_n_train:
        :param rpn_pre_nms_top_n_test:
        :param rpn_post_nms_top_n_train:
        :param rpn_post_nms_top_n_test:
        :param rpn_nms_thresh:
        :param rpn_fg_iou_thresh:
        :param rpn_bg_iou_thresh:
        :param rpn_batch_size_per_image:
        :param rpn_positive_fraction:
        :param rpn_score_thresh:
        :param box_roi_pool:
        :param box_head:
        :param box_predictor:
        :param box_score_thresh:
        :param box_nms_thresh:
        :param box_detections_per_img:
        :param box_fg_iou_thresh:
        :param box_bg_iou_thresh:
        :param box_batch_size_per_image:
        :param box_positive_fraction:
        :param bbox_reg_weights:
        """
        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        backbone = FeatureExtractor(backbone_name, pretrained=pretrained, returned_layers=returned_layers,
                                        trainable_layers=trainable_layers)
        out_channels = backbone.out_channels
        channels = out_channels

        # transform
        if image_mean is None:
            image_mean = [0., 0., 0.]
        if image_std is None:
            image_std = [1, 1, 1]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        # RPN
        if rpn_anchor_generator is None:
            anchor_sizes = ((128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn: RegionProposalNetwork = AirTransRPN(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh
        )

        # Head
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0'],
                output_size=roi_size,
                sampling_ratio=2)
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = AirTransBoxHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = AirTransPredictHead(way, shot, representation_size)
        roi_heads: AirTransRoIHeads = AirTransRoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)
        super(AirTrans, self).__init__(backbone, rpn, roi_heads, transform)
        self.way = way
        self.shot = shot
        self.resolution = roi_size ** 2
        self.roi_size = roi_size
        self.support_transform = GeneralizedRCNNTransform(320, 320, image_mean, image_std)
        self.attention: FullCrossAttentionModule = FullCrossAttentionModule(way, shot, channels, roi_size)
        self.rpn_focal = rpn_focal
        self.head_focal = head_focal

    def forward(self, support, images, targets=None):
        r"""

        :param support: [tensor(3, w, h)]
        :param images: [tensor(3, w, h)]
        :param targets: [Dict{'boxes': tensor(n, 4), 'labels': tensor(n,)}, 'image_id': int, 'category_id': int, 'id': int]
        :return:
        """
        # 校验及预处理
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                            boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        support, _ = self.support_transform(support)
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # 特征提取, 列成字典
        support = self.backbone.forward(support.tensors)  # (way * shot, channels, h, w)
        features = self.backbone.forward(images.tensors)  # (n, channels, h, w)

        # 注意力
        # [n, m, c, s, s]
        attention_f, support_aggregate = self.attention.forward(support, features)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        proposals, proposal_losses = self.rpn.forward(images, attention_f, targets)
        # AirTransRoIHeads
        detections, detector_losses, support = self.roi_heads.forward(
            support_aggregate,
            features,
            proposals,
            images.image_sizes,
            targets)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)


        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)


def min_max_scale(ten: torch.Tensor):
    n, c, w, h = ten.shape
    tmp = ten.permute(1, 0, 2, 3).reshape(c, n * w * h)
    _min = torch.reshape(tmp.min(1).values, [1, c, 1, 1])
    _max = torch.reshape(tmp.max(1).values, [1, c, 1, 1])
    # _min = ten.min()
    # _max = ten.max()
    ten = (ten - _min) / (_max - _min)
    print(f'{ten.min(1).values} {ten.max(1).values}')
    # print(f'{ten.min()} {ten.max()}')
    # assert ten.min() == 0 and ten.max() == 1, f'{ten.min()} {ten.max()}'
    return ten
