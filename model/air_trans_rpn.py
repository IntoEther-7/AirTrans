# -*- coding:utf-8 -*-
"""
@Editor: Ether
@Time: 2023/9/2 19:04
@File: air_trans_rpn
@Contact: 211307040003@hhu.edu.cn
@Version: 1.0
@Description: None
"""
from typing import Dict, Optional, List, Tuple

import torch
from torch import Tensor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import RegionProposalNetwork, concat_box_prediction_layers
from torchvision.models.detection import _utils as det_utils
from torch.nn import functional as F
from tqdm import tqdm


class AirTransRPN(RegionProposalNetwork):
    def forward(
            self,
            images: ImageList,
            features: Dict[str, Tensor],
            targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (OrderedDict[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        lst = []
        for f in features:
            lst.append(f.mean(0))
        features = lst  # [n, m, c, w, h] -> [m, c, w, h]
        objectness, pred_bbox_deltas = self.head(features)  # [n, m, c, s, s] -> [m, c, s, s]
        anchors = self.anchor_generator.forward(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses
