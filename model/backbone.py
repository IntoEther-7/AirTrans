# -*- coding:utf-8 -*-
"""
@Editor: Ether
@Time: 2023/9/2 19:06
@File: backbone
@Contact: 211307040003@hhu.edu.cn
@Version: 1.0
@Description: None
"""
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FeatureExtractor(nn.Module):

    def __init__(self,
                 # fpn backbone
                 backbone_name='resnet50', pretrained=True, returned_layers=None, trainable_layers=3):
        r"""

        :param backbone_name: 'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        :param pretrained: 是否预训练
        :param returned_layers: 返回那一层, [1, 4], 对应着c1-c4层, 返回
        :param trainable_layers: 训练哪几层, 从最后一层往前数
        """
        super(FeatureExtractor, self).__init__()
        if returned_layers is None:
            returned_layers = [3, 4]
        self.out_channels = 256
        self.s_scale = 16

        self.backbone = resnet_fpn_backbone(
            backbone_name=backbone_name,
            weights=ResNet50_Weights.IMAGENET1K_V2,
            trainable_layers=trainable_layers,
            returned_layers=returned_layers)  # (n, 256, x, x)

    def forward(self, x):
        r"""
        进行特征提取
        :param x_list: List[Tensor]
        :return: List[Tensor]
        """
        return self.backbone.forward(x)
