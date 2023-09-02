# -*- coding:utf-8 -*-
"""
@Editor: Ether
@Time: 2023/9/2 19:04
@File: air_trans_head
@Contact: 211307040003@hhu.edu.cn
@Version: 1.0
@Description: None
"""
import torch
from torch import nn, Tensor
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torch.nn import functional as F


class AirTransBoxHead(TwoMLPHead):
    def __init__(self, in_channels, representation_size):
        super(AirTransBoxHead, self).__init__(in_channels, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class AirTransPredictHead(nn.Module):
    def __init__(self, way, shot, representation_size):
        super().__init__()
        self.way = way
        self.shot = shot
        self.representation_size = representation_size
        self.bbox_pred = nn.Linear(representation_size, 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.predict_bg_score = nn.Sequential(
            nn.Linear(representation_size, 1)
        )
        self.scale = nn.Parameter(torch.FloatTensor([3.0]), requires_grad=False)

    def forward(self, support, query, x):
        r"""

        :param support: [n, c, s, s], support_aggregate
        :param query: [box_num, c, s, s]
        :param x:
        :return:
        """
        # 回归
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)

        # 分类
        scores = self.cls_predictor(support, query, x)
        return scores, torch.cat([bbox_deltas for _ in range(self.way + 1)], dim=1)

    def cls_predictor(self, support: Tensor, boxes_features: Tensor, x: Tensor):
        s = self.encoder(support)
        s = s.unsqueeze(0)
        q = self.encoder(boxes_features)
        q = q.unsqueeze(1)
        metrics = (s - q).mean([2, 3, 4])  # [n, box_num]
        bg = self.predict_bg_score(x)
        score = torch.cat([bg, metrics], dim=1) * self.scale
        return score
