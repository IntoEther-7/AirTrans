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
    def __init__(self, way, shot, representation_size, roi_size, is_flatten):
        super().__init__()
        self.way = way
        self.shot = shot
        self.representation_size = representation_size
        self.bbox_pred = nn.Linear(representation_size, 4)
        self.is_flatten = is_flatten

        self.encoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=roi_size, stride=1, padding=0),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.encoder_flatten = nn.Sequential(
            nn.Flatten(),  # [num, c, s, s] -> # [num, c * s * s]
            nn.Linear(256 * roi_size * roi_size, 1024),
            nn.Linear(1024, 64),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True))

        self.decoder = nn.Sequential(
            nn.Linear(6, 6),
            nn.LeakyReLU(inplace=True),
            nn.Sigmoid()
        )  # [0.5, 1]

        self.predict_bg_score = nn.Sequential(
            nn.Linear(representation_size, 1),
        )
        self.scale = nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True)

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
        if self.is_flatten:
            distance = self.cls_predictor_flatten(support, query, x)
        else:
            distance = self.cls_predictor(support, query, x)
        scores = self.metric(distance)
        return scores, torch.cat([bbox_deltas for _ in range(self.way + 1)], dim=1)

    def metric(self, distance):
        distance = self.decoder(distance)
        distance = 2 * (distance - 0.5)
        confidence = distance.neg()
        score = confidence * self.scale.exp()
        return score

    def cls_predictor_flatten(self, support: Tensor, boxes_features: Tensor, x: Tensor):
        s = self.encoder_flatten(support)
        s = s.unsqueeze(0)
        q = self.encoder_flatten(boxes_features)
        q = q.unsqueeze(1)
        fg_distance = (s - q).mean(2).pow(2)  # [box_num, n, 512]
        # fg_distance = (s - q).mean([2, 3, 4])  # [box_num, n]
        bg_distance = self.predict_bg_score(x)  # [box_num, 1]
        distance = torch.cat([fg_distance, bg_distance], dim=1)  # [box_num, n + 1]
        return distance

    def cls_predictor(self, support: Tensor, boxes_features: Tensor, x: Tensor):
        s = self.encoder(support)
        s = s.unsqueeze(0)
        q = self.encoder(boxes_features)
        q = q.unsqueeze(1)
        fg_distance = (s - q).mean([2, 3, 4]).pow(2)  # [box_num, n]
        bg_distance = self.predict_bg_score(x)  # [box_num, 1]
        distance = torch.cat([fg_distance, bg_distance], dim=1)  # [box_num, n + 1]
        return distance
