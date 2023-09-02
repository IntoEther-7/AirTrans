# -*- coding:utf-8 -*-
"""
@Editor: Ether
@Time: 2023/9/2 19:05
@File: attention
@Contact: 211307040003@hhu.edu.cn
@Version: 1.0
@Description: None
"""
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class FullCrossAttentionModule(nn.Module):
    def __init__(self, way, shot, channel, roi_size):
        super().__init__()
        self.way = way
        self.shot = shot

        self.encoder = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1),
            nn.Conv2d(channel, channel, 1, 1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 1, 1, 1),
            nn.Sigmoid()
        )
        self.aggregate_support_module = nn.Sequential(
            nn.Flatten(start_dim=1),  # [n * k, c, s, s] -> [n * k, c * s * s]
            nn.Linear(channel * roi_size ** 2, channel),  # [n * k, c * s * s] -> [n * k, channel]
            nn.Linear(channel, 1),  # [n * k, channel] -> [n * k, 1]
        )

    def forward(self, support: Tensor, query: Tensor):
        r"""

        :param support: [n * k, c, s, s]
        :param query: [m, c, w, h]
        :param image_tensors:
        :param target:
        :return: [n, m, c, s, s], [n, c, s, s]
        """
        out = {}
        sup = {}
        sup_w = self.support_weight(self.encoder(support['pool']))

        for k in query.keys():
            s = self.encoder(support[k])
            q = self.encoder(query[k])
            out[k], sup[k] = self.one_layer(s, q, sup_w)
        # [n, m, c, s, s]
        return out, sup

    def one_layer(self, support: Tensor, query: Tensor, sup_w):
        r"""

        :param support: [n * k, c, s, s]
        :param query: [m, c, w, h]
        :return: [n, m, c, s, s]
        """
        support_w = self.support_aggregate(support, sup_w)  # [n * k, c ,s, s] -> [n, c, s, s]

        # 对某个way和某个query激活
        lst = []
        for i in range(self.way):
            # [m, c, s, s]
            spatial = self.spatial_attention(support_w[i:i + 1, :, :, :], query)
            spatial_attention = self.decoder(spatial)
            lst.append(spatial_attention * query)

        attention_query = torch.stack(lst, dim=0)  # [n, m, 1, s, s]
        return attention_query, support_w

    def spatial_attention(self, support: Tensor, query: Tensor):
        r"""

        :param support: [1, c, s, s]
        :param query: [m, c, w, h]
        :return: [m, 1, w, h]
        """

        n, c, s1, s2 = support.shape
        if s1 % 2 == 0:
            p1 = s1 // 2
            p2 = s1 // 2 - 1
        else:
            p1 = (s1 - 1) // 2
            p2 = (s1 - 1) // 2
        if s2 % 2 == 0:
            p3 = s2 // 2
            p4 = s2 // 2 - 1
        else:
            p3 = (s2 - 1) // 2
            p4 = (s2 - 1) // 2
        q = F.pad(query, [p1, p2, p3, p4])
        spatial = F.conv2d(q, support)  # [m, 1, w, h]
        return spatial

    def support_weight(self, support: Tensor):
        r"""
        加权support聚合
        :param support: [n * k, c ,s, s]
        :return: [n, c, s, s]
        """
        sup = self.aggregate_support_module(support)  # [n * k, 1]
        # [n * k, 1] -> [n, k]
        sup = sup.reshape([self.way, self.shot])
        sup_w = sup.softmax(1)  # [n, k]
        # sup = support.reshape([self.way, self.shot, c, s1, s2]) * sup_w.reshape([self.way, self.shot, 1, 1, 1])
        # sup = sup.sum(1)  # [n, c, s, s]
        return sup_w

    def support_aggregate(self, support, sup_w):
        r"""
        加权
        :param support: [n, k, c, s, s]
        :param sup_w: [n, k]
        :return: [n, c, s, s]
        """
        _, c, s1, s2 = support.shape
        sup = support.reshape([self.way, self.shot, c, s1, s2]) * sup_w.reshape([self.way, self.shot, 1, 1, 1])
        return sup.sum(1)
