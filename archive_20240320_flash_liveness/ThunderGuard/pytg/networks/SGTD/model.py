import numpy as np
from .base_layer import *
import torch
from torchvision import transforms
import torch.nn.functional as F


class SGTD(nn.Module):
    def __init__(self, infer_type=None, frame_num=6, num_blocks=3, **kwargs):
        super(SGTD, self).__init__()
        self.infer_type = infer_type
        self.frame_num = frame_num
        # [3, 256, 256]->[64, 64, 64]
        self.head_stem = nn.Sequential(
            DownConvNormAct(3, 32),
            DownConvNormAct(32, 64),
        )

        # 注意力值
        # 1、用来融合（预测出的）多帧深度图成一帧，然后再分类
        # 2、用来计算多帧的置信度，置信度低的帧，loss权重会小
        self.attention_stem = nn.Sequential(
            DownConvNormAct(64, 32),
            Conv1X1Bn(32, 1),
        )

        # 输出颜色序列分类
        self.r_net = nn.Sequential(
            DownConvNormAct(64, 64, kernel_size=7),  # [32, 32, 64]

            # 如果有AdaptiveAvgPool2d算子，就用此语句
            torch.nn.AdaptiveAvgPool2d((1, 1)),  # [1, 1, 64]
            Reshape(64),

            # 如果没有AdaptiveAvgPool2d算子，就用此语句
            # Reshape(32*32, 64),
            # Mean(1),

            nn.Linear(64, 32),
            nn.Linear(32, 3),
        )

        self.Init_RsGB = DownResidualGradientConv(3, 64)
        self.RsGBs = nn.Sequential(
            *[ResidualGradientBlock(in_channels=(64 if i == 0 else 128), multiplier=2, use_pool=(i > 0)) for i in
              range(num_blocks)]
        )

        self.transform = transforms.Resize((32, 32))

        self.concat_RsGB = nn.Sequential(
            ResidualGradientConv(64 * 6, 64 * 2),
            ResidualGradientConv(64 * 2, 64 * 1)
        )
        self.conv3x3 = nn.Conv2d(64 * 1, 1, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        head_x = self.head_stem(x)
        depth_map_attention = self.attention_stem(head_x)
        depth_map_attention = torch.softmax(torch.reshape(depth_map_attention, [-1, self.frame_num, 32, 32]), dim=1)

        # 做深度图----------------------------------------------------------------------
        batchsize, channels, height, weight = x.shape
        # batchsize = batchsize // self.frame_num
        pool_list, pre_pool_list = [], []
        feature = self.Init_RsGB(x)  # [bs, channel, 128, 128]
        for blocks in self.RsGBs:
            feature, pre_pool_feature = blocks(feature)
            pool_list.append(feature)
            pre_pool_list.append(pre_pool_feature)

        feature1 = self.transform(pool_list[0])
        feature2 = self.transform(pool_list[1])

        feature = pool_list[2]

        pool_concat = torch.cat([feature1, feature2, feature], dim=1)
        feature = self.concat_RsGB(pool_concat)
        depth_map = self.conv3x3(feature)

        # 融合多帧深度图成单帧
        single_depth_map = torch.sum(
            torch.reshape(depth_map, [-1, self.frame_num, 32, 32]) * depth_map_attention, dim=1,
            keepdim=True)

        if self.infer_type == "depth":
            return single_depth_map.permute(0, 2, 3, 1)

        # 做分类-----------------------------------------------------------------------
        single_pred = torch.clamp(torch.mean(torch.reshape(single_depth_map, [-1, 32 * 32]), dim=1, keepdim=True), 1e-5, 0.9999)

        # 做闪光颜色回归-----------------------------------------------------------------
        sc = torch.reshape(self.r_net(head_x), [-1, 3])
        sc_p = torch.cat((sc[:, 0:1] - sc[:, 1:2], sc[:, 1:2] - sc[:, 2:3], sc[:, 2:3] - sc[:, 0:1]), 1)

        if self.infer_type == "score":
            return torch.sigmoid(sc_p), single_pred
        else:
            return sc_p, depth_map, depth_map_attention, single_pred
