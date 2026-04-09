import numpy as np
from .res_u_net import ResUNet
from .inverted_residual import *
import torch
# 在TGAN的基础上加上交互信息


class DTGAN(nn.Module):
    def __init__(self, infer_type=None, frame_num=6, **kwargs):
        super(DTGAN, self).__init__()
        self.infer_type = infer_type
        self.frame_num = frame_num
        # [3, 256, 256]->[64, 64, 64]
        self.head_stem = nn.Sequential(
            DownConvNormAct(6, 32),
            DownConvNormAct(32, 64),
        )

        # 注意力值
        # 1、用来融合（预测出的）多帧深度图成一帧，然后再分类
        # 2、用来计算多帧的置信度，置信度低的帧，loss权重会小
        self.attention_stem = nn.Sequential(
            DownConvNormAct(6, 32),
            DownConvNormAct(32, 64),
            Conv1X1Bn(64, 1),
        )

        self.u_net = ResUNet()
        depth_map_cor = np.reshape(np.arange(256) / 255., [1, 1, 1, -1]).astype(np.float32)
        self.depth_map_cof = torch.from_numpy(depth_map_cor)

        # 输出分类
        self.f_net = nn.Sequential(
            DownConvNormAct(1, 16),  # 32*32*16
            ConvNormAct(16, 8, 3),  # 32*32*8
            ConvNormAct(8, 4, 3),  # 32*32*4
            Reshape(32 * 32 * 4),
            # L2Normalize(1),
            # nn.Linear(32 * 32 * 4, 2),
        )
        self.c_net = nn.Linear(32 * 32 * 4, 2)
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

        # 输出样本置信度，置信度。
        # 1、应该保证大多数样本置信度尽可能高
        # 2、置信度低的帧，loss权重小
        # self.confidence_net = nn.Sequential(
        #     DownConvNormAct(self.frame_num, 32, kernel_size=7),  # [32, 32, 64]
        #
        #     torch.nn.AdaptiveAvgPool2d((1, 1)),  # [1, 1, 32]
        #     Reshape(32),
        #
        #     nn.Linear(32, 16),
        #     nn.Linear(16, 1),
        #     nn.Sigmoid()
        # )

    def to(self, device=None):
        self.depth_map_cof = self.depth_map_cof.to(device)
        return super(DTGAN, self).to(device)

    def get_cross_x(self, x):
        x = torch.reshape(x, [-1, self.frame_num, 3, 256, 256])
        lx = x[:, :self.frame_num - 1, :, :, :] - x[:, 1:self.frame_num, :, :, :]
        rx = x[:, self.frame_num - 1:self.frame_num, :, :, :] - x[:, 0:1, :, :, :]
        dx = torch.cat((lx, rx), dim=1)

        x = torch.reshape(x, [-1, 3, 256, 256])
        dx = torch.reshape(dx, [-1, 3, 256, 256])
        x = torch.cat((x, dx), dim=1)
        return x

    def forward(self, x):
        x = self.get_cross_x(x)

        depth_map_attention = self.attention_stem(x)
        depth_map_attention = torch.softmax(torch.reshape(depth_map_attention, [-1, self.frame_num, 64, 64]), dim=1)
        x = self.head_stem(x)

        # 做深度图---------------------------------------------------------------------
        depth_x = self.u_net(x)
        depth_soft_max = self.pixel_wise_softmax(depth_x)
        depth_map = torch.unsqueeze(torch.sum(self.depth_map_cof * depth_soft_max, dim=-1), dim=1)

        # 融合多帧深度图成单帧
        single_depth_map = torch.sum(
            torch.reshape(depth_map, [-1, self.frame_num, 64, 64]) * depth_map_attention, dim=1,
            keepdim=True)

        if self.infer_type == "depth":
            return single_depth_map.permute(0, 2, 3, 1)
        # 做分类-----------------------------------------------------------------------
        feature = self.f_net(single_depth_map)
        norm_embed, corr = self.norm_n_corr(feature)
        single_p = torch.reshape(self.c_net(feature), [-1, 2])
        single_pred = torch.softmax(single_p, dim=1)[:, 1]

        # 做闪光颜色回归-----------------------------------------------------------------
        sc = torch.reshape(self.r_net(x), [-1, 3])
        sc_p = torch.cat((sc[:, 0:1] - sc[:, 1:2], sc[:, 1:2] - sc[:, 2:3], sc[:, 2:3] - sc[:, 0:1]), 1)

        if self.infer_type == "score":
            return torch.sigmoid(sc_p), single_pred
        else:
            return sc_p, torch.log(depth_soft_max + 1e-5), single_p, single_pred, depth_map_attention, corr

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                print("copy value to %s" % name)
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

    @classmethod
    def pixel_wise_softmax(cls, x):
        # 将像素交换到最后的维度
        x = x.permute(0, 2, 3, 1)
        channel_max, _ = torch.max(x, dim=3, keepdim=True)
        exponential_map = torch.exp(x - channel_max)
        normalize = torch.sum(exponential_map, dim=3, keepdims=True)
        return exponential_map / (normalize + 1e-5)

    def norm_n_corr(self, x):
        norm = torch.norm(x, 2, dim=1, keepdim=True)
        norm_embed = torch.div(x, norm + 1e-5)
        corr = (torch.matmul(norm_embed.squeeze(), norm_embed.squeeze().T) + 1.) / 2.
        return norm_embed, corr
