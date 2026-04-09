import numpy as np
from .res_u_net import ResUNet
from .inverted_residual import *
import torch


class TG(nn.Module):
    def __init__(self, infer_type=None, frame_num=6, **kwargs):
        super(TG, self).__init__()
        self.infer_type = infer_type
        self.frame_num = frame_num
        # [3, 256, 256]->[64, 64, 64]
        self.head_stem = nn.Sequential(
            DownConvNormAct(3, 32),
            DownConvNormAct(32, 64),
        )

        self.pos_embedding = torch.nn.Parameter(torch.randn(1, 64, 64, 64))

        self.u_net = ResUNet()
        depth_map_cor = np.reshape(np.arange(256) / 255., [1, 1, 1, -1]).astype(np.float32)
        self.depth_map_cof = torch.from_numpy(depth_map_cor)

        # 输出深度图
        self.depth_conv = nn.Sequential(
            nn.Conv2d(32, 1, 1, padding=0),
            # nn.ReLU()
            nn.Sigmoid()
        )
        # 输出分类
        self.f_net = nn.Sequential(
            DownConvNormAct(1, 16),  # 32*32*16
            ConvNormAct(16, 8, 3),  # 32*32*8
            ConvNormAct(8, 4, 3),  # 32*32*4
            Reshape(32 * 32 * 4),
            nn.Linear(32 * 32 * 4, 2),
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

    def to(self, device=None):
        self.depth_map_cof = self.depth_map_cof.to(device)
        return super(TG, self).to(device)

    def forward(self, x):
        x = self.head_stem(x) + self.pos_embedding

        # 做深度图---------------------------------------------------------------------
        depth_x = self.u_net(x)
        depth_soft_max = self.pixel_wise_softmax(depth_x)
        depth_map = torch.unsqueeze(torch.sum(self.depth_map_cof * depth_soft_max, dim=-1), dim=1)

        if self.infer_type == "depth":
            return depth_map.permute(0, 2, 3, 1)

        # 做分类-----------------------------------------------------------------------
        p = torch.reshape(self.f_net(depth_map), [-1, 2])
        # 预测时返回颜色预测和最后预测
        pred = torch.mean(torch.reshape(torch.softmax(p, dim=1)[:, 1], [-1, self.frame_num]), dim=1)

        # 做闪光颜色回归-----------------------------------------------------------------
        sc = torch.reshape(self.r_net(x), [-1, 3])
        sc_p = torch.cat((sc[:, 0:1] - sc[:, 1:2], sc[:, 1:2] - sc[:, 2:3], sc[:, 2:3] - sc[:, 0:1]), 1)
        if self.infer_type == "score":
            return torch.sigmoid(sc_p), pred
        else:
            return sc_p, torch.log(depth_soft_max + 1e-5), p, pred

    def load_state_dict(self, state_dict, strict=True):
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
