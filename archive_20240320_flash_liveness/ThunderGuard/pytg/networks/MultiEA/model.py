import numpy as np
from .res_u_net import ResUNet
from .easy_res_u_net import EasyResUNet
from .inverted_residual import *
import torch


class MultiEA(nn.Module):
    def __init__(self, infer_type=None, frame_num=6, **kwargs):
        super(MultiEA, self).__init__()
        self.infer_type = infer_type
        self.frame_num = frame_num
        # [3, 256, 256]->[64, 64, 64]
        self.head_stem = nn.Sequential(
            DownConvNormAct(3, 32),
            DownConvNormAct(32, 64),
        )

        self.pos_embedding = torch.nn.Parameter(torch.randn(1, 64, 64, 64))

        # 注意力值
        # 1、用来融合（预测出的）多帧深度图成一帧，然后再分类
        self.attention_stem = nn.Sequential(
            DownConvNormAct(3, 16),
            DownConvNormAct(16, 32),
        )
        self.easy_u_net_attention = EasyResUNet()

        # 两个任务
        self.u_net_print = ResUNet()
        self.u_net_screen = ResUNet()

        depth_map_cor = np.reshape(np.arange(256) / 255., [1, 1, 1, -1]).astype(np.float32)
        self.depth_map_cof = torch.from_numpy(depth_map_cor)

        # 输出分类
        # self.f_net_print = nn.Sequential(
        #     DownConvNormAct(1, 16),  # 32*32*16
        #     ConvNormAct(16, 8, 3),  # 32*32*8
        #     ConvNormAct(8, 4, 3),  # 32*32*4
        #     Reshape(32 * 32 * 4),
        #     L2Normalize(1),
        #     nn.Linear(32 * 32 * 4, 2),
        # )
        # self.f_net_screen = nn.Sequential(
        #     DownConvNormAct(1, 16),  # 32*32*16
        #     ConvNormAct(16, 8, 3),  # 32*32*8
        #     ConvNormAct(8, 4, 3),  # 32*32*4
        #     Reshape(32 * 32 * 4),
        #     L2Normalize(1),
        #     nn.Linear(32 * 32 * 4, 2),
        # )
        self.f_net = nn.Sequential(
            DownConvNormAct(1, 16),  # 32*32*16
            ConvNormAct(16, 8, 3),  # 32*32*8
            ConvNormAct(8, 4, 3),  # 32*32*4
            Reshape(32 * 32 * 4),
            L2Normalize(1),
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

    def to(self, device):
        # self.pos_embedding = self.pos_embedding.to(device)
        self.depth_map_cof = self.depth_map_cof.to(device)
        return super(MultiEA, self).to(device)

    # tast_id：
    #   0：印刷品任务
    #   1：电子屏任务
    #   2：综合任务
    def forward(self, x, task_id=2):
        atten_x = self.attention_stem(x)
        depth_map_attention = self.infer_depth_map_attention(atten_x)
        if self.infer_type == "attention":
            return depth_map_attention.permute(1, 2, 3, 0)

        x = self.head_stem(x) + self.pos_embedding

        # 做深度图---------------------------------------------------------------------
        single_depth_map_print, depth_soft_max_print, depth_map_print = None, None, None
        if task_id == 0 or task_id == 2:
            single_depth_map_print, depth_soft_max_print, depth_map_print = self.infer_depth_map(self.u_net_print, x, depth_map_attention)

        single_depth_map_screen, depth_soft_max_screen, depth_map_screen = None, None, None
        if task_id == 1 or task_id == 2:
            single_depth_map_screen, depth_soft_max_screen, depth_map_screen = self.infer_depth_map(self.u_net_screen, x, depth_map_attention)

        if self.infer_type == "depth":
            if task_id == 0:
                depth_map = depth_map_print
            else:
                depth_map = depth_map_screen
            return depth_map.permute(0, 2, 3, 1)
        elif self.infer_type == "sdepth":
            if task_id == 0:
                single_depth_map = single_depth_map_print
            else:
                single_depth_map = single_depth_map_screen
            return single_depth_map.permute(0, 2, 3, 1)

        # 做分类-----------------------------------------------------------------------
        single_p_print = torch.reshape(self.f_net(single_depth_map_print), [-1, 2]) if (
                    task_id == 0 or task_id == 2) else None
        single_pred_print = torch.softmax(single_p_print, dim=1)[:, 1] if single_p_print is not None else None

        single_p_screen = torch.reshape(self.f_net(single_depth_map_screen), [-1, 2]) if (
                    task_id == 1 or task_id == 2) else None
        single_pred_screen = torch.softmax(single_p_screen, dim=1)[:, 1] if single_p_screen is not None else None

        # 做闪光颜色回归-----------------------------------------------------------------
        sc = torch.reshape(self.r_net(x), [-1, 3])
        sc_p = torch.cat((sc[:, 0:1] - sc[:, 1:2], sc[:, 1:2] - sc[:, 2:3], sc[:, 2:3] - sc[:, 0:1]), 1)

        if self.infer_type == "score":
            single_pred = torch.cat(
                [torch.unsqueeze(single_pred_print, dim=1), torch.unsqueeze(single_pred_screen, dim=1)], dim=1)
            single_pred, _ = torch.min(single_pred, dim=1)
            return torch.sigmoid(sc_p), single_pred
        elif task_id == 0:
            return sc_p, torch.log(depth_soft_max_print + 1e-5), single_p_print, single_pred_print, depth_map_attention
        elif task_id == 1:
            return sc_p, torch.log(
                depth_soft_max_screen + 1e-5), single_p_screen, single_pred_screen, depth_map_attention
        else:
            depth_soft_max = (torch.log(depth_soft_max_print + 1e-5) + torch.log(depth_soft_max_screen + 1e-5)) * 0.5
            single_p = (single_p_print + single_p_screen) * 0.5
            single_pred = torch.cat(
                [torch.unsqueeze(single_pred_print, dim=1), torch.unsqueeze(single_pred_screen, dim=1)], dim=1)
            single_pred, _ = torch.min(single_pred, dim=1)
            return sc_p, depth_soft_max, single_p, single_pred, depth_map_attention

    def infer_depth_map_attention(self, x):
        attention_x = self.easy_u_net_attention(x)
        attention_soft_max = self.pixel_wise_softmax(attention_x)
        attention_map = torch.unsqueeze(torch.sum(self.depth_map_cof * attention_soft_max, dim=-1), dim=1)
        attention_map = torch.softmax(torch.reshape(attention_map, [-1, self.frame_num, 64, 64]), dim=1)
        return attention_map

    def infer_depth_map(self, u_net, x, depth_map_attention):
        depth_x = u_net(x)
        depth_soft_max = self.pixel_wise_softmax(depth_x)
        depth_map = torch.unsqueeze(torch.sum(self.depth_map_cof * depth_soft_max, dim=-1), dim=1)

        # 融合多帧深度图成单帧
        single_depth_map = torch.sum(
            torch.reshape(depth_map, [-1, self.frame_num, 64, 64]) * depth_map_attention, dim=1,
            keepdim=True)
        return single_depth_map, depth_soft_max, depth_map

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
