import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from functools import partial


class DownConvNormAct(nn.Sequential):
    def __init__(self,
                 in_features,
                 out_features,
                 norm: nn.Module = nn.BatchNorm2d,
                 act: nn.Module = nn.ReLU,
                 kernel_size: int = 3,
                 **kwargs
                 ):
        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
            ),
            norm(out_features),
            act(),
        )


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)


class ConvNorm(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm: nn.Module = nn.BatchNorm2d,
        **kwargs
    ):

        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            norm(out_features)
        )


Conv1X1Bn = partial(ConvNorm, kernel_size=1)




def cal_gradient(input, sobel_kernel):
    in_channels = input.shape[1]
    kernel_size = 3

    sobel_kernel = sobel_kernel.view(1, 1, kernel_size, kernel_size).to(input.device)
    # depthwise convolution needs sobel_kernel.shape[1] == in_channels / groups
    sobel_kernel = sobel_kernel.expand(in_channels, 1, kernel_size, kernel_size)
    # when groups == in_channels: conv2d equals to depthwise convolution
    gradient = F.conv2d(input, weight=sobel_kernel, padding=kernel_size // 2, groups=in_channels)
    return gradient


class ResidualGradientConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualGradientConv, self).__init__()
        self.smooth = 1e-8
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1 // 2)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

        self.x_sobel_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.y_sobel_kernel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)

    def forward(self, input):
        features = self.conv3x3(input)
        gradient_x = cal_gradient(input, self.x_sobel_kernel)
        gradient_y = cal_gradient(input, self.y_sobel_kernel)
        gradient = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2) + self.smooth)
        gradient = self.conv1x1(gradient)
        gradient = self.bn_1(gradient)
        features = features + gradient
        features = self.bn_2(features)
        features = self.relu(features)
        return features


class DownResidualGradientConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownResidualGradientConv, self).__init__()
        self.smooth = 1e-8

        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=3 // 2)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=1 // 2)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

        self.x_sobel_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.y_sobel_kernel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)

    def forward(self, input):
        features = self.conv3x3(input)

        gradient_x = cal_gradient(input, self.x_sobel_kernel)
        gradient_y = cal_gradient(input, self.y_sobel_kernel)
        gradient = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2) + self.smooth)
        gradient = self.conv1x1(gradient)
        gradient = self.bn_1(gradient)

        features = features + gradient
        features = self.bn_2(features)
        features = self.relu(features)
        return features


class ResidualGradientBlock(nn.Module):
    def __init__(self, in_channels, multiplier=2, use_pool=True):
        super(ResidualGradientBlock, self).__init__()
        self.blocks = nn.Sequential(
            ResidualGradientConv(in_channels, 64 * multiplier),
            ResidualGradientConv(64 * multiplier, 96 * multiplier),
            ResidualGradientConv(96 * multiplier, 64 * multiplier),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) if use_pool else None

    def forward(self, input):
        pre_pool_features = self.blocks(input)
        features = self.maxpool(pre_pool_features) if self.maxpool is not None else pre_pool_features
        return features, pre_pool_features


class PoolBlock(nn.Module):
    def __init__(self, in_channels=128, reduc_num=32, frame_len=5, use_pool=True):
        super(PoolBlock, self).__init__()
        self.reduc_num = reduc_num
        self.frame_len = frame_len
        self.use_pool = use_pool
        self.conv1x1 = nn.Conv2d(in_channels, reduc_num, kernel_size=1, padding=1 // 2)
        self.bn = nn.BatchNorm2d(reduc_num)
        self.conv3x3 = nn.Conv2d(288, in_channels, kernel_size=3, padding=3 // 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.x_sobel_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.y_sobel_kernel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)

    def forward(self, input):
        bs, channels, height, weight = input.shape
        bs = bs // self.frame_len
        feature = self.conv1x1(input)
        feature = self.bn(feature)
        feature_reshape = feature.view(bs, self.frame_len, self.reduc_num, height, weight)
        gradient_x = cal_gradient(feature, self.x_sobel_kernel)
        gradient_x = gradient_x.view(bs, self.frame_len, self.reduc_num, height, weight)
        gradient_y = cal_gradient(feature, self.y_sobel_kernel)
        gradient_y = gradient_y.view(bs, self.frame_len, self.reduc_num, height, weight)
        temporal_gradient = feature_reshape[:, :-1, :, :, :] - feature_reshape[:, 1:, :, :, :]
        pre_pool_features = input.view(bs, self.frame_len, -1, height, weight)
        pool_features = torch.cat([
            pre_pool_features[:, :-1, :, :, :],
            gradient_x[:, :-1, :, :, :],
            gradient_y[:, :-1, :, :, :],
            gradient_x[:, 1:, :, :, :],
            gradient_y[:, 1:, :, :, :],
            temporal_gradient
        ], dim=2)
        pool_features_batch = pool_features.view(-1, pool_features.shape[2], height, weight)
        res_features = self.conv3x3(pool_features_batch)
        if self.use_pool:
            res_features = self.maxpool(res_features)
        return res_features


def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''

    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]

    kernel_filter = np.array(kernel_filter_list, np.float32)

    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1).to(input.device)

    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1], input.shape[2])

    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv

    return contrast_depth


class ContrastDepthLoss(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(ContrastDepthLoss, self).__init__()
        return

    def forward(self, out, label):
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)

        criterion_MSE = nn.MSELoss()

        loss = criterion_MSE(contrast_out, contrast_label)
        # loss = torch.pow(contrast_out - contrast_label, 2)
        # loss = torch.mean(loss)

        return loss

