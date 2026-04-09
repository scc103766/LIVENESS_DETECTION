import torch
import torch.nn as nn
from .base_layer import ContrastDepthLoss


def get_loss_module(device):
    # 颜色回归损失（仅仅针对正样本）
    loss_sc = nn.BCEWithLogitsLoss(reduction="none")
    criterion_absolute_loss = nn.MSELoss()
    criterion_contrastive_loss = ContrastDepthLoss()
    loss_cls = nn.NLLLoss()

    loss_sc = loss_sc.to(device)
    criterion_absolute_loss = criterion_absolute_loss.to(device)
    criterion_contrastive_loss = criterion_contrastive_loss.to(device)
    loss_cls = loss_cls.to(device)
    return loss_sc, criterion_absolute_loss, criterion_contrastive_loss, loss_cls


def build_loss(batch_data, loss_list, model, device, data_type=None):
    batch_data = [d.to(device) for d in batch_data]
    normal_cue, sc_flag, map_label, single_label = batch_data
    normal_cue = torch.reshape(normal_cue, [-1, 3, 256, 256])

    sc_label = torch.reshape(torch.clamp(sc_flag, 0, 1), [-1]).float()
    sc_label_w_1 = torch.reshape(torch.abs(sc_flag), [-1]).float()
    # 仅对正样本回归
    sc_label_w_2 = torch.reshape(torch.reshape(single_label, [-1, 1]).repeat(1, 6 * 3), [-1]).float()
    sc_label_w = sc_label_w_1 * sc_label_w_2
    single_label = torch.reshape(single_label, [-1])

    sc_p, map_x, depth_map_attention, single_pred = model(normal_cue)
    sc_p = torch.reshape(sc_p, [-1])

    # 计算预测值
    spoofing_pred = torch.cat([1 - single_pred, single_pred], dim=1)

    loss_sc, criterion_absolute_loss, criterion_contrastive_loss, loss_cls = loss_list

    map_x = torch.reshape(map_x, [-1, 32, 32])
    map_label = torch.reshape(map_label, [-1, 32, 32]).float()
    absolute_loss = criterion_absolute_loss(map_x, map_label)
    contrastive_loss = criterion_contrastive_loss(map_x, map_label)
    map_loss = absolute_loss + 0.5 * contrastive_loss

    score_loss = loss_cls(torch.log(spoofing_pred), single_label.long())

    loss_0 = torch.sum(loss_sc(sc_p, sc_label) * sc_label_w) / (torch.sum(sc_label_w) + 1e-6)

    loss_dict = {
        "loss": loss_0 + map_loss + 0.01 * score_loss,
        "color_loss": loss_0,
        "score_loss": score_loss,
        "map_loss": map_loss,
    }
    return loss_dict, single_pred, single_label

