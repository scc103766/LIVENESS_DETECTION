import torch
import torch.nn as nn


def get_loss_module(device):
    # 颜色回归损失（仅仅针对正样本）
    loss_sc = nn.BCEWithLogitsLoss(reduction="none")
    # 分类损失（仅仅针对颜色验证通过的样本）
    single_loss_cls = nn.CrossEntropyLoss()
    # 伪深度损失（仅仅针对颜色验证通过的样本）
    loss_depth = nn.NLLLoss(reduction="none")
    # 门的开关损失
    loss_gate = nn.CrossEntropyLoss(reduction="none")

    loss_sc = loss_sc.to(device)
    single_loss_cls = single_loss_cls.to(device)
    loss_depth = loss_depth.to(device)
    loss_gate = loss_gate.to(device)
    return loss_sc, single_loss_cls, loss_depth, loss_gate


def build_loss(batch_data, loss_list, model, device, data_type=None):
    batch_data = [d.to(device) for d in batch_data]
    normal_cue, sc_flag, depth_map_label, single_label = batch_data
    normal_cue = torch.reshape(normal_cue, [-1, 3, 256, 256])

    sc_label = torch.reshape(torch.clamp(sc_flag, 0, 1), [-1]).float()
    sc_label_w_1 = torch.reshape(torch.abs(sc_flag), [-1]).float()
    # 仅对正样本回归
    sc_label_w_2 = torch.reshape(torch.reshape(single_label, [-1, 1]).repeat(1, 6 * 3), [-1]).float()
    sc_label_w = sc_label_w_1 * sc_label_w_2

    depth_map_label = torch.reshape(depth_map_label, [-1]).long()

    single_label = torch.reshape(single_label, [-1])
    label = torch.reshape(single_label, [-1, 1]).repeat(1, 6)
    label = torch.reshape(label, [-1])

    sc_p, depth_log_softmax, single_p, single_pred, depth_map_attention, gate = model(normal_cue)

    depth_map_attention = torch.reshape(depth_map_attention, [-1, 64 * 64])
    sc_p = torch.reshape(sc_p, [-1])
    depth_log_softmax = torch.reshape(depth_log_softmax, [-1, 256])

    loss_sc, single_loss_cls, loss_depth, loss_gate = loss_list
    if data_type == "print":
        # label=0表示印刷品
        loss_3 = torch.mean(loss_gate(gate, label) * (1 - label.float()))
    elif data_type == "screen":
        # label=0表示电子屏
        loss_3 = torch.mean(loss_gate(gate, 1 - label) * (1 - label.float()))
    elif data_type == "model":
        # label=0表示3d攻击
        loss_3 = torch.mean(loss_gate(gate, 2 - label) * (1 - label.float()))
    else:
        loss_3 = None
    loss_0 = torch.sum(loss_sc(sc_p, sc_label) * sc_label_w) / (torch.sum(sc_label_w) + 1e-6)
    loss_1 = single_loss_cls(single_p, single_label.long())

    # 伪深度损失，如果颜色验证不通过（sc_flag==0），就不计算损失
    depth_map_loss = torch.mean(torch.reshape(loss_depth(depth_log_softmax, depth_map_label), [-1, 64 * 64]) * depth_map_attention, dim=1)
    depth_w = label.float() * 0.0 + 1.0
    loss_2 = torch.sum(depth_map_loss * depth_w) / (torch.sum(depth_w) + 1e-6)

    loss_dict = {
        "loss": loss_0 + loss_1 * 0.16 + loss_2 * 6 + (0 if loss_3 is None else loss_3),
        "color_loss": loss_0,
        "score_loss": loss_1,
        "map_loss": loss_2,
    }
    if loss_3 is not None:
        loss_dict["gate_loss"] = loss_3
    return loss_dict, single_pred, single_label
