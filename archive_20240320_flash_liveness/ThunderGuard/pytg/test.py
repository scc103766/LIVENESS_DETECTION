import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from config import set_test_args
import networks
from networks import load_checkpoint
from dataset import TGDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

USE_CUDA = torch.cuda.is_available()
args = set_test_args()
if not USE_CUDA:
    args.num_workers = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

network = getattr(networks, args.network)

if __name__ == '__main__':
    test_data_set = TGDataset(TGDataset.get_images(os.path.join(args.data_path, "test")), is_train=False)
    test_data_loader = DataLoader(test_data_set, 1, shuffle=False)

    # 构造模型
    torch.manual_seed(123)
    model = getattr(network, args.model)(**{"infer_type": args.infer_type})

    if os.path.exists(os.path.join(args.model_path, args.model, "model_best.pth.tar")):
        res = load_checkpoint(os.path.join(args.model_path, args.model, "model_best.pth.tar"), model)
        val_threshold = res["val_threshold"]
        if args.threshold < 0:
            print("set threshold:", val_threshold)
            args.threshold = val_threshold
    else:
        print("没有模型！")
        assert False
    model = model.to(device)
    model.eval()
    p_list, l_list = [], []
    flag_list = []
    for step, batch_data in tqdm(enumerate(test_data_loader)):
        if USE_CUDA:
            batch_data = [d.to(device) for d in batch_data]
        normal_cue, sc_flag, depth_map_label, label = batch_data
        normal_cue = torch.reshape(normal_cue, [-1, 3, 256, 256])
        sc, p = model(normal_cue)

        # 验证颜色验证是否通过
        sc = torch.reshape(sc, [-1, 3]).detach().cpu().numpy()
        sc_flag = torch.reshape(sc_flag, [-1, 3]).detach().cpu().numpy()
        flag = 1  # 验证通过就是1
        for sp, l in zip(sc, sc_flag):
            for i in range(3):
                if np.min(l * (sp - 0.5)) < 0:
                    flag = 0
                    break
            if flag == 0:
                break

        pred = p.detach().cpu().item()
        label = label.detach().cpu().item()

        p_list.append(pred)
        l_list.append(label)
        flag_list.append(flag)

    p_list = np.array(p_list)
    l_list = np.array(l_list)
    flag_list = np.array(flag_list)
    id_list = np.arange(len(flag_list))

    print("正样本合格率：", len(flag_list[(flag_list > 0.1) & (l_list > 0.1)]) / len(flag_list[l_list > 0.1]))
    print("负样本合格率：", len(flag_list[(flag_list > 0.1) & (l_list < 0.1)]) / len(flag_list[l_list < 0.1]))

    p_list = p_list[flag_list > 0.1]
    l_list = l_list[flag_list > 0.1]
    id_list = id_list[flag_list > 0.1]

    # 模型预测为正的假样本
    FP = 0
    # 模型预测为负的真样本
    FN = 0
    # 预测为正或负的数量
    P1, P0 = 0, 0
    # 数量
    cnt = 0
    threshold = args.threshold
    for (id, score, label) in zip(id_list, p_list, l_list):
        if score > threshold:
            P1 += 1
        else:
            P0 += 1
        if score > threshold and label < 1:
            print(test_data_set.images[id][1])
            FP += 1
        if score <= threshold and label > 0:
            print(test_data_set.images[id][1])
            FN += 1
        cnt += 1
    print("预测数：", cnt, " 预测为真数：", P1, " 预测为假数：", P0)
    print("误识率：", FP / np.sum(1-l_list))
    print("拒真率：", FN / np.sum(l_list))
