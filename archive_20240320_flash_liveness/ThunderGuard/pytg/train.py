import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np
from dataset import TGDataset
from config import set_train_args
import matplotlib.pyplot as plt
from util.score_base import performances
import networks
from networks import load_checkpoint, save_checkpoint

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
USE_CUDA = torch.cuda.is_available()
args = set_train_args()

if args.local_rank >= 0:
    # 使用多卡支持
    #   a.根据local_rank来设定当前使用哪块GPU
    torch.cuda.set_device(args.local_rank)
    #   b.初始化DDP，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端。
    dist.init_process_group(backend='nccl')
    # device = torch.device("cuda", args.local_rank)
    device = args.local_rank
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 选择合适的模型
network = getattr(networks, args.network)

if not USE_CUDA:
    args.num_workers = 0

if not os.path.exists(os.path.join(args.model_path, args.model)):
    os.mkdir(os.path.join(args.model_path, args.model))


def show_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    print("finish")


def create_dataset(img_list, is_train, data_len):
    is_random_crop = (True if (args.network == "AG" or args.network == "SGTD") else False)
    if args.network == "SGTD":
        return TGDataset(img_list, is_train=is_train, is_random_crop=is_random_crop, map_size=args.mask_map_size,
                         data_len=data_len, is_mask=True,
                         neg_scale=args.neg_scale, is_resample=is_random_crop)

    return TGDataset(img_list, is_train=is_train, is_random_crop=is_random_crop, map_size=args.depth_map_size,
                     data_len=data_len, is_mask=False,
                     neg_scale=args.neg_scale, is_resample=is_random_crop)


if __name__ == '__main__':
    real_images = TGDataset.get_images(os.path.join(args.data_path, "train"), set([1]))
    print_images = TGDataset.get_images(os.path.join(args.data_path, "train"), set([2, 3, 7]))
    screen_images = TGDataset.get_images(os.path.join(args.data_path, "train"), set([4, 5, 7]))
    model_images = TGDataset.get_images(os.path.join(args.data_path, "train"), set([6, 7]))

    data_len = len(real_images) + len(print_images) + len(screen_images) + len(model_images)

    # 真人和印刷品数据集
    train_print_data_set = create_dataset(real_images + print_images, is_train=True, data_len=data_len)
    train_print_data_sampler = DistributedSampler(train_print_data_set) if args.local_rank >= 0 else None
    train_print_data_loader = DataLoader(train_print_data_set, args.batch_size, sampler=train_print_data_sampler,
                                         num_workers=args.num_workers)

    # 真人和电子屏数据集
    train_screen_data_set = create_dataset(real_images + screen_images, is_train=True, data_len=data_len)
    train_screen_data_sampler = DistributedSampler(train_screen_data_set) if args.local_rank >= 0 else None
    train_screen_data_loader = DataLoader(train_screen_data_set, args.batch_size, sampler=train_screen_data_sampler,
                                          num_workers=args.num_workers)

    # 真人和3D攻击数据集
    train_model_data_set = create_dataset(real_images + model_images, is_train=True, data_len=data_len)
    train_model_data_sampler = DistributedSampler(train_model_data_set) if args.local_rank >= 0 else None
    train_model_data_loader = DataLoader(train_model_data_set, args.batch_size, sampler=train_model_data_sampler,
                                          num_workers=args.num_workers)

    # 验证集
    dev_data_set = create_dataset(TGDataset.get_images(os.path.join(args.data_path, "test")), is_train=False,
                                  data_len=None)
    dev_data_loader = DataLoader(dev_data_set, args.batch_size, shuffle=False)

    # 构造模型
    torch.manual_seed(args.local_rank + 123)
    np.random.seed(args.local_rank + 123)
    # 得到使用的模型
    model = getattr(network, args.model)()


    # 降低训练显存占用---------------------------------------------------------------------
    def inplace_relu(m):
        classname = m.__class__.__name__
        if classname.find('ReLU') != -1:
            m.inplace = True


    model.apply(inplace_relu)

    # 得到使用是损失
    get_loss_module = getattr(network, "get_loss_module")
    build_loss = getattr(network, "build_loss")

    max_val_acc = 0
    if os.path.exists(os.path.join(args.model_path, args.model, "checkpoint.pth.tar")) and args.local_rank <= 0:
        res = load_checkpoint(os.path.join(args.model_path, args.model, "checkpoint.pth.tar"), model)
        max_val_acc = res["max_val_acc"]

    if args.local_rank >= 0:
        if dist.get_world_size() > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        else:
            model = model.to(device)
        # ddp支持
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        model = model.to(device)

    param_group = [
        {"params": [p for n, p in list(model.named_parameters()) if "f_net" not in n],
         "weight_decay": args.l2_weight},
        {"params": [p for n, p in list(model.named_parameters()) if "f_net" in n],
         "weight_decay": args.l2_weight * 10.0},
    ]

    if args.opt == "Adam":
        optimizer = optim.Adam(param_group, lr=args.lr, betas=(0.5, 0.999))
    else:
        optimizer = optim.SGD(param_group, lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    loss_loss_module = get_loss_module(device)

    for epoch in range(args.epoch_num):
        model.train()
        loss_sum_dict, cnt = {}, 0
        pbars = tqdm(enumerate(zip(train_print_data_loader, train_screen_data_loader, train_model_data_loader)))
        for step, (print_data, screen_data, model_data) in pbars:
            print_loss_dict, _, _ = build_loss(print_data, loss_loss_module, model, device, "print")
            screen_loss_dict, _, _ = build_loss(screen_data, loss_loss_module, model, device, "screen")
            model_loss_dict, _, _ = build_loss(model_data, loss_loss_module, model, device, "model")
            keys = [k for k, v in print_loss_dict.items()]
            loss_dict = {k: (print_loss_dict[k] + screen_loss_dict[k] + model_loss_dict[k]) * 0.5 for k in keys}
            for loss_name, loss_value in loss_dict.items():
                loss_sum_dict[loss_name] = loss_sum_dict.get(loss_name, 0.0) + loss_value.detach().cpu().item()
            cnt += 1
            loss_dict["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()  # 在下次计算以前将梯度归零
            pbars.set_description("%.4f" % (loss_sum_dict["loss"] / cnt))

        scheduler.step()
        loss_sum_dict = {key: value / cnt for key, value in loss_sum_dict.items()}
        print("train %d:" % args.local_rank, loss_sum_dict)

        if args.local_rank > 0:
            # 只在0卡或cpu测试
            continue
        model.eval()
        loss_sum_dict, cnt = {}, 0
        pbars = tqdm(enumerate(dev_data_loader))
        val_scores, val_labels = [], []
        with torch.no_grad():
            for step, batch_data in pbars:
                loss_dict, spoofing_pred, spoofing_label = build_loss(batch_data, loss_loss_module, model, device)
                for loss_name, loss_value in loss_dict.items():
                    loss_sum_dict[loss_name] = loss_sum_dict.get(loss_name, 0.0) + loss_value.detach().cpu().item()
                cnt += 1
                val_scores.append(np.reshape(spoofing_pred.detach().cpu().numpy(), [-1]))
                val_labels.append(np.reshape(spoofing_label.detach().cpu().numpy(), [-1]))

        loss_sum_dict = {key: value / cnt for key, value in loss_sum_dict.items()}
        val_scores = np.concatenate(val_scores, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_threshold, val_acc, val_apcer, val_bpcer, val_acer = performances(val_scores, val_labels)

        if val_acc >= max_val_acc:
            is_best = True
            max_val_acc = val_acc
        else:
            is_best = False
        val_dict = {
            "val_threshold": val_threshold,
            "val_acc": val_acc,
            "max_val_acc": max_val_acc,
            "val_apcer": val_apcer,
            "val_bpcer": val_bpcer,
            "val_acer": val_acer,
        }
        print("eval:", {**val_dict, **loss_sum_dict})
        if isinstance(model, DDP):
            # 如果是多卡上的模型，需要保存model.module
            model_dict = {"model_state_dict": model.module.state_dict()}
        else:
            model_dict = {"model_state_dict": model.state_dict()}

        save_checkpoint({**val_dict, **model_dict}, is_best=is_best,
                        filename=os.path.join(args.model_path, args.model, "checkpoint.pth.tar"))
