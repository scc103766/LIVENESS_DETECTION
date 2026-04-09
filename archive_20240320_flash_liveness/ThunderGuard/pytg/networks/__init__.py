import networks.AG as AG
import networks.TGA as TGA
import networks.MoE as MoE
import networks.MoEA as MoEA
import networks.MultiEA as MultiEA
import os
import shutil
import torch


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        dir_name = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(dir_name, 'model_best.pth.tar'))


def load_checkpoint(filename, model=None):
    try:
        checkpoint = torch.load(filename)
    except:
        # load weight saved on gpy device to cpu device
        # see https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/3
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint
