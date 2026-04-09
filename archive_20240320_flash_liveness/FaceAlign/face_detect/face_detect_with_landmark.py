import numpy as np
import torch
import torch.backends.cudnn as cudnn
from face_tool.models.net_rfb import RFB
from face_tool.layers.functions.prior_box import PriorBox
from face_tool.utils.box_utils import decode, decode_landm
from face_tool.utils.nms.py_cpu_nms import py_cpu_nms
import cv2
import os
import shutil
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=r"../data/choose", help="生成文件路径")
parser.add_argument('--meta_num', type=int, default=2, help="原始描述文件的长度，用来区分是否已经在描述文件中插入了双目坐标")
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class LandmarkFaceDetector(object):
    def __init__(self, trained_model, long_side=640, origin_size=False, threshold=0.7, top_k=100, keep_top_k=1, nms_threshold=0.4):
        self.long_side = long_side
        self.origin_size = origin_size
        self.threshold = threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.cfg = {
            'name': 'RFB',
            'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
            'steps': [8, 16, 32, 64],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'gpu_train': True,
            'batch_size': 32,
            'ngpu': 1,
            'epoch': 250,
            'decay1': 190,
            'decay2': 220,
            'image_size': 300
        }
        net = RFB(cfg=self.cfg, phase='test')
        net = load_model(net, trained_model, True)
        net.eval()
        cudnn.benchmark = True
        self.device = torch.device("cpu")
        net = net.to(self.device)
        self.net = net

    def predict_landmark(self, img_raw):
        landmark_array = np.empty([7, 2], dtype=np.int32)
        img = np.float32(img_raw)

        # testing scale
        target_size = self.long_side
        max_size = self.long_side
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if self.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape

        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        if len(scores) == 0:
            return None

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        b = list(map(int, dets[0]))
        landmark_array[0, 0] = b[0]
        landmark_array[0, 1] = b[1]
        landmark_array[1, 0] = b[2]
        landmark_array[1, 1] = b[3]

        lm = np.array(b[5:15]).reshape([5, 2])
        landmark_array[2:, :] = lm
        return landmark_array


def read_img_list(path):
    file_list = []
    for file in os.listdir(path):
        if file.endswith(".png") or file.endswith(".jpg"):
            tmp = file.split('.')
            file_list.append((int(tmp[0]), tmp[1]))
    file_list = sorted(file_list, key=lambda a: a[0])
    file_list = list(map(lambda a: "%d.%s" % (a[0], a[1]), file_list))
    frame_list = [cv2.imread(os.path.join(path, file)) for file in file_list]
    return frame_list


def read_img_line_list(path):
    img_line_list = []
    try:
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                # frame_line_list.append(list(map(lambda a: int(a), line.split(','))))
                img_line_list.append(line)
                line = f.readline()
    except Exception as e:
        print(path)
        return img_line_list
    return img_line_list


if __name__ == '__main__':
    face_detect = LandmarkFaceDetector('./face_tool/RBF_Final.pth')
    root = args.data_path
    for file in tqdm(os.listdir(root)):
        if os.path.isdir(os.path.join(root, file)):
            meta_file = os.path.join(root, file + ".txt")
            if not os.path.exists(meta_file):
                continue
            frame_line_list = read_img_line_list(meta_file)
            if len(frame_line_list) < 1 or len(frame_line_list[0].split(',')) > args.meta_num:
                # 如果已经插入双目坐标，就别再插入了
                continue
            frame_list = read_img_list(os.path.join(root, file))
            try:
                # 覆盖
                with open(os.path.join(root, file + ".txt"), 'w') as f:
                    for id, (frame, frame_line) in enumerate(zip(frame_list, frame_line_list)):
                        b = face_detect.predict_landmark(frame).reshape([-1])
                        pre_line_list = frame_line.split(',')
                        new_line_list = [pre_line_list[0], str(b[4]), str(b[5]), str(b[6]), str(b[7])]
                        for i in range(1, len(pre_line_list)):
                            new_line_list.append(pre_line_list[i])
                        line = ",".join(new_line_list)
                        f.write(line)
            except Exception as e:
                shutil.rmtree(os.path.join(root, file))
                # print(file, "error")
                pass