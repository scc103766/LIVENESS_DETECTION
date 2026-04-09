import os
import argparse
import random

import cv2
import shutil
from tqdm import tqdm
import numpy as np
import math

from libtg import TG

parser = argparse.ArgumentParser()
# parser.add_argument('--opt', type=str, default="choose", help="命令")
# parser.add_argument('--from_path', type=str, default="../data/raw/train", help='原始文件路径')
# parser.add_argument('--to_path', type=str, default="../data/choose/train", help="生成文件路径")

# parser.add_argument('--opt', type=str, default="align", help="命令")
# parser.add_argument('--from_path', type=str, default="../data/FAS_choose", help='原始文件路径')
# parser.add_argument('--to_path', type=str, default="../data/FAS_align", help="生成文件路径")

parser.add_argument('--opt', type=str, default="normal", help="命令")
parser.add_argument('--sample_time', type=int, default=5, help="取样次数")
parser.add_argument('--from_path', type=str, default="../data/align/train", help='原始文件路径')
parser.add_argument('--to_path', type=str, default="../data/normal/train", help="生成文件路径")
parser.add_argument('--check_path', type=str, default="", help="生成文件路径")

parser.add_argument('--video_type', type=str, default="avi", help="视频类型")

args = parser.parse_args()

if len(args.to_path) == 0:
    args.to_path = args.from_path

COLOR_255 = (255 << 16) + (255 << 8) + 255


def read_frame_list(path):
    cap = cv2.VideoCapture(path)
    frame_list = []
    ret, frame = cap.read()
    while ret:
        frame_list.append(frame)
        ret, frame = cap.read()
    return frame_list


def read_img_list(path):
    file_list = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            file_list.append(file)
    file_list.sort()
    frame_list = [cv2.imread(os.path.join(path, file)) for file in file_list]
    return frame_list


def read_frame_line_list(path):
    frame_line_list = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            t = list(map(lambda a: int(a), line.split(',')))
            if len(t) == 0:
                continue
            frame_line_list.append(t)
            line = f.readline()
    return frame_line_list


def filter_frame_list(frame_list, frame_line_list):
    if len(frame_list) != len(frame_line_list):
        return [], [], []
    # 过滤变色时交界的帧
    new_frame_list, new_frame_line_list, frame_id_list = [], [], []
    
    for i in range(len(frame_list)):
        if i < 2 or i > (len(frame_list) - 3):
            continue
        if frame_line_list[i - 2][-1] != frame_line_list[i][-1] or frame_line_list[i + 2][-1] != frame_line_list[i][-1]:
            # 当前是交界帧
            continue
        new_frame_list.append(frame_list[i])
        new_frame_line_list.append(frame_line_list[i])
        frame_id_list.append(i)
    return new_frame_list, new_frame_line_list, frame_id_list


def random_video_face_choose(frame_list, frame_line_list, export_file):
    # 过滤交界帧
    frame_list, frame_line_list, frame_id_list = filter_frame_list(frame_list, frame_line_list)

    pre_color = -1
    # 选择5张图
    id_list = []
    pre_id_list = []

    cur_color_frame_list = []

    def refresh_cur():
        if len(cur_color_frame_list) == 0:
            return
        f_id, pre_f_id = random.choice(cur_color_frame_list)
        cur_color_frame_list.clear()
        id_list.append(f_id)
        pre_id_list.append(pre_f_id)

    for frame_id, (frame_line, pre_frame_id) in enumerate(zip(frame_line_list, frame_id_list)):
        color = frame_line[-1]
        if color != pre_color:
            refresh_cur()
        else:
            cur_color_frame_list.append((frame_id, pre_frame_id))
        pre_color = color
    refresh_cur()

    # 开始对齐
    if len(id_list) != 5:
        print("%s 帧数不对 %d" % (export_file, len(id_list)))
    # assert len(id_list) == 5
    dir = os.path.join(args.to_path, export_file.replace(".txt", ""))
    if not os.path.exists(dir):
        os.mkdir(dir)
        with open(dir + ".txt", "w") as f:
            for id, (frame_id, pre_frame_id) in enumerate(zip(id_list, pre_id_list)):
                frame = frame_list[frame_id]
                color = frame_line_list[frame_id][-1]
                cv2.imwrite(os.path.join(dir, "%d.png" % id), frame)
                f.write("%d,%d\n" % (pre_frame_id, color))


def video_face_choose(frame_list, frame_line_list, export_file):
    # 过滤交界帧
    frame_list, frame_line_list, frame_id_list = filter_frame_list(frame_list, frame_line_list)

    pre_color = -1
    # 选择5张图
    id_list = []
    pre_id_list = []

    for frame_id, (frame, frame_line, pre_frame_id) in enumerate(zip(frame_list, frame_line_list, frame_id_list)):
        color = frame_line[-1]
        if color != pre_color:
            # 变色
            if frame_id == 0 or color == 0:
                # 直接插入，并记录亮度
                state = 0
            else:
                state = 1
        else:
            if color == COLOR_255:
                state = 2
            elif color == 0:
                state = 3
            else:
                state = 4
        if tg.is_insert_image(frame, color, state):
            # print(state, "选择", pre_frame_id)
            if len(id_list) == 0 or pre_color != color:
                id_list.append(frame_id)
                pre_id_list.append(pre_frame_id)
            else:
                id_list[-1] = frame_id
                pre_id_list[-1] = pre_frame_id
        pre_color = color

    # 开始对齐
    if len(id_list) != 5:
        print("%s 帧数不对 %d" % (export_file, len(id_list)))
    # assert len(id_list) == 5
    dir = os.path.join(args.to_path, export_file.replace(".txt", ""))
    if not os.path.exists(dir):
        os.mkdir(dir)
        with open(dir + ".txt", "w") as f:
            for id, (frame_id, pre_frame_id) in enumerate(zip(id_list, pre_id_list)):
                frame = frame_list[frame_id]
                color = frame_line_list[frame_id][-1]
                cv2.imwrite(os.path.join(dir, "%d.png" % id), frame)
                f.write("%d,%d\n" % (pre_frame_id, color))


def fill_frame_line_list(frame_line_list, target_len):
    if len(frame_line_list) == 5:
        new_frame_line_list = []
        scale = target_len // 5
        offset = (target_len - scale * 5) >> 1
        for i in range(5):
            cur_scale = scale
            if i == 0:
                cur_scale = scale + offset
            elif i == 4:
                cur_scale = target_len - len(new_frame_line_list)
            for j in range(cur_scale):
                new_frame_line_list.append(frame_line_list[i])
        return new_frame_line_list
    return frame_line_list


def main():
    if args.opt == 'choose':
        # 从视频中选择最合适的5张照片
        for file in tqdm(os.listdir(args.from_path)):
            if file.endswith(".txt"):
                if len(file.split('_')) != 6:
                    print("文件命名错误：", file)
                    continue
                avi_path = os.path.join(args.from_path, file.replace(".txt", ".%s" % args.video_type))
                txt_path = os.path.join(args.from_path, file)
                if os.path.exists(avi_path):
                    if args.sample_time == 1:
                        if os.path.exists(os.path.join(args.to_path, file)):
                            # 表示已经生成
                            continue
                        if len(args.check_path) > 0 and os.path.exists(os.path.join(args.check_path, file)):
                            # 表示已经生成在其他目录
                            continue
                        frame_list = read_frame_list(avi_path)
                        frame_line_list = fill_frame_line_list(frame_line_list, len(frame_list))
                        video_face_choose(frame_list, frame_line_list, file)
                    else:
                        frame_list = None
                        for i in range(args.sample_time):
                            sfile = str(i+1) + "_" + file
                            if os.path.exists(os.path.join(args.to_path, sfile)):
                                # 表示已经生成
                                continue
                            if len(args.check_path) > 0 and os.path.exists(os.path.join(args.check_path, sfile)):
                                # 表示已经生成在其他目录
                                continue
                            if frame_list is None:
                                frame_list = read_frame_list(avi_path)
                                frame_line_list = read_frame_line_list(txt_path)
                                frame_line_list = fill_frame_line_list(frame_line_list, len(frame_list))
                            random_video_face_choose(frame_list, frame_line_list, sfile)
    elif args.opt == 'normal':
        for file in tqdm(os.listdir(args.from_path)):
            if file.endswith(".txt") and not os.path.exists(os.path.join(args.to_path, file)):
                png_path = os.path.join(args.from_path, file.replace(".txt", ".png"))
                txt_path = os.path.join(args.from_path, file)
                img = np.array(cv2.imread(png_path, cv2.IMREAD_UNCHANGED), dtype=np.uint8)
                color_list = []
                with open(txt_path, 'r') as f:
                    line = f.readline()
                    while line:
                        color_list.append(int(line.split(',')[-1]))
                        line = f.readline()
                if len(color_list) != 5:
                    print("color error:", file)
                    continue
                normal_cue = tg.cal_normal_cues_map(img, color_list)
                if normal_cue is not None:
                    cv2.imwrite(os.path.join(args.to_path, file.replace(".txt", ".jpg")), normal_cue)
                    with open(os.path.join(args.to_path, file), 'w') as f:
                        for color in color_list[1:-1]:
                            f.write(str(color) + '\n')

                    # 处理深度图
                    depth_map_path = png_path.replace(".png", "_d.jpg")
                    if os.path.exists(depth_map_path):
                        map_x = np.array(cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis], dtype=np.uint8)
                        height, width = map_x.shape[:2]
                        offset = (width - 288) // 2
                        depth_list = []
                        for i in range(1, 4):
                            cur_map_x = map_x[i * width: (i + 1) * width, :, :]
                            cur_map_x = cur_map_x[offset:offset + 288, offset:offset + 288, :]
                            depth_list.append(cur_map_x)
                        cv2.imwrite(os.path.join(args.to_path, file.replace(".txt", "_d.jpg")),
                                    np.concatenate(depth_list, axis=0))


if __name__ == '__main__':
    tg = TG("../data/ovalMask.jpg")
    main()
