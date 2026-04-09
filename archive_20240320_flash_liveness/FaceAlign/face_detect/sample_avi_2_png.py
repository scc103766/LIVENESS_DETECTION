import os
import argparse
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--from_path', type=str, default="../data/source", help='原始文件路径')
parser.add_argument('--to_path', type=str, default="../data/choose", help="生成文件路径")
parser.add_argument('--sample_num', type=int, default=5, help="取样间隔，间隔太小没意义")
parser.add_argument('--frame_num', type=int, default=9, help="取样帧数")
args = parser.parse_args()


def read_frame_list(path):
    cap = cv2.VideoCapture(path)
    frame_list = []
    ret, frame = cap.read()
    while ret:
        frame_list.append(frame)
        ret, frame = cap.read()
    return frame_list


def read_frame_line_list(path):
    frame_line_list = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                # frame_line_list.append(list(map(lambda a: int(a), line.split(','))))
                frame_line_list.append(line)
                line = f.readline()
    return frame_line_list


def choose_img(frame_list, frame_line_list, export_path):
    dis = len(frame_list) // args.sample_num
    start_id = (dis // 2) - (args.sample_num // 2)
    if start_id <= 0:
        print(export_path, "取样失败")

    if args.sample_num == 1:
        # 如果只是取样一帧，就全部放在一个目录
        os.mkdir(export_path)
        with open(export_path + ".txt", "w") as f:
            for id in range(args.sample_num):
                frame = frame_list[start_id]
                cv2.imwrite(os.path.join(export_path, "%d.png" % (id + 1)), frame)
                f.write(frame_line_list[start_id])
                start_id += dis
    else:
        file = os.path.basename(export_path)
        for id in range(args.sample_num):
            cur_export_path = export_path.replace(file, "%d_%s" % (id + 1, file))
            sub_frame_list = frame_list[start_id: start_id + args.frame_num]
            sub_frame_line_list = frame_line_list[start_id: start_id + args.frame_num]
            if len(sub_frame_list) == args.frame_num:
                if os.path.exists(cur_export_path):
                    continue
                else:
                    os.mkdir(cur_export_path)
                    with open(cur_export_path + ".txt", "w") as f:
                        for j, (frame, frame_line) in enumerate(zip(sub_frame_list, sub_frame_line_list)):
                            cv2.imwrite(os.path.join(cur_export_path, "%d.png" % (j + 1)), frame)
                            f.write(frame_line)
            start_id += dis


def is_path_exist(export_path):
    if args.sample_num == 1:
        if os.path.exists(export_path):
            # 目录存在，表示已经生成
            return True
    else:
        file = os.path.basename(export_path)
        for id in range(args.sample_num):
            cur_export_path = export_path.replace(file, "%d_%s" % (id + 1, file))
            if not os.path.exists(cur_export_path):
                return False
        return True


def main():
    # 从视频中选择最合适的5张图片
    for file in tqdm(os.listdir(args.from_path)):
        if file.endswith(".avi"):
            avi_path = os.path.join(args.from_path, file)
            txt_path = avi_path.replace(".avi", ".txt")
            export_path = os.path.join(args.to_path, file.replace(".avi", ""))

            if is_path_exist(export_path):
                continue

            frame_list = read_frame_list(avi_path)
            frame_line_list = read_frame_line_list(txt_path)
            if len(frame_list) == len(frame_line_list):
                choose_img(frame_list, frame_line_list, export_path)
            else:
                print("视频长度和描述文件长度不匹配")


if __name__ == '__main__':
    main()
