import os
import argparse
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--from_path', type=str, default=r"D:\share\ThunderGuard\data\test\test\raw", help='原始文件路径')
parser.add_argument('--to_path', type=str, default=r"D:\share\ThunderGuard\data\test\test\choose", help="生成文件路径")
args = parser.parse_args()


def read_frame_list(path):
    cap = cv2.VideoCapture(path)
    frame_list = []
    ret, frame = cap.read()
    while ret:
        frame_list.append(frame)
        ret, frame = cap.read()
    return frame_list


def main():
    # 从视频中选择最合适的5张图片
    for file in tqdm(os.listdir(args.from_path)):
        if file.endswith(".avi"):
            avi_path = os.path.join(args.from_path, file)
            # txt_path = avi_path.replace(".avi", ".txt")
            export_path = os.path.join(args.to_path, file.replace(".avi", ""))
            os.mkdir(export_path)
            frame_list = read_frame_list(avi_path)
            for id, frame in enumerate(frame_list):
                cv2.imwrite(os.path.join(export_path, "%d.png" % id), frame)


if __name__ == '__main__':
    main()
