import argparse
import os
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--from_path', type=str, default="../../../E/oulu_align/train", help='原始文件路径')
parser.add_argument('--to_path', type=str, default="../../../E/oulu_align/train_jpg", help="生成文件路径")
args = parser.parse_args()

if __name__ == '__main__':
    for file in tqdm(os.listdir(args.from_path)):
        if file.endswith(".png"):
            i_img_path = os.path.join(args.from_path, file)
            img = cv2.imread(i_img_path, cv2.IMREAD_UNCHANGED)

            o_img_path = os.path.join(args.to_path, file.replace(".png", ".jpg"))
            cv2.imwrite(o_img_path, img[:, :, :3])

            o_img_path = os.path.join(args.to_path, file.replace(".png", "_d.jpg"))
            cv2.imwrite(o_img_path, img[:, :, 3:])
