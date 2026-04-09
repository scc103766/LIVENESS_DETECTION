import argparse
import numpy as np
import os
from PIL import Image
from libinfer import TGInfer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="../pytg/resources", help='模型位置')
parser.add_argument("--infer_type", type=str, default="", help="推理类型，仅仅在生产onnx时使用，可以选择depth、score")
parser.add_argument("--normal_cues_path", type=str, default="../data/sample/test", help="推理需要输入的法线信息图")
args = parser.parse_args()


def infer(threshold=0.5):
    # 颜色预测数量和命中数量
    color_cnt = 0
    color_hit = 0
    # 模型预测为正的假样本
    FP = 0
    # 模型预测为负的真样本
    FN = 0
    # 预测为正或负的数量
    P1, P0 = 0, 0
    # 数量
    cnt = 0

    for file in tqdm(os.listdir(args.normal_cues_path)):
        label = None
        if file.startswith("1_"):
            label = 1
        elif file.startswith("0_"):
            label = 0
            continue
        if label is not None:
            sample_path = os.path.join(args.normal_cues_path, file)
            normal_cue_list = []
            color_list = []
            cur_path_list = []
            for sample_file in os.listdir(sample_path):
                if sample_file.endswith("_g.png") or not sample_file.endswith(".png"):
                    continue
                cur_path = os.path.join(sample_path, sample_file)
                # print(cur_path)
                pil_normal_cue = Image.open(cur_path)
                normal_cue = np.asarray(pil_normal_cue)
                normal_cue = normal_cue[16:288-16, 16:288-16, :3].astype(np.uint8)
                normal_cue_list.append(normal_cue)
                cur_path_list.append(cur_path)
                color = list(map(lambda a: int(a), sample_file.replace(".png", "").split("_")))
                color = (color[0] << 16) + (color[1] << 8) + color[2]
                color_list.append(color)
            score = tg_infer.predict(normal_cue_list, color_list)
            if score is None:
                print("推理失败！")
                continue
            if score >= 0:
                # 颜色验证成功
                color_hit += 1
            color_cnt += 1

            # 仅仅对颜色校验通过的样本做后续工作
            if score >= 0:
                if score > threshold:
                    P1 += 1
                else:
                    P0 += 1
                if score > threshold and label < 1:
                    FP += 1
                    print("误识：", cur_path)
                if score <= threshold and label > 0:
                    FN += 1
                    print("拒真：", cur_path)
                cnt += 1
    print("颜色验证准确率：", color_hit / color_cnt)
    print("预测数：", cnt, " 预测为真数：", P1, " 预测为假数：", P0)
    print("误识率：", FP / cnt)
    print("拒真率：", FN / cnt)


if __name__ == '__main__':
    tg_infer = TGInfer(os.path.join(args.model_path, "AGModel.opt.tnnproto"), os.path.join(args.model_path, "AGModel.opt.tnnmodel"))
    infer()
