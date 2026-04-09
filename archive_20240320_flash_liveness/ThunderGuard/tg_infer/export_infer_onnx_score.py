import argparse
import cv2
import numpy as np
import os
import onnxruntime
from tqdm import tqdm
import matplotlib.pyplot as plt
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="../resources", help='模型位置')
parser.add_argument('--from_path', type=str, default=r"E:\tg_video\dataset\dnormal\dfake", help='原始图片路径')
parser.add_argument('--to_path', type=str, default=r"D:\BaiduNetdiskDownload\fake", help='输出文件路径')
args = parser.parse_args()


class ONNXModel(object):
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor[name]
        return input_feed

    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return output


def show_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def read_colors(file):
    line_list = []
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            color = int(line)
            c1 = ((color & 0x00ff0000) >> 16)
            c2 = ((color & 0x0000ff00) >> 8)
            c3 = (color & 0x0000ff)
            line_list.append((c1, c2, c3))
            line = f.readline()
    return line_list


def load_image(file):
    txt_file = os.path.join(args.from_path, file)
    sc = np.array(read_colors(txt_file))
    img = cv2.imread(txt_file.replace(".txt", ".jpg"), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(txt_file)
    assert img.shape[2] == 3
    normal_cue_list = []
    for i in range(3):
        for j in range(2):
            nid = (i * 2 + j)
            cur_img = img[nid * 288 + 16:(nid + 1) * 288 - 16, 16:288-16, :].transpose((2, 0, 1)).astype(np.float32)
            cur_img = cur_img.reshape(1, 3, 256, 256)
            normal_cue_list.append(cur_img)
    normal_cues = np.concatenate(normal_cue_list, axis=0)

    # 每张图有3个分类值，表示当前通道的闪光差是否大于下一个
    sc_flag = []
    for i in range(3):
        sc_flag.append(np.where(sc[i] > sc[i, [1, 2, 0]], 1, 0) + np.where(sc[i] < sc[i, [1, 2, 0]], -1, 0))
        sc_flag.append(sc_flag[-1])
    return normal_cues,  np.array(sc_flag, dtype=np.int32)


def copyto(file, score):
    from_file = os.path.join(args.from_path, file).replace(".txt", ".jpg")
    pre_text = str(int(score*1000))
    while len(pre_text) < 4:
        pre_text = "0" + pre_text
    to_file = os.path.join(args.to_path, "%s_%s" % (pre_text, file)).replace(".txt", ".jpg")
    copyfile(from_file, to_file)


def infer():
    # 先排除以及生成好的
    exclude_file = []
    for file in os.listdir(args.to_path):
        if file.endswith(".jpg"):
            exclude_file.append("_".join(file.split('_')[1:]))
    exclude_file = set(exclude_file)

    p_list, l_list = [], []
    flag_list = []
    file_list = []
    for file in tqdm(os.listdir(args.from_path)):
        if file.endswith(".txt") and file.replace(".txt", ".jpg") not in exclude_file:
            if file.endswith("_1.txt"):
                label = 1
            else:
                label = 0
        else:
            continue
        normal_cues, sc_flag = load_image(file)
        sc, p = worker.forward({"input": (normal_cues - 127.5) / 128.0})
        pred = p[0]

        # 验证颜色验证是否通过
        sc = np.reshape(sc, [-1, 2, 3])
        sc_flag = np.reshape(sc_flag, [-1, 2, 3])
        flag = 1  # 验证通过就是1
        for sc_pair, flag_pair in zip(sc, sc_flag):
            error_time = 0
            for i in range(2):
                sp, l = sc_pair[i], flag_pair[i]
                for _ in range(3):
                    if np.min(l * (sp - 0.5)) < 0:
                        error_time += 1
                        break
            if error_time == 2:
                flag = 0
                break

        if flag == 0:
            copyto(file, 0)
        else:
            copyto(file, pred)
        file_list.append(file)
        p_list.append(pred)
        l_list.append(label)
        flag_list.append(flag)

    p_list = np.array(p_list)
    l_list = np.array(l_list)
    flag_list = np.array(flag_list)
    id_list = np.arange(len(flag_list))
    print("正样本合格率：", len(flag_list[(flag_list > 0.1) & (l_list > 0.1)]) / (len(flag_list[l_list > 0.1]) + 1e-5))
    print("负样本合格率：", len(flag_list[(flag_list > 0.1) & (l_list < 0.1)]) / (len(flag_list[l_list < 0.1]) + 1e-5))

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
    threshold = 0.8
    for (id, score, label) in zip(id_list, p_list, l_list):
        if score > threshold:
            P1 += 1
        else:
            P0 += 1
        if score > threshold and label < 1:
            print(file_list[id], score)
            FP += 1
        if score <= threshold and label > 0:
            print(file_list[id], score)
            FN += 1
        cnt += 1
    print("预测数：", cnt, " 预测为真数：", P1, " 预测为假数：", P0)
    print("误识率：", FP / np.sum(1 - l_list))
    print("拒真率：", FN / np.sum(l_list))


if __name__ == '__main__':
    onnx_path = os.path.join(args.model_path, "TGModel.onnx")
    worker = ONNXModel(onnx_path)
    infer()

