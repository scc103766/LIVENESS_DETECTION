import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import onnxruntime
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="../resources", help='模型位置')
parser.add_argument('--network', type=str, default='MoEA', help='网络解构分类')
parser.add_argument("--infer_type", type=str, default="sdepth", help="推理类型，仅仅在生产onnx时使用，可以选择depth、score")
parser.add_argument("--normal_cues_path", type=str, default="../data/sample/test_depth", help="推理需要输入的法线信息图")
parser.add_argument('--threshold', type=float, default=0.5, help='threshold')
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


def only_load_image(file):
    img = cv2.imread(os.path.join(args.normal_cues_path, file), cv2.IMREAD_UNCHANGED)
    normal_cue_list = []
    for i in range(3):
        for j in range(2):
            nid = (i * 2 + j)
            cur_img = img[nid * 288 + 16:(nid + 1) * 288 - 16, 16:288 - 16, :].transpose((2, 0, 1)).astype(
                np.float32)
            cur_img = cur_img.reshape(1, 3, 256, 256)
            normal_cue_list.append(cur_img)
    normal_cues = np.concatenate(normal_cue_list, axis=0)
    return normal_cues


def load_image(file):
    txt_file = os.path.join(args.normal_cues_path, file)
    sc = np.array(read_colors(txt_file))
    normal_cues = only_load_image(file.replace(".txt", ".jpg"))

    # 每张图有3个分类值，表示当前通道的闪光差是否大于下一个
    sc_flag = []
    for i in range(3):
        sc_flag.append(np.where(sc[i] > sc[i, [1, 2, 0]], 1, 0) + np.where(sc[i] < sc[i, [1, 2, 0]], -1, 0))
        sc_flag.append(sc_flag[-1])
    return normal_cues,  np.array(sc_flag, dtype=np.int32)


def infer_map():
    for file in tqdm(os.listdir(args.normal_cues_path)):
        if not file.endswith(".jpg") or file.endswith("_g.jpg"):
            continue
        # 读取图片
        normal_cues = only_load_image(file)
        depth_map_array = worker.forward({"input": (normal_cues - 127.5) / 128.0})[0]
        for i in range(len(depth_map_array)):
            depth_map = depth_map_array[i]
            depth_map = depth_map * np.ones([1, 1, 3])

            if args.infer_type == "attention":
                depth_map = (depth_map - np.min(depth_map)) * 6
                depth_map = np.where(depth_map > 1, 1, depth_map)

            if len(depth_map_array) > 1:
                cv2.imwrite(os.path.join(args.normal_cues_path, file).replace(".jpg", "_%d_g.jpg" % i), depth_map * 255)
            else:
                cv2.imwrite(os.path.join(args.normal_cues_path, file).replace(".jpg", "_g.jpg"), depth_map * 255)


def infer():
    p_list, l_list = [], []
    flag_list = []
    file_list = []
    for file in tqdm(os.listdir(args.normal_cues_path)):
        if file.endswith(".txt"):
            if len(file.split('_')) == 8 and file.startswith('0_'):
                # 不考虑超纲样本
                continue
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

        file_list.append(file)
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
    if args.infer_type == "score" or args.infer_type is None or args.infer_type == "":
        onnx_path = os.path.join(args.model_path, "TGModel.onnx")
    else:
        onnx_path = os.path.join(args.model_path, "%s_%s.onnx" % (args.network, args.infer_type))

    worker = ONNXModel(onnx_path)
    if args.infer_type == "score":
        infer()
    else:
        infer_map()

