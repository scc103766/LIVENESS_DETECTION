import numpy as np
import onnxruntime
import os
import cv2


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

# input----------------------------------------------------------------------


pre_path = "./resources"
threshold = 0.93


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
    txt_file = os.path.join(pre_path, file)
    sc = np.array(read_colors(txt_file))
    img = cv2.imread(txt_file.replace(".txt", ".jpg"), cv2.IMREAD_UNCHANGED)
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


def check_color(color_list, color_pred):
    color_pred = np.reshape(color_pred, -1)
    for i in range(3):
        c = color_list[i*3:(i+1)*3]
        p = color_pred[i*3:(i+1)*3]
        curColorLabel = [
            (1.0 if c[0] > c[1] else 0.0) if abs(c[0] - c[1]) > 128 else -1.0,
            (1.0 if c[1] > c[2] else 0.0) if abs(c[1] - c[2]) > 128 else -1.0,
            (1.0 if c[2] > c[0] else 0.0) if abs(c[2] - c[0]) > 128 else -1.0,
        ]
        for j in range(3):
            if curColorLabel[j] < -0.1:
                continue
            if curColorLabel[j] < 0.5 < p[j]:
                return False
            if curColorLabel[j] > 0.5 > p[j]:
                return False
    return True


worker = ONNXModel("./resources/AGModel.onnx")  # load model

for file in os.listdir(pre_path):
    if not file.endswith(".txt"):
        continue
    normal_cues, sc_flag = load_image(file)
    sc, p = worker.forward({"input": (normal_cues - 127.5) / 128.0})
    pred = p[0]

    # 验证颜色验证是否通过
    sc = np.reshape(sc, [-1, 3])
    sc_flag = np.reshape(sc_flag, [-1, 3])
    flag = 1  # 验证通过就是1
    for sp, l in zip(sc, sc_flag):
        for i in range(3):
            if np.min(l * (sp - 0.5)) < 0:
                flag = 0
                break
        if flag == 0:
            break

    print(file, "打分：%.4f" % pred, "是真人" if pred > threshold else "是假人", "颜色验证通过" if flag == 1 else "颜色验证不通过")








