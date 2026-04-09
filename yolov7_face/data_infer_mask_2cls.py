from PIL import Image
import torch
import requests
import base64
import cv2
import time
import numpy as np
import sys
sys.path.append("/supercloud/llm-code/mkl/project/facer/pytorch-image-models") # 分类 timm
import timm
# 口罩识别模型，输入路径或者图片的list， 输出无口罩的概率
class ImageMaskClassifier:
    def __init__(self, model_name, checkpoint_path, num_classes=2, device=None):
        """
        图像分类器类，用于模型初始化和推理。

        Args:
            model_name (str): 模型名称。
            checkpoint_path (str): 检查点路径。
            num_classes (int): 分类数量，默认是二分类。
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.device = device
        # 初始化模型
        self.model, self.transforms = self._initialize_model()

    def _initialize_model(self):
        """内部方法：加载模型并初始化。"""
        model = timm.create_model(self.model_name, pretrained=False)
        checkpoint = torch.load(self.checkpoint_path)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        if self.device:
            model = model.to(self.device)
            print(f"mask cls model to device P{self.device}")
        # 设置模型为评估模式
        model.eval()

        # 获取模型特定的变换（归一化和尺寸调整）
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        return model, transforms

    def predict(self, inputs):
        """
        对输入图像进行推理，支持批量预测。
        Args:
            inputs (list): 图像路径列表。  或者 cv2 的读取数据
        Returns:
            List[float]: 每张图像的 0 类别概率。
        """
        try:
            input_tensors = []
            # 加载并预处理所有图像
            for item in inputs:
                if isinstance(item, str) and (item.endswith('.jpg') or item.endswith('.png') or item.endswith('.jpeg')): # 输入为路径
                    img = Image.open(item)
                    input_tensor = self.transforms(img).unsqueeze(0)  # 添加 batch 维度
                else:  #if isinstance(item, str):  # 输入为 Base64 编码的图像数据
                    # print(f"item type {type(item)}")
                    img_array = cv2.imdecode(np.frombuffer(base64.b64decode(item), np.uint8), cv2.IMREAD_COLOR)
                    if img_array is None or img_array.empty():
                        print("mask图像读取失败")
                    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                    input_tensor = self.transforms(img).unsqueeze(0)  # 添加 batch 维度
            
                input_tensors.append(input_tensor)

            # 将所有图像堆叠成一个批次,由外部输入控制batch大小
            batch_tensor = torch.cat(input_tensors, dim=0).to(self.device)
            output = self.model(batch_tensor)
            # 转为概率并获取 0 类别的概率  0 无口罩， 1 有口罩  只看0概率即可，0.5阈值
            probabilities = torch.softmax(output, dim=1)
            zero_class_probabilities = probabilities[:, 0].detach().cpu().numpy()
        except Exception as err:
            print(f"err  Pmask 2cls error {err}")
            return [0] * len(inputs)
        return zero_class_probabilities.tolist()


MODEL_MASK_NAME = 'resnet18d.ra4_e3600_r224_in1k'
MASK_CHECKPOINT_PATH = '/supercloud/llm-code/mkl/project/facer/pytorch-image-models/outputs/res180106/20250106-204702-resnet18d_ra4_e3600_r224_in1k-224/checkpoint-15.pth.tar'
# 创建分类器实例
device = "cuda:2"
mask_classifier = ImageMaskClassifier(MODEL_MASK_NAME, MASK_CHECKPOINT_PATH, device=device)

if __name__ == "__main__":

    mask_classifier = ImageMaskClassifier(MODEL_MASK_NAME, MASK_CHECKPOINT_PATH, device=device)
    # 口罩分类  推理代码，  多张一起输入的batch
    # 图像路径列表
    img_paths = [
        '/supercloud/llm-code/mkl/project/facer/pytorch-image-models/models/resnet18d/2-0_0_cropped.jpg',
        '/supercloud/llm-code/mkl/project/facer/pytorch-image-models/models/resnet18d/2-0_1_cropped.jpg'
    ]
    # 进行推理
    predictions = mask_classifier.predict(img_paths)
    # 打印结果
    for idx, zero_class_probability in enumerate(predictions):
        print(f"输入路径，Image {idx + 1}: Probability of Class 0无口罩 = {zero_class_probability:.2f}")

    #输入方式2 ， cv2读完再放到list中推理
    imglist = []
    for aimage_path in img_paths:
        image = cv2.imread(aimage_path)
        _, buffer = cv2.imencode('.jpg', image)  # 这块要小心  jpg 压缩，压缩的太多了，可以设置参数不压缩，本来应该是原图的裁剪，没有压缩的
        image_base64 = base64.b64encode(buffer).decode("utf-8")  # 【是大图里裁剪成小图的部分，不是直接大图】
        imglist.append(image_base64)
    print(f"输入图片cv2读取 图片batch {len(imglist)}")
    predictions2 = mask_classifier.predict(imglist)
    # 打印结果
    for idx, zero_class_probability in enumerate(predictions2):
        print(f"输入cv2读取，Image {idx + 1}: Probability of Class 0无口罩 = {zero_class_probability:.2f}")
    