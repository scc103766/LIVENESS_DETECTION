import os
import cv2
import numpy as np
import torch
import faiss
import sys
from yolov7_face.models.experimental import attempt_load
from pathlib import Path
#from infer_one_img_batch_cut import  model as detect_model  # 检测模型v7
from typing import List
from yolov7_face.utils.plots import plot_one_box
from utils.datasets import letterbox
from utils.torch_utils import select_device, time_synchronized
from utils.general import check_img_size, non_max_suppression, scale_coords,xyxy2xywh
# from detect_yolov8.yolo import YoloClient
# save_detect_fold = "/supercloud/llm-code/mkl/dataset/face/public/tmp/detect_land_tmp" # 检测结果保存暂存路径

# # 配置模型和文件路径
# MODEL_WEIGHT = "/supercloud/llm-code/scc/scc/insightface/recognition/arcface_torch/arcface_r100_t9_glint/model.pt" # rec model
# ROOT_FOLDER = "/supercloud/llm-code/scc/scc/face_pic3"
# INDEX_FILE = "face_index_r101__0.4.faiss"
# MAPPING_FILE = "file_mapping_r101_0.4.npy"

# 加载识别模型

print(f"rec model load success")


##detect model
class YOLOv7_face_mkl:
    def __init__(self, model_path='yolov7-w6-face.pt',img_size = 640,device = "cuda:4"):
        self.model_path = model_path
        self.img_size = img_size
        self.device = device
        self.model,_,_ = self.initialize_model()

    def initialize_model(self):
        """初始化设备和模型"""
        model = attempt_load(self.model_path, map_location= self.device)
        stride = int(model.stride.max())
        self.img_size = check_img_size(self.img_size, s=stride)
        return model, stride, self.img_size



    def infer_batch(self, image_list: List, image_name_list: List[str], save_detect_dir=None, conf_threshold=0.45,
                    iou_threshold=0.5):
        """
        对一批图片进行推理

        参数:
            model: 预训练的YOLO模型
            image_paths (List[str]): 图像文件路径列表
            save_detect_dir (str): 保存检测结果的目录
            conf_threshold (float): 置信度阈值
            iou_threshold (float): IOU阈值
        返回:
            List[dict]: 每张图片的检测结果
            float: 推理时间
        """
        imgs = []
        original_images = []
        image_shapes = []

        # 创建保存结果的目录
        if save_detect_dir:
            Path(save_detect_dir).mkdir(parents=True, exist_ok=True)

        # 预处理所有图像
        for img in image_list:
            # img = cv2.imread(img_path)
            # if img is None:
            #     print(f"Warning: Unable to read image {img_path}. Skipping.")
            #     continue
            original_images.append(img.copy())
            image_shapes.append(img.shape)
            stride = int(self.model.stride.max().cpu().item())
            img_resized, ratio, pad = letterbox(img, self.img_size, stride=stride)  # 调整图像尺寸
            img_resized = img_resized[:, :, ::-1].copy().transpose(2, 0, 1)  # HWC to CHW, BGR to RGB
            img_resized = torch.from_numpy(img_resized).to(self.device).float() / 255.0  # 归一化
            if img_resized.ndimension() == 3:
                imgs.append(img_resized.unsqueeze(0))
            else:
                imgs.append(img_resized)

        if not imgs:
            print("No valid images to process.")
            return [], 0.0

        # 创建批量张量
        batch_tensor = torch.cat(imgs, dim=0)

        # 执行推理
        #t1 = time_synchronized()
        with torch.no_grad():
            pred = self.model(batch_tensor, augment=False)[0]
        pred = non_max_suppression(pred, conf_threshold, iou_threshold, kpt_label=5)
        #t2 = time_synchronized()

        results_batch = []

        # 处理每张图像的检测结果
        for img_idx, det in enumerate(pred):
            im0 = original_images[img_idx].copy()
            img_shape = image_shapes[img_idx]
            if len(det):
                # 调整检测框的坐标
                scale_coords(batch_tensor.shape[2:], det[:, :4], img_shape, kpt_label=False)
                scale_coords(batch_tensor.shape[2:], det[:, 6:], img_shape, kpt_label=5, step=3)

                image_results = []
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    # 提取矩形框
                    if save_detect_dir:
                        label = f'{int(cls)} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=(255, 0, 0), line_thickness=2)

                    # 检测关键点数据
                    keypoint_data = []
                    for kpt_idx in range(5):  # 假设关键点数量为5
                        kpt_start = 6 + kpt_idx * 3
                        kpt_end = kpt_start + 2
                        if kpt_end <= det.shape[1]:  # 检查索引是否超出范围
                            kpt_x, kpt_y = det[det_index, kpt_start:kpt_end].cpu().numpy()
                            if kpt_x > 0 and kpt_y > 0:  # 过滤无效关键点
                                keypoint_data.append((float(kpt_x), float(kpt_y)))
                                if save_detect_dir:
                                    cv2.circle(im0, (int(kpt_x), int(kpt_y)), radius=3, color=(0, 255, 0), thickness=-1)

                    detection = {
                        "bbox": [float(coord.cpu().numpy()) for coord in xyxy],
                        "confidence": float(conf),
                        "class": int(cls),
                        "keypoints": keypoint_data
                    }
                    image_results.append(detection)

                results_batch.append({
                    "image_time": image_name_list[img_idx],
                    "detections": image_results
                })

                # 保存检测后的图片（可选）
                if save_detect_dir:
                    save_img_path = Path(save_detect_dir) / f"{image_name_list[img_idx]}_detected.jpg"
                    cv2.imwrite(str(save_img_path), im0)
            else:
                print(f"未检测到人脸: {image_name_list[img_idx]}")
                results_batch.append({
                    "image_time": image_name_list[img_idx],
                    "detections": []
                })
        return results_batch
