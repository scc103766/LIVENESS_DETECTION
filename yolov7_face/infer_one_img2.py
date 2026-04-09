## 可以可视化  矩形框和 关键点了

import os
import cv2
import json
import torch
from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords,xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
import numpy as np
import time
import multiprocessing
import time
import concurrent.futures
device = "cuda:4"
img_size = 640
model_path = 'yolov7-w6-face.pt'  # 模型路径

def initialize_model(model_path, device, img_size):
    """初始化设备和模型"""
    model = attempt_load(model_path, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    return model, stride, img_size

model, stride, img_size = initialize_model(model_path, device, img_size)

def infer_image(model, one_image_path, save_detect_name, conf_threshold=0.45, iou_threshold=0.5):
    """对单张图片进行推理"""
    global device, img_size
    img = cv2.imread(one_image_path)
    stride = int(model.stride.max().cpu().item())  
    img_resized, ratio, pad = letterbox(img, img_size, stride=stride)  # 调整图像尺寸
    img_resized = img_resized[:, :, ::-1].copy().transpose(2, 0, 1)  # HWC to CHW, BGR to RGB
    img_resized = torch.from_numpy(img_resized).to(device).float() / 255.0  # 归一化
    if img_resized.ndimension() == 3:
        img_resized = img_resized.unsqueeze(0)

    t1 = time_synchronized()
    pred = model(img_resized, augment=False)[0]
    pred = non_max_suppression(pred, conf_threshold, iou_threshold, kpt_label=5)
    t2 = time_synchronized()

    results = []  # 保存所有检测信息
    for i, det in enumerate(pred):
        im0 = img.copy()
        if len(det):
            # 调整检测框的坐标
            scale_coords(img_resized.shape[2:], det[:, :4], im0.shape, kpt_label=False)
            scale_coords(img_resized.shape[2:], det[:, 6:], im0.shape, kpt_label=5, step=3)
            
            for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                # 提取矩形框
                label = f'{int(cls)} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=(255, 0, 0), line_thickness=2)
                
                # 检测关键点数据
                keypoint_data = []
                for kpt_idx in range(5):  # 假设关键点数量为5
                    kpt_start = 6 + kpt_idx * 3
                    kpt_end = kpt_start + 2
                    if kpt_end <= det.shape[1]:  # 检查索引是否超出范围
                        kpt_x, kpt_y = det[det_index,kpt_start:kpt_end].cpu().numpy()
                        if kpt_x > 0 and kpt_y > 0:  # 过滤无效关键点
                            keypoint_data.append((float(kpt_x), float(kpt_y)))
                            cv2.circle(im0, (int(kpt_x), int(kpt_y)), radius=3, color=(0, 255, 0), thickness=-1)
                
                results.append({
                    "bbox": [float(one.cpu().numpy()) for one in xyxy],
                    "confidence": float(conf),
                    "class": int(cls),
                    "keypoints": keypoint_data
                })

    # 保存检测结果到txt文件
    txt_save_path = save_detect_name+".json"
    with open(txt_save_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # # 保存检测后的图片
    # mysave = "/supercloud/llm-code/mkl/project/facer/yolov7-face/runs/detect/single_detect/save.jpg"
    # cv2.imwrite(mysave, im0)
    # print("____save success")
    return results, t2 - t1


def process_fold_image(image_fold, model_path, conf_threshold=0.45, iou_threshold=0.5, img_size=640, output_folder='./output'):
    # 初始化设备和模型
    beg = time.time()
    for idx, one_fold in enumerate(os.listdir(image_fold)):
        # print(one_fold)
        for one_image in os.listdir(os.path.join(image_fold, one_fold)):
            try:
                one_image_path = os.path.join(image_fold, one_fold, one_image) # 输入图片路径
                one_output_folder = os.path.join(output_folder, one_fold)
                # print("one_image",one_image_path)
                # print("one_output_folder",one_output_folder)
                if not os.path.exists(one_output_folder):
                    os.makedirs(one_output_folder)
                img_basename, ext = os.path.splitext(one_image) # 提取不带后缀的文件名
                save_detect_name = os.path.join(one_output_folder, img_basename)
                
                # print(">>>",save_detect_name)
                results, inference_time = infer_image(model, one_image_path, save_detect_name)
                # print("__")
                if idx % 100 == 1:
                    print(f'processed {idx} images.  one {(time.time()-beg)/(idx+0.0001)}    time {time.time()-beg}')
            except Exception as err:
                print("______error")
                print(err)
    print('done')
    print(f'processed {idx} images.  one {(time.time()-beg)/(idx+0.0001)}    time {time.time()-beg}')


def draw_from_file(img_path, txt_path, save_path):
    """从txt文件读取信息并绘制到图片上"""
    # 确保图像文件存在
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图片文件不存在: {img_path}")

    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片文件: {img_path}")

    # 确保 txt 文件存在
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"TXT 文件不存在: {txt_path}")

    max_area = 0
    max_bbox = None
    # 逐行读取 JSON 数据
    with open(txt_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                result = json.loads(line)
                bbox = result.get("bbox")
                confidence = result.get("confidence", 0.0)
                cls = result.get("class", -1)
                keypoints = result.get("keypoints", [])

                # 验证 bbox 格式
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    raise ValueError(f"第 {line_num} 行 bbox 格式无效: {bbox}")

                # 验证 keypoints 格式
                for point in keypoints:
                    if not (isinstance(point, list) and len(point) == 2):
                        raise ValueError(f"第 {line_num} 行 keypoint 格式无效: {point}")

                # 绘制边界框
                label = f'{cls} {confidence:.2f}'
                plot_one_box(bbox, img, label=label, color=(255, 0, 0), line_thickness=2)

                # 计算面积并找出最大面积的人脸
                x1, y1, x2, y2 = map(int, bbox)
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    max_bbox = bbox

                # 绘制关键点
                for kpt_x, kpt_y in keypoints:
                    cv2.circle(img, (int(kpt_x), int(kpt_y)), radius=3, color=(0, 255, 0), thickness=-1)
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解码失败: {e}")
            except Exception as e:
                print(f"第 {line_num} 行处理失败: {e}")

    # 保存最终结果
    cv2.imwrite(save_path, img)





def crop_face_from_image(image, faces, target_size=(112, 112)):
    """
    从图像中裁剪出人脸区域，并按长宽比保持裁剪区域，填充黑色。
    :param image: 输入图像
    :param faces: 人脸框列表，每个框为 (x1, y1, x2, y2)
    :param target_size: 输出目标大小
    :return: 裁剪后的图片
    """
    # 选择面积最大的脸部框
    max_area = 0
    selected_face = None
    for face in faces:
        x1, y1, x2, y2 = face
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            selected_face = face
    if selected_face is None:
        return None
    # 提取最大人脸框
    x1, y1, x2, y2 = selected_face
    face = image[y1:y2, x1:x2]
    # 计算长宽比
    face_width = x2 - x1
    face_height = y2 - y1
    aspect_ratio = face_width / face_height
    # 目标大小的长宽比
    target_width, target_height = target_size
    target_aspect_ratio = target_width / target_height

    # 按照长宽比调整裁剪
    if aspect_ratio > target_aspect_ratio:
        # 宽度较大，按宽度填充
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # 高度较大，按高度填充
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    # 调整人脸图像大小
    resized_face = cv2.resize(face, (new_width, new_height))

    # 创建一个黑色背景图
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # 将调整后的人脸粘贴到中心位置
    start_x = (target_width - new_width) // 2
    start_y = (target_height - new_height) // 2
    result[start_y:start_y + new_height, start_x:start_x + new_width] = resized_face

    return result
def cut_from_fold():
    pass

if __name__ == "__main__":
    image_path = '/raid/scc/DebugCase1/IdRsult/failed_samples/no_face_detected/uaw2rlmqui5igpj5qd5yfjr7m2300vvs.jpg'  # 输入图像路径
    # image_fold = "/supercloud/llm-code/mkl/dataset/face/lfw_demo"
    
    output_folder = '/supercloud/llm-code/scc/scc/BCTC_code/yolov7_face/runs/detect/single_detect'  # 输出文件夹路径
    im0_with_detections = process_fold_image(image_path, model_path, output_folder=output_folder) # 检测 + 保存结果到txt
    print(f"save detect result to {output_folder}")

    # 从txt文件读取信息并绘制到图片上 验证对不对
    draw_from_file(image_path, "//supercloud/llm-code/scc/scc/BCTC_code/yolov7_face/runs/detect/single_detect/bctc_results.txt", "/supercloud/llm-code/mkl/project/facer/yolov7-face/runs/detect/single_detect/bctc_resave.jpg")
    #cut_from_fold()
    print("___end")
