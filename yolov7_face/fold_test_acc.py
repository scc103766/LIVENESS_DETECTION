# 根据文件夹，统计检测到的比例
import os
import cv2
import json
import shutil
import time
import torch
from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

def get_image_paths(input_folder):
    """使用 os.walk 获取所有图片路径"""
    supported_formats = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng", ".webp", ".mpo")
    image_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_paths.append(os.path.join(root, file))
    return image_paths

def initialize_model(model_path, device, img_size):
    """初始化设备和模型"""
    model = attempt_load(model_path, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    return model, stride, img_size

def infer_image(model, device, img, img_size, conf_threshold, iou_threshold):
    """对单张图片进行推理"""
    # 确保 stride 是普通数值而不是张量
    stride = int(model.stride.max().cpu().item())  
    img = letterbox(img, img_size, stride=stride)[0]  # 调整图像尺寸
    img = img[:, :, ::-1].copy().transpose(2, 0, 1)  # HWC to CHW, BGR to RGB，修复负步长问题
    img = torch.from_numpy(img).to(device).float() / 255.0  # 归一化
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    t1 = time_synchronized()
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_threshold, iou_threshold)
    t2 = time_synchronized()
    return pred, t2 - t1


def visualize_and_save(image_path, im0, detections, output_folder):
    """可视化检测结果并保存"""
    file_name = Path(image_path).name
    output_path = os.path.join(output_folder, file_name)

    for *xyxy, conf, cls in detections:
        label = f"{conf:.2f}"
        plot_one_box(xyxy, im0, label=label, color=(255, 0, 0), line_thickness=2)

    cv2.imwrite(output_path, im0)

def process_images(input_folder, output_folder, model_path, conf_threshold=0.45, iou_threshold=0.5, img_size=640):
    # 初始化设备和模型
    device = select_device('')
    model, stride, img_size = initialize_model(model_path, device, img_size)

    # 创建输出文件夹
    no_face_folder = os.path.join(output_folder, "noface")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(no_face_folder, exist_ok=True)

    # 初始化统计变量
    detection_results = {}
    idx, noface_idx = 0, 0
    start_time = time.time()

    # 获取所有图片路径
    image_paths = get_image_paths(input_folder)

    for image_path in image_paths:
        file_name = Path(image_path).name
        im0 = cv2.imread(image_path)  # 原始图像

        # 推理
        pred, inference_time = infer_image(model, device, im0, img_size, conf_threshold, iou_threshold)

        # 处理检测结果
        detection_results[file_name] = []
        if len(pred[0]):  # 检测到目标
            idx += 1
            det = pred[0]
            det[:, :4] = scale_coords(im0.shape[:2], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                detection_results[file_name].append({
                    "box": [int(x) for x in xyxy],
                    "score": float(conf),
                    "class": int(cls)
                })

            visualize_and_save(image_path, im0, det, output_folder)
        else:  # 未检测到目标
            noface_idx += 1
            shutil.copy(image_path, os.path.join(no_face_folder, file_name))

    # 打印统计信息
    total_time = time.time() - start_time
    print(f"Total images processed: {idx + noface_idx}")
    print(f"Images with faces: {idx}")
    print(f"Images without faces: {noface_idx}")
    print(f"Time taken: {total_time:.3f}s")
    print(f"Average time per image: {total_time / (idx + noface_idx):.3f}s")

    # 保存检测结果到JSON文件
    json_output_path = os.path.join(output_folder, "detection_results.json")
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(detection_results, json_file, indent=4, ensure_ascii=False)

    print(f"处理完成，结果已保存到: {output_folder}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default="/supercloud/llm-code/mkl/dataset/face/t9demo/demo", type=str, help="输入图片文件夹路径")
    parser.add_argument('--output_folder', default="./runs/detect/fold_detect", type=str, help="输出结果文件夹路径")
    parser.add_argument('--model_path', default="yolov7-w6-face.pt", type=str, help="模型文件路径")
    parser.add_argument('--conf_threshold', type=float, default=0.45, help="置信度阈值")
    parser.add_argument('--iou_threshold', type=float, default=0.5, help="NMS IOU阈值")
    parser.add_argument('--img_size', type=int, default=640, help="输入图片大小")
    args = parser.parse_args()

    process_images(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model_path=args.model_path,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        img_size=args.img_size
    )
    print("Done.")
