## 可以可视化  矩形框和 关键点了
# 裁剪  贴图112  未对齐
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
from typing import List
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
    """对单张图片进行推理
    save_detect_name  是保存的 txt 关键点box 文件名
    """
    global device, img_size
    img = cv2.imread(one_image_path)
    stride = int(model.stride.max().cpu().item())  
    img_resized, ratio, pad = letterbox(img, img_size, stride=stride)  # 调整图像尺寸
    img_resized = img_resized[:, :, ::-1].copy().transpose(2, 0, 1)  # HWC to CHW, BGR to RGB
    img_resized = torch.from_numpy(img_resized).to(device).float() / 255.0  # 归一化
    if img_resized.ndimension() == 3:
        img_resized = img_resized.unsqueeze(0)

    t1 = time_synchronized()
    #model(torch.concat([img_resized,img_resized],0), augment=False)
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
    mysave = "/supercloud/llm-code/scc/scc/BCTC_code/yolov7_face/runs/detect/single_detect/detect_face_no_face.jpg"
    cv2.imwrite(mysave, im0)
    print("____save success")
    return results, t2 - t1


def infer_batch(model, image_list: List,image_name_list: List[str], save_detect_dir=None,conf_threshold=0.45, iou_threshold=0.5):
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
    global device, img_size
    imgs = []
    original_images = []
    image_shapes = []

    # 创建保存结果的目录
    if save_detect_dir:
        Path(save_detect_dir).mkdir(parents=True, exist_ok=True)

    # 预处理所有图像
    for img in image_list:
        #img = cv2.imread(img_path)
        # if img is None:
        #     print(f"Warning: Unable to read image {img_path}. Skipping.")
        #     continue
        original_images.append(img.copy())
        image_shapes.append(img.shape)
        stride = int(model.stride.max().cpu().item())
        img_resized, ratio, pad = letterbox(img, img_size, stride=stride)  # 调整图像尺寸
        img_resized = img_resized[:, :, ::-1].copy().transpose(2, 0, 1)  # HWC to CHW, BGR to RGB
        img_resized = torch.from_numpy(img_resized).to(device).float() / 255.0  # 归一化
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
    t1 = time_synchronized()
    with torch.no_grad():
        pred = model(batch_tensor, augment=False)[0]
    pred = non_max_suppression(pred, conf_threshold, iou_threshold, kpt_label=5)
    t2 = time_synchronized()

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
            for det_index, (*xyxy, conf, cls) in enumerate(det[:, :6]):
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
    return results_batch, t2 - t1


def process_image(image_path, output_folder, idx, beg):
    try:
        one_image = os.path.basename(image_path)
        one_fold = os.path.basename(os.path.dirname(image_path))
        one_output_folder = os.path.join(output_folder, one_fold)
        
        # if not os.path.exists(one_output_folder):
        #     os.makedirs(one_output_folder)
        os.makedirs(one_output_folder, exist_ok=True)

        img_basename, ext = os.path.splitext(one_image)
        save_detect_name = os.path.join(one_output_folder, img_basename)
        
        # 进行推理处理
        results, inference_time = infer_image(model, image_path, save_detect_name)
        
        if idx % 100 == 1:
            print(f'processed {idx} images.  one {(time.time()-beg)/(idx+0.0001)}    time {time.time()-beg}')
    except Exception as err:
        print("______error")
        print(err)


def process_fold_image(image_fold, model_path, conf_threshold=0.45, iou_threshold=0.5, img_size=640, output_folder='./output'):
    # 初始化设备和模型
    beg = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        idx = 0
        for one_fold in os.listdir(image_fold):
            for one_image in os.listdir(os.path.join(image_fold, one_fold)):
                one_image_path = os.path.join(image_fold, one_fold, one_image)
                futures.append(executor.submit(process_image, one_image_path, output_folder, idx, beg))
                idx += 1
                if idx % 100 == 1:
                    print(f'processed {idx} images.  one {(time.time()-beg)/(idx+0.0001)}    time {time.time()-beg}')
        
        # 等待所有线程完成
        concurrent.futures.wait(futures)
    
    print('done')
    print(f'processed {idx} images.  one {(time.time()-beg)/(idx+0.0001)}    time {time.time()-beg}')

def get_image_path(afull_name):
    # print(afull_name)
    # 构造可能的路径
    png_path = afull_name.replace('.json', '.png')
    jpg_path = afull_name.replace('.json', '.jpg')
    jpeg_path = afull_name.replace('.json', '.jpeg')
    # 检查文件是否存在
    if os.path.exists(png_path):
        return png_path
    elif os.path.exists(jpg_path):
        return jpg_path
    elif os.path.exists(jpeg_path):
        return jpeg_path
    else:
        print(f"error {afull_name}")
        return None


def fun_duiqi(img, keypoints, size=112):
    """
    从图片中裁剪人脸，并根据五个关键点对齐人脸。

    参数：
        img (numpy.ndarray): 原始图片。
        keypoints (list of tuples): 五个关键点的坐标 [(x1, y1), ..., (x5, y5)]。
        size (int): 输出图片的大小 (size x size)，默认112。

    返回：
        numpy.ndarray: 对齐后的人脸图片。
    """
    # 定义标准化五点参考坐标，用于仿射变换（基于112x112的图像尺寸）
    ref_landmarks = np.array([
        [38.2946, 51.6963],  # 左眼
        [73.5318, 51.5014],  # 右眼
        [56.0252, 71.7366],  # 鼻尖
        [41.5493, 92.3655],  # 左嘴角
        [70.7299, 92.2041]   # 右嘴角
    ], dtype=np.float32)
    # 缩放参考坐标以匹配目标尺寸
    ref_landmarks *= size / 112.0
    # 将输入关键点转换为 numpy 数组
    src_landmarks = np.array(keypoints, dtype=np.float32)
    # 计算仿射变换矩阵
    transform_matrix = cv2.estimateAffinePartial2D(src_landmarks, ref_landmarks)[0]
    # 对原始图片进行仿射变换，输出对齐图片
    aligned_img = cv2.warpAffine(img, transform_matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return aligned_img

def cut_from_fold(image_fold, txt_folder, fold_112, target_size=(112, 112), duiqi=False):
    os.makedirs(fold_112, exist_ok=True)

    for one_person_fold in os.listdir(txt_folder):
        txt_folder_one = os.path.join(txt_folder, one_person_fold)
        for atxt in os.listdir(txt_folder_one):
            atxt_path = os.path.join(txt_folder_one, atxt) # txt 路径名字
            fold_name = os.path.basename(txt_folder_one) # 每个人文件夹名字 
            # fold_name Roberto_Canessa 
            # txt_folder_one /supercloud/llm-code/mkl/dataset/face/lfw_demo_txt/Roberto_Canessa
            # atxt_path /supercloud/llm-code/mkl/dataset/face/lfw_demo_txt/Roberto_Canessa/Roberto_Canessa_0001.json
            imgpath = get_image_path(os.path.join(image_fold, fold_name, atxt))
            if imgpath is None:
                print(f"error---- a image is None  imgpath {imgpath}   atxt_path {atxt_path}  fold_name {fold_name}")
                continue
            # print("imgpath", imgpath)


            # 读取图片
            image = cv2.imread(imgpath)
            # 从txt文件中读取矩形框坐标

            max_area = 0
            max_bbox = None
            with open(atxt_path, "r") as f:
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

                        # 绘制边界框  裁剪时去掉
                        # label = f'{cls} {confidence:.2f}'
                        # plot_one_box(bbox, image, label=label, color=(255, 0, 0), line_thickness=2)

                        # 计算面积并找出最大面积的人脸
                        x1, y1, x2, y2 = map(int, bbox)
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            max_bbox = bbox
                            max_kpt = keypoints

                        # # 绘制关键点
                        # for kpt_x, kpt_y in keypoints:
                        #     cv2.circle(image, (int(kpt_x), int(kpt_y)), radius=3, color=(0, 255, 0), thickness=-1)
                    except Exception as eliff:
                        print(f"第 {line_num} 行处理失败: {eliff}")

            # 裁剪出最大矩形并保存
            if max_area > 1:
                x1, y1, x2, y2 = map(int, max_bbox)
                cropped_image = image[y1:y2, x1:x2]

                # 计算裁剪图片的长宽比
                cropped_h, cropped_w = cropped_image.shape[:2]
                aspect_ratio = cropped_w / cropped_h

                # 确定缩放后的尺寸
                if aspect_ratio > 1:  # 宽大于高
                    new_w = target_size[0]
                    new_h = int(target_size[0] / aspect_ratio)
                else:  # 高大于宽
                    new_h = target_size[1]
                    new_w = int(target_size[1] * aspect_ratio)

                # 缩放图片   对齐模块放这？
                resized_image = cv2.resize(cropped_image, (new_w, new_h))
                if duiqi:
                    aligned_im = fun_duiqi(resized_image, max_kpt)
                    resized_image = aligned_im

                # 创建一个黑色背景图像
                padded_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

                # 将缩放后的图片放置在中心
                x_offset = (target_size[0] - new_w) // 2
                y_offset = (target_size[1] - new_h) // 2
                padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

                # 确保目标文件夹存在
                save_folder = os.path.join(fold_112, fold_name)
                os.makedirs(save_folder, exist_ok=True)

                # 保存裁剪后的图像
                save_path = os.path.join(save_folder, f"{os.path.splitext(atxt)[0]}_cropped.jpg")
                cv2.imwrite(save_path, padded_image)
                # print(f"裁剪后的图像已保存: {save_path}")
            else:
                print(f"未找到有效的矩形框，跳过文件: {atxt_path}")


if __name__ == "__main__":
    # image_path = '/media/face2/data/haiguan'  # 输入图像路径
    
    #image_fold = "/supercloud/llm-code/mkl/dataset/face/lfw_demo"  # 240 大图 原图
    
    # output_folder = '/media/face2/data/data_2024/haiguan_det_txt'  
    #output_folder = '/supercloud/llm-code/mkl/dataset/face/lfw_demo_txt'  # 输出 json 检测结果文件夹路径
    # im0_with_detections = process_fold_image(image_fold, model_path, output_folder=output_folder) # 检测 + 保存结果到txt
    #print(f"save detect result to {output_folder}")

    # 从txt文件读取信息并绘制到图片上 验证对不对
    # draw_from_file(image_path, "/supercloud/llm-code/mkl/project/facer/yolov7-face/runs/detect/single_detect/results.txt", "/supercloud/llm-code/mkl/project/facer/yolov7-face/runs/detect/single_detect/resave2.jpg")
    #fold_112 = '/supercloud/llm-code/mkl/dataset/face/lfw_demo_112img'
    # fold_112 = '/media/face2/data/data_2024/haiguan_det_112cut'
    #cut_from_fold(image_fold, output_folder, fold_112, duiqi=False)
    #print("___end")
    pass
