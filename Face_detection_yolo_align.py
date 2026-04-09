import os
import cv2
import numpy as np
import torch
import time
from yolov7_face.models.experimental import attempt_load
from pathlib import Path
from yolov7_face.utils.plots import plot_one_box
from yolov7_face.utils.datasets import letterbox
from yolov7_face.utils.general import check_img_size, non_max_suppression, scale_coords

class YOLOv7_face_mkl:
    def __init__(self, model_path='yolov7-w6-face.pt', img_size=640, device="cuda:0", log=None, cam_id=None):
        self.model_path = model_path
        self.img_size = img_size
        self.device = torch.device(device)
        self.cam_id = cam_id
        self.log = log
        # 禁用日志记录器以避免不必要的输出，如果需要可以传递一个真正的logger
        if self.log is None:
            import logging
            self.log = logging.getLogger("dummy")
            self.log.disabled = True

        self.model, self.stride, self.img_size = self.initialize_model()
        self.model.to(self.device).eval()
        self.all_img = 0
        # Warm-up
        print(f"[{self.__class__.__name__}] Warming up model on {self.device}...")
        self.infer_batch([np.zeros((640, 640, 3), dtype=np.uint8)], ['warmup'])
        print(f"[{self.__class__.__name__}] Model warmed up successfully.")

    def initialize_model(self):
        """初始化设备和模型"""
        model = attempt_load(self.model_path, map_location=self.device, log=self.log)
        stride = int(model.stride.max())
        img_size = check_img_size(self.img_size, s=stride)
        return model, stride, img_size

    # def align_faces(self, original_image, bboxes, all_keypoints, target_size=(112, 112)):
    #     """
    #     根据检测到的边界框和关键点，高效地对人脸进行仿射变换对齐
    #     """
    #     aligned_face_list = []
    #     # 标准的112x112参考关键点
    #     ref_landmarks = np.array([
    #         [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    #         [41.5493, 92.3655], [70.7299, 92.2041]
    #     ], dtype=np.float32)

    #     for bbox, keypoints in zip(bboxes, all_keypoints):
    #         if not keypoints or len(keypoints) != 5:
    #             aligned_face_list.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
    #             continue
            
    #         src_landmarks = np.array(keypoints, dtype=np.float32)

    #         try:
    #             # 使用estimateAffine2D获得更稳健的变换矩阵
    #             M, _ = cv2.estimateAffine2D(src_landmarks, ref_landmarks)
    #             if M is None:
    #                 aligned_face = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    #             else:
    #                 aligned_face = cv2.warpAffine(
    #                     original_image, M, target_size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    #                 )
    #         except cv2.error:
    #             aligned_face = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            
    #         aligned_face_list.append(aligned_face)
            
    #     return aligned_face_list
    
    def align_faces( self, original_image, bboxes, all_keypoints,  target_size=(112, 112), expand_percentage= 10):
        """
        根据检测到的边界框和关键点，对人脸进行高效对齐。
        采用“先扩展裁剪，后在小图上进行仿射变换”的策略。

        Args:
            original_image (np.ndarray): 原始的、未经缩放的图像。
            bboxes (List[List[float]]): 边界框列表 [[x1, y1, x2, y2], ...]。
            all_keypoints (List[List[Tuple[float, float]]]): 关键点列表 [[(x,y), ...], ...]。
            target_size (Tuple[int, int]): 对齐后的人脸目标尺寸。
            expand_percentage (int): 裁剪时向外扩展的百分比。

        Returns:
            List[np.ndarray]: 一个包含所有对齐后人脸图像（BGR, uint8）的列表。
        """
        aligned_face_list = []
        
        for bbox, keypoints in zip(bboxes, all_keypoints):
            # 检查关键点是否有效
            if not keypoints or len(keypoints) != 5 or any(kp is None for kp in keypoints):
                # 如果关键点无效，则返回一个黑色图像作为占位符
                aligned_face_list.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            w, h = x2 - x1, y2 - y1

            # 1. 扩展裁剪
            margin = expand_percentage / 100.0
            crop_x1 = max(0, int(x1 - w * margin))
            crop_y1 = max(0, int(y1 - h * margin))
            crop_x2 = min(original_image.shape[1], int(x2 + w * margin))
            crop_y2 = min(original_image.shape[0], int(y2 + h * margin))
            
            face_region = original_image[crop_y1:crop_y2, crop_x1:crop_x2]

            if face_region.size == 0:
                aligned_face_list.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
                continue

            # 2. 坐标转换
            local_keypoints = [(kp[0] - crop_x1, kp[1] - crop_y1) for kp in keypoints]

            # 3. 仿射变换对齐
            # 标准的112x112参考关键点
            ref_landmarks = np.array([
                [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                [41.5493, 92.3655], [70.7299, 92.2041]
            ], dtype=np.float32)
            
            scale = target_size[0] / 112.0
            ref_landmarks *= scale
            
            src_landmarks = np.array(local_keypoints, dtype=np.float32)

            try:
                M, _ = cv2.estimateAffinePartial2D(src_landmarks, ref_landmarks)
                if M is None:
                    aligned_face = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                else:
                    aligned_face = cv2.warpAffine(
                        face_region, M, target_size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
                    )
            except cv2.error:
                aligned_face = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            
            aligned_face_list.append(aligned_face)
            
        return aligned_face_list
    
    

    def infer_batch(self, image_list: list, image_name_list: list[str], conf_threshold=0.7, iou_threshold=0.5):
        """
        对一批图片进行推理，并返回包含对齐人脸的结果。
        """
        imgs = []
        original_images = []
        image_shapes = []

        # 预处理所有图像
        for img in image_list:
            self.all_img += 1
            original_images.append(img.copy())
            image_shapes.append(img.shape)
            img_resized = letterbox(img, self.img_size, stride=self.stride, auto=False)[0]
            img_resized = img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img_resized = np.ascontiguousarray(img_resized)
            imgs.append(torch.from_numpy(img_resized).to(self.device).float() / 255.0)

        if not imgs:
            return []

        batch_tensor = torch.stack(imgs, 0)   #得到 （n,c,h,w）的张量
        
        # 推理
        with torch.no_grad():
            pred = self.model(batch_tensor, augment=False)[0]

        # NMS
        pred = non_max_suppression(pred, conf_threshold, iou_threshold, kpt_label=5)
        
        results_batch = []
        for img_idx, det in enumerate(pred):
            im0 = original_images[img_idx]
            img_shape = image_shapes[img_idx]
            
            bbox_list, confidence_list, class_list, keypoints_list = [], [], [], []

            if len(det):
                # 将坐标从模型输入尺寸映射回原始图像尺寸
                scale_coords(batch_tensor.shape[2:], det[:, :4], img_shape, kpt_label=False)
                scale_coords(batch_tensor.shape[2:], det[:, 6:], img_shape, kpt_label=5, step=3)

                # <--- 改进: 直接迭代以获取所有信息 --->
                for row in det:
                    # 使用切片来提取数据，而不是多星号解包
                    xyxy = row[:4]      # 前4个元素是边界框
                    conf = row[4]       # 第5个元素是置信度
                    cls = row[5]        # 第6个元素是类别
                    kpts_flat = row[6:] # 从第7个元素开始都是关键点数据

                    bbox_list.append([coord.item() for coord in xyxy])
                    confidence_list.append(conf.item())
                    class_list.append(int(cls.item()))
                    
                    # 处理关键点
                    keypoints = []
                    # kpts_flat 是平铺的 [x1, y1, conf1, x2, y2, conf2, ...]
                    # 我们每3个元素取前2个（x, y）
                    for i in range(5):  # 5个关键点
                        kpt_x = kpts_flat[i * 3].item()
                        kpt_y = kpts_flat[i * 3 + 1].item()
                        keypoints.append((kpt_x, kpt_y))
                    keypoints_list.append(keypoints)
                    
                # <--- 调用对齐函数并获取其结果 --->
                aligned_faces = self.align_faces(im0, bbox_list, keypoints_list)

                results_batch.append({
                    "image_time": image_name_list[img_idx],
                    "image": im0,
                    "bbox": bbox_list,
                    "confidence": confidence_list,
                    "class": class_list,
                    "keypoints": keypoints_list,
                    "aligned_faces": aligned_faces  # <--- 将对齐后的人脸添加到结果中 --->
                })
            else:
                # 即使没有检测到人脸，也返回一个标准的空结果结构
                results_batch.append({
                    "image_time": image_name_list[img_idx],
                    "image": im0,
                    "bbox": [], "confidence": [], "class": [], "keypoints": [], "aligned_faces": []
                })

        return results_batch
    

def test_and_visualize_results(model_instance, input_folder, output_folder, batch_size):
    """
    (关键修改) 使用分批处理的方式，测试文件夹中的所有图像。

    Args:
        model_instance (YOLOv7_face_mkl): 初始化好的模型实例。
        input_folder (str): 包含测试图像的文件夹路径。
        output_folder (str): 保存结果的根文件夹路径。
        batch_size (int): 每个推理批次处理的图片数量。
    """
    print(f"\n开始测试与可视化...")
    print(f"  - 输入图片文件夹: {input_folder}")
    print(f"  - 输出结果文件夹: {output_folder}")
    print(f"  - 推理批处理大小 (Batch Size): {batch_size}")

    output_landmarks_path = os.path.join(output_folder, "landmarks_on_original")
    output_aligned_path = os.path.join(output_folder, "aligned_faces")
    os.makedirs(output_landmarks_path, exist_ok=True)
    os.makedirs(output_aligned_path, exist_ok=True)

    # 1. 先收集所有图片路径
    image_paths = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_paths:
        print(f"错误: 在文件夹 '{input_folder}' 中未找到任何图片。")
        return

    total_images = len(image_paths)
    total_faces_detected = 0
    print(f"共找到 {total_images} 张图片，开始分批处理...")

    # (关键修改) 禁用梯度计算，进一步节省显存
    with torch.no_grad():
        # 2. 循环处理每个小批次
        for i in range(0, total_images, batch_size):
            # 创建当前批次的切片
            batch_paths = image_paths[i:i + batch_size]
            print(f"  处理批次 {i // batch_size + 1} / { -(-total_images // batch_size) }: 图片 {i+1}-{min(i + batch_size, total_images)}")

            # 读取当前批次的图片
            image_batch = [cv2.imread(p) for p in batch_paths if cv2.imread(p) is not None]
            name_batch = [os.path.basename(p) for p in batch_paths if cv2.imread(p) is not None]

            if not image_batch:
                continue

            # 3. 对当前小批次进行推理
            results = model_instance.infer_batch(image_batch, name_batch)

            # 4. 处理并保存当前批次的结果
            for result_dict in results:
                original_image = result_dict['image']
                original_name = Path(result_dict['image_time']).stem
                
                num_faces = len(result_dict['bbox'])
                if num_faces == 0: continue

                total_faces_detected += num_faces
                landmarks_image = original_image.copy()

                for j in range(num_faces):
                    bbox, conf, kpts, aligned_face = result_dict['bbox'][j], result_dict['confidence'][j], result_dict['keypoints'][j], result_dict['aligned_faces'][j]
                    
                    label = f'Face {j} ({conf:.2f})'
                    plot_one_box(bbox, landmarks_image, label=label, color=(0, 255, 0), line_thickness=2)
                    for kpt in kpts:
                        cv2.circle(landmarks_image, (int(kpt[0]), int(kpt[1])), 3, (0, 0, 255), -1, cv2.LINE_AA)
                    
                    if aligned_face.any():
                        save_path_aligned = os.path.join(output_aligned_path, f"{original_name}_face_{j}.jpg")
                        cv2.imwrite(save_path_aligned, aligned_face)

                save_path_landmarks = os.path.join(output_landmarks_path, f"{original_name}_landmarks.jpg")
                cv2.imwrite(save_path_landmarks, landmarks_image)

    print(f"\n测试完成！共检测到 {total_faces_detected} 张人脸。")
    print(f"  - 带关键点的原图已保存至: {output_landmarks_path}")
    print(f"  - 对齐后的人脸小图已保存至: {output_aligned_path}")

if __name__ == '__main__':
    # 1. 设置路径
    # --- 请根据您的环境修改以下路径 ---
    MODEL_PATH = '/supercloud/llm-code/scc/scc/insightface/recognition/arcface_torch/xxt_run_dataset/yolov7_face/yolov7-w6-face.pt'  # 您的YOLOv7-face模型权重路径
    INPUT_IMAGE_FOLDER = '/supercloud/llm-code/scc/scc/insightface/recognition/arcface_torch/xxt_run_dataset/DATA/Target_6.12_2/big'   # 存放待测试图片的文件夹
    OUTPUT_FOLDER = './cut_result' # 存放结果的文件夹
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 16

    # 准备示例图片 (如果文件夹不存在)
    if not os.path.exists(INPUT_IMAGE_FOLDER):
        print(f"示例图片文件夹 '{INPUT_IMAGE_FOLDER}' 不存在，正在创建并下载示例图片...")
        os.makedirs(INPUT_IMAGE_FOLDER)
        try:
            # 你可以替换成任何你想测试的图片URL
            import urllib.request
            urllib.request.urlretrieve("https://raw.githubusercontent.com/WongKinYiu/yolov7/main/inference/images/horses.jpg", f"{INPUT_IMAGE_FOLDER}/test1.jpg")
            urllib.request.urlretrieve("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg", f"{INPUT_IMAGE_FOLDER}/test2.jpg")
            print("示例图片下载完成。")
        except Exception as e:
            print(f"下载示例图片失败，请手动放置一些图片到 '{INPUT_IMAGE_FOLDER}' 文件夹中。错误: {e}")
            exit()


    # 2. 初始化模型
    try:
        face_detector = YOLOv7_face_mkl(model_path=MODEL_PATH, device=DEVICE)
    except Exception as e:
        print(f"初始化模型失败，请检查模型路径 '{MODEL_PATH}' 是否正确。错误: {e}")
        exit()


    # 3. 运行验证和可视化函数
    test_and_visualize_results(
        model_instance=face_detector,
        input_folder=INPUT_IMAGE_FOLDER,
        output_folder=OUTPUT_FOLDER,
        batch_size=BATCH_SIZE
    )