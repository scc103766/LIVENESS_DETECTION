import os
import cv2
import numpy as np
import torch
import faiss
import sys
from infer_one_img_batch_cut import  img_size,infer_image
from infer_one_img_batch_cut import  model as detect_model  # 检测模型v7
sys.path.append("/supercloud/llm-code/scc/scc/insightface/recognition/arcface_torch") # 175的识别模型
from backbones import get_model # rec model
# from detect_yolov8.yolo import YoloClient
save_detect_fold = "/supercloud/llm-code/mkl/dataset/face/public/tmp/detect_land_tmp" # 检测结果保存暂存路径

# 提取人脸特征
@torch.no_grad()
def extract_feature(model, img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    feature = model(img).numpy().flatten()
    return feature / np.linalg.norm(feature)


# 检测与对齐人脸函数
def detect_and_align(img_path, detector):
# 只crop  还没对齐
    img = cv2.imread(img_path)
    
    os.makedirs(save_detect_fold, exist_ok=True)
    save_detect_name = os.path.join(save_detect_fold, os.path.basename(os.path.splitext(img_path)[0]) + '.json') # save_detect_fold下同名json
    # faces = detector.detect_faces(img)
    faces,  durtime = infer_image(detector, img_path , save_detect_name)
    if not faces:
        raise ValueError(f"未检测到人脸: {img_path}")

    face = faces[0] #要升级为 挑选最大人脸
    face = face["bbox"]
    x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])

    cropped_face = img[y1:y2, x1:x2]
    # cv2.imshow('crop', cropped_face)
    cv2.imwrite("./test.jpg", cropped_face)
    return cropped_face


# 创建特征库
def build_face_index(image_folder, model, detector, index_file, mapping_file, dim=512):
    index = faiss.IndexFlatL2(dim)
    file_paths = []
    idx = 0
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if not file.lower().endswith(("-0_1.jpeg")):
                continue

            img_path = os.path.join(root, file)
            try:
                face = detect_and_align(img_path, detector)
                feature = extract_feature(model, face).astype("float32")
                index.add(np.expand_dims(feature, axis=0))
                file_paths.append(img_path)
                idx += 1
                if idx % 10 == 0:
                    print(f"已处理: {idx} 张   成功处理: {img_path}")

            except Exception as e:
                print(f"处理失败: {img_path} 错误: {e}")

    # if file_paths:
    #     faiss.write_index(index, index_file)
    #     np.save(mapping_file, np.array(file_paths, dtype=object))
    #     print(f"底库一共 {idx} 张！ 特征库保存成功！ 索引库: {index_file}, 映射文件: {mapping_file}")
    # else:
    #     print("未生成任何特征，检查图片和检测器配置。")


if __name__ == "__main__":
    # 配置模型和文件路径
    MODEL_WEIGHT = "/supercloud/llm-code/scc/scc/insightface/recognition/arcface_torch/arcface_r100_t9_glint/model.pt" # rec model
    ROOT_FOLDER = "/supercloud/llm-code/scc/scc/face_pic3"
    INDEX_FILE = "face_index_r101__0.4.faiss"
    MAPPING_FILE = "file_mapping_r101_0.4.npy"

    # 加载识别模型
    model = get_model("r100", fp16=False)
    model.load_state_dict(torch.load(MODEL_WEIGHT))
    model.eval()
    print(f"rec model load success")

    # # 构建人脸索引库
    build_face_index(ROOT_FOLDER, model, detect_model, INDEX_FILE, MAPPING_FILE)
    # print("__end success")
