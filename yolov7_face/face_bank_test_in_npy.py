import os
import cv2
import numpy as np
import faiss
import torch
import sys
sys.path.append("/supercloud/llm-code/scc/scc/insightface/recognition/arcface_torch") # 175的识别模型
from datetime import datetime
from backbones import get_model
# from detect_yolov8.yolo import YoloClient
### 输入  512 特征，输出匹配情况，这里没有检测模块

# 提取特征
@torch.no_grad()
def extract_feature(model, imgs):
    processed_imgs = []
    for img  in imgs:
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        img.div_(255).sub_(0.5).div_(0.5)
        processed_imgs.append(img)
    batch = torch.stack(processed_imgs)  # (batch_size, C, H, W)
    features = model(batch).numpy()
    normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
    print("__batch")
    return normalized_features



# 搜索特征库
def search_face_in_index(feature, index, filenames):
    feature = feature.reshape(1, -1).astype("float32")
    distances, indices = index.search(feature, 1)
    best_match_idx = indices[0][0]
    best_match_file = filenames[best_match_idx]
    best_distance = distances[0][0]
    return best_match_file, best_distance


# 比较函数
def compare_faces_inf(in_feature_512, index, filenames, log_file, threshold=0.5):
    # 在索引库中搜索
    best_match_file, best_distance = search_face_in_index(in_feature_512, index, filenames)
    print('best_match_file', best_match_file)
    print('best_distance', best_distance)


def infer_once(infile="/supercloud/llm-code/mkl/project/facer/yolov7-face/2400-0_1small.jpeg"):
    MODEL_WEIGHT = "/supercloud/llm-code/scc/scc/insightface/recognition/arcface_torch/arcface_r100_t9_glint/model_r100_PFC.pt"
    MODEL_NAME = "r100"
    # 加载人脸识别模型
    model = get_model(MODEL_NAME, fp16=False)
    model.load_state_dict(torch.load(MODEL_WEIGHT))
    model.eval()
    img = cv2.imread(infile)
    img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
    # b = np.array(img)
    # print(b.shape)
    print("img.shape ", img.shape)
    imgs = [img, img, img, img, img, img, img, img]
    feature = extract_feature(model, imgs)[0]
    return feature

if __name__ == "__main__":
    # 加载模型、检测器和索引库
    # MODEL_WEIGHT = "/supercloud/llm-code/scc/scc/insightface/recognition/arcface_torch/arcface_r100_t9_glint/model_r100_PFC.pt"
    # MODEL_NAME = "r100"
    INDEX_FILE = "face_index_r101__0.4.faiss"
    MAPPING_FILE = "file_mapping_r101_0.4.npy"
    IMAGE_DIR = "/supercloud/llm-code/scc/scc/face_pic2"
    LOG_FILE = "face_comparison_log.log"
    threshold=0.5

    index = faiss.read_index(INDEX_FILE)
    filenames = np.load(MAPPING_FILE, allow_pickle=True)


    in_feature_512 = np.random.rand(512)
    a_np_file = "./runs/in_feature_512.npy"
    np.save(a_np_file, in_feature_512)
    in_feature_512 = np.load(a_np_file)
    # 推理一张图片
    fea = infer_once()
    # 开始比对
    res = compare_faces_inf(fea, index, filenames, LOG_FILE, threshold)
    print(f"res  {res}")
    print("_end success")
