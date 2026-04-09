from flask import Flask, request, jsonify
import os, time
import cv2
import numpy as np
import faiss
import torch
import sys
import base64
sys.path.append("/supercloud/llm-code/scc/scc/insightface/recognition/arcface_torch") # 175的识别模型
from backbones import get_model
from concurrent.futures import ThreadPoolExecutor
# 创建线程池
executor = ThreadPoolExecutor(max_workers=1)

# 初始化 Flask 应用
app = Flask(__name__)

# 加载人脸识别模型和索引库
MODEL_WEIGHT = "/supercloud/llm-code/scc/scc/insightface/recognition/arcface_torch/arcface_r100_t9_glint/model_r100_PFC.pt"
MODEL_NAME = "r100"
INDEX_FILE = "face_index_r101__0.4.faiss"
MAPPING_FILE = "file_mapping_r101_0.4.npy"
THRESHOLD = 0.5
device = "cuda:3"
# 初始化模型、索引库
model = get_model(MODEL_NAME, fp16=False)
model.load_state_dict(torch.load(MODEL_WEIGHT, map_location=device))
model = model.to(device)
model.eval()


res = faiss.StandardGpuResources() # gpu1
print("Reading index...")
index = faiss.read_index(INDEX_FILE)
print("Index read successfully")
print(f"Moving index to GPU: {device}")
index = faiss.index_cpu_to_gpu(res, int(device.split(":")[1]), index)
print("Index moved to GPU successfully")

filenames = np.load(MAPPING_FILE, allow_pickle=True)

print("index gpu _1")
# index = faiss.index_cpu_to_gpu(res, int(device.split(":")[1]), index)
print("index gpu _2")
# 提取特征
@torch.no_grad()
def extract_feature(model, imgs):
    processed_imgs = []
    for img in imgs:
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float().to(device)
        img.div_(255).sub_(0.5).div_(0.5)
        processed_imgs.append(img)
    batch = torch.stack(processed_imgs)  # (batch_size, C, H, W)
    features = model(batch).cpu().numpy()
    normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return normalized_features

# 搜索特征库
def search_face_in_index(feature, index, filenames):
    feature = feature.reshape(1, -1).astype("float32")
    distances, indices = index.search(feature, 1)
    best_match_idx = indices[0][0]
    best_match_file = filenames[best_match_idx]
    best_distance = distances[0][0]
    return best_match_file, best_distance

# API: 接收图像列表，返回检索结果
@app.route('/face_recognition', methods=['POST'])
def face_recognition():
    beg = time.time()
    try:
        # 从请求中获取图片列表
        image_list = request.json.get("images", [])
        if not image_list:
            return jsonify({"error": "No images provided"}), 400

        # 解码 Base64 编码的图像
        decoded_images = []
        for img_base64 in image_list:
            img_data = np.frombuffer(base64.b64decode(img_base64), dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            decoded_images.append(img)

        # 提取特征
        features = extract_feature(model, decoded_images)
        rec_time_dur = time.time()
        print(f"rec_time  :{time.time()-beg:.2f}")
        # 并行检索每张图片的结果
        def process_feature(feature):
            best_match_file, best_distance = search_face_in_index(feature, index, filenames)
            return {
                "match": best_match_file,
                "distance": float(best_distance)
            }
            
        # 检索每张图片的结果
        # results = []
        results = list(executor.map(process_feature, features))
        rec_time_dur = time.time()
        print(f"idx time :{time.time()-rec_time_dur:.2f}")
        return jsonify({"results": results})
        # for feature in features:
        #     best_match_file, best_distance = search_face_in_index(feature, index, filenames)
            
        #     results.append({
        #         "match": best_match_file, 
        #         "distance": float(best_distance)  # 转换为标准 float 类型
        #     })
        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=15100)
