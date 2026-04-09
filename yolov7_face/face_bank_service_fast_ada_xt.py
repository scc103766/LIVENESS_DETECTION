from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import time
import cv2
import numpy as np
import faiss
import torch
import base64
import uvicorn
import random
import sys
from concurrent.futures import ProcessPoolExecutor
import logging
from PIL import Image
import io

from data_infer_mask_2cls import mask_classifier
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=6)

sys.path.append("/supercloud/llm-code/mkl/project/lab/AdaFace")
import net

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

import argparse

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
parser.add_argument("--port", type=int, default=15119, help="Port to run the Flask app")
args = parser.parse_args()

# 获取设备和端口
device = args.device
port = args.port
GPU_NUM = int(device.split(":")[1])
print(f"INDEXGPU NUM  {GPU_NUM}")

# 初始化 FastAPI 应用
app = FastAPI()

# 全局进程池，避免 CPU 线程竞争
process_pool_executor = ProcessPoolExecutor(4)


# 在 FastAPI 关闭时正确释放进程池
@app.on_event("shutdown")
def shutdown_event():
    process_pool_executor.shutdown()


# 加载人脸识别模型和索引库
MODEL_WEIGHT = "/supercloud/llm-code/mkl/project/lab/AdaFace-master/experiments/ir101_glint_t9_1_7a_01-07_4/epoch=24-step=162525.ckpt"
INDEX_FILE_FACE = "./bank/face_index_ada_align_450_t9_50all_01_16.faiss"
MAPPING_FILE_FACE = "./bank/file_mapping_ada_align_450_t9_50all_01_16.npy"
INDEX_FILE_MASK = "./bank/face_index_ada_align_450_t9_50all_01_16_mask.faiss"
MAPPING_FILE_MASK = "./bank/file_mapping_ada_align_450_t9_50all_01_16_mask.npy"


# 初始化模型
def load_pretrained_model():
    model = net.build_model("ir_101")
    statedict = torch.load(MODEL_WEIGHT, map_location=device)['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model.to(device)


model = load_pretrained_model()
model.eval()

# 初始化索引库
res = faiss.StandardGpuResources()
index_face = faiss.read_index(INDEX_FILE_FACE)
index_mask = faiss.read_index(INDEX_FILE_MASK)
index_face = faiss.index_cpu_to_gpu(res, GPU_NUM, index_face)
index_mask = faiss.index_cpu_to_gpu(res, GPU_NUM, index_mask)
filenames_face = np.load(MAPPING_FILE_FACE, allow_pickle=True)
filenames_mask = np.load(MAPPING_FILE_MASK, allow_pickle=True)


@app.get("/")
def read_root():
    return {"message": "Face Bank Service Running"}


@torch.no_grad()
def extract_feature(model, imgs):
    processed_imgs = []
    for img in imgs:
        if img is None or img.empty():
            print("图像读取失败")
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32, device=device)  # 直接在 GPU 上创建
        img.div_(255).sub_(0.5).div_(0.5)
        processed_imgs.append(img)
    batch = torch.stack(processed_imgs)  # (batch_size, C, H, W)
    feature, _ = model(batch)
    features = feature.cpu().numpy()
    return features / np.linalg.norm(features, axis=1, keepdims=True)

async def extract_feature_async(model, imgs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, extract_feature, model, imgs)

def decode_image(img_base64):
    if not isinstance(img_base64, str):
        logger.error("Invalid base64-encoded string: Not a string type")
        return None

    # 修正 base64 长度，使其成为 4 的倍数
    missing_padding = len(img_base64) % 4
    if missing_padding:
        img_base64 += '=' * (4 - missing_padding)

    try:
        img_data = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        return np.array(img)
    except Exception as e:
        logger.error(f"Base64 decoding failed: {e}")
        return None
    if not isinstance(img_base64, str) or len(img_base64) % 4 != 0:
        logger.error("Invalid base64-encoded string: incorrect padding or format")
        return None
    img_data = base64.b64decode(img_base64)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    return np.array(img)

async def decode_image_async(img_base64):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, decode_image, img_base64)


def search_face_in_index(feature, index, filenames):
    feature = feature.reshape(1, -1).astype("float32")
    distances, indices = index.search(feature, 1)
    return filenames[indices[0][0]], distances[0][0]

async def search_face_in_index_async(feature, index, filenames):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, search_face_in_index, feature, index, filenames)

# async def predict_async(self, images):
#     # 假设 mask_classifier 是一个类，可以将其方法改为异步
#     loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(executor, self.predict, images)

@app.post("/face_recognition")
async def face_recognition(request: Request):
    beg = time.time()
    try:
        body = await request.json()
        image_list = body.get("images", [])
        if not image_list:
            return JSONResponse(content={"error": "No images provided"}, status_code=400)

        # 异步解码图像
        img_time = time.time()
        decoded_images = []
        tasks = [decode_image_async(img_base64) for img_base64 in image_list]
        decoded_images = [img for img in await asyncio.gather(*tasks) if img is not None]
        logger.info(f"decoded_images time: {time.time() - img_time}, num: {len(decoded_images)}")

        # 异步进行口罩分类
        mask_time = time.time()
        mask_list_res = []
        tasks = [mask_classifier.predict([img]) for img in decoded_images]
        mask_list_res = [res[0] if isinstance(res, list) else res for res in await asyncio.gather(*tasks)]
        logger.info(f"mask_list_res time: {time.time() - mask_time}, num: {len(mask_list_res)}")

        # 异步提取特征
        features = await extract_feature_async(model, decoded_images)

        # 异步搜索索引
        index_time = time.time()
        results = []
        tasks = [search_face_in_index_async(feature, index_mask if mask < 0.5 else index_face,
                                            filenames_mask if mask < 0.5 else filenames_face) for feature, mask in
                 zip(features, mask_list_res)]
        results = await asyncio.gather(*tasks)
        logger.info(f"search_face_in_index time: {time.time() - index_time}, num: {len(results)}")

        response = {
            "results": [
                {
                    "match": match,
                    "distance": float(dist),
                    "nomask": float(mask),
                    "rec_time": time.time() - beg
                } for (match, dist), mask in zip(results, mask_list_res)
            ]
        }
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    host = "0.0.0.0"
    print(f"Starting service on device {device} at {host}:{port}")
    uvicorn.run(app, host=host, port=port)
