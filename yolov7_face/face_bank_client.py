import requests
import base64
import cv2
import time
# 配合的服务是 /supercloud/llm-code/mkl/project/facer/yolov7-face/face_bank_service2.py
# 输入112*112小图，返回匹配到了谁以及置信读； 可以按 batch 输入
# 图像Base64
image_path = "/supercloud/llm-code/mkl/project/facer/yolov7-face/2400-0_1small.jpeg"
image = cv2.imread(image_path)
_, buffer = cv2.imencode('.jpg', image)
image_base64 = base64.b64encode(buffer).decode("utf-8")  # 【是大图里裁剪成小图的部分，不是直接大图】
N = 2
start = time.time()
for _ in range(N):
    url = "http://192.168.17.175:15100/face_recognition"
    payload = {"images": [image_base64]*10}  # 支持多个图像， 默认batch=8， 最好可配置
    # payload = {"images": [image_base64]}
    response = requests.post(url, json=payload)
# 查看返回结果
if response.status_code == 200:
    print(response.json())
    pass
else:
    print("Error:", response.status_code, response.text)



print(f"time {time.time() - start}")
# 检索单进程  60 iter， 8 batch， 8.0s
# 检索多进程4  60 iter， 8 batch， 8.3s
# 检索多进程8  60 iter， 16 batch， 16.8 s
# 检索多进程1  60 iter， 16 batch， 14.9 s
#############GPU
# 识别+检索多进程1  60 iter， 32 batch， 4.3 s 60*32张图