import cv2 as cv
import torch
from torchvision import transforms
from Model import DeePixBiS
import os

# 1. 初始化模型与配置
model = DeePixBiS()

model.load_state_dict(torch.load('./DeePixBiS.pth', map_location='cpu')) 
model.eval()

tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

faceClassifier = cv.CascadeClassifier('Classifiers/haarface.xml')

# 2. 读取原始视频
video_path = '/supercloud/llm-code/scc/scc/Liveness_Detection/炫彩闪烁活体/20230828 头模 正常光/FaceCollect/1693184996070_note_1_1_toumo_6.avi'
camera = cv.VideoCapture(video_path)

if not camera.isOpened():
    print(f"错误：无法打开视频文件！请检查路径：{video_path}")
    exit()

# 3. 配置视频写入器 (VideoWriter) 准备保存处理后的视频
# 获取原视频的宽度、高度和帧率
frame_width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(camera.get(cv.CAP_PROP_FPS))
if fps == 0: fps = 25 # 如果读取不到帧率，默认给 25

output_path = 'output_liveness_result.mp4'
# mp4v 是通用的 MP4 编码格式
fourcc = cv.VideoWriter_fourcc(*'mp4v') 
out_video = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print("开始处理视频，请稍候...")
frame_count = 0

# 4. 视频逐帧处理循环
while True:
    # 增加 ret 接收状态
    ret, img = camera.read()
    
    # 【核心修复】：如果读不到帧（视频结束），立刻跳出循环，防止报错
    if not ret:
        print("\n视频处理完毕！")
        break
        
    frame_count += 1
    
    # 转灰度图给人脸检测器使用
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)

    for x, y, w, h in faces:
        faceRegion = img[y:y + h, x:x + w]
        faceRegion = cv.cvtColor(faceRegion, cv.COLOR_BGR2RGB)

        faceRegion = tfms(faceRegion)
        faceRegion = faceRegion.unsqueeze(0)

        # 模型推理
        with torch.no_grad(): # 推理时一定要加 no_grad 节省内存
            mask, binary = model.forward(faceRegion)
            res = torch.mean(mask).item()

        # 绘制结果到当前帧
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if res < 0.5:
            cv.putText(img, f'Fake: {res:.2f}', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        else:
            cv.putText(img, f'Real: {res:.2f}', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # 5. 【核心修改】：不弹窗，直接将画好框的帧写入到新视频中
    out_video.write(img)
    
    # 打印进度提示，证明程序没卡死
    print(f"\r正在处理第 {frame_count} 帧...", end="")

# 6. 释放资源并保存文件
camera.release()
out_video.release()
print(f"结果视频已成功保存到当前目录: {output_path}")
