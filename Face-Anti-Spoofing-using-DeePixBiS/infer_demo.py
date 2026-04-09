import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import sys
from colorama import Fore, Style

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MiniFASNetV2(nn.Module):
    def __init__(self):
        super(MiniFASNetV2, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

anti_spoof = MiniFASNetV2().to(device)
state = torch.load("models/anti_spoof.pth", map_location=device)
anti_spoof.load_state_dict(state, strict=False)
anti_spoof.eval()

mtcnn = MTCNN(keep_all=False, device=device, image_size=256, margin=20)

if len(sys.argv) < 2:
    print("Usage: python infer_demo.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: could not open video:", video_path)
    sys.exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video opened: {frame_count} frames, {fps:.2f} fps")

target_fps = 3
step = max(1, int(fps / target_fps))

probabilities = []

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % step != 0:
        frame_idx += 1
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)

    if face is not None:
        face_np = face.squeeze(0).cpu().numpy()           # (3,256,256)
        face_np = np.transpose(face_np, (1,2,0))          # (256,256,3)  <-- correct HWC
        face_np = cv2.resize(face_np, (80, 80))           # (80,80,3)
        face_np = face_np.astype(np.float32) / 255.0
        face_np = np.transpose(face_np, (2,0,1))          # back to CHW (3,80,80)
        face_tensor = torch.from_numpy(face_np).unsqueeze(0).to(device)


        with torch.no_grad():
            out = anti_spoof(face_tensor)
            prob = F.softmax(out, dim=1)[0][0].item()  
        probabilities.append(prob)

    frame_idx += 1

cap.release()

if len(probabilities) == 0:
    print("No face frames detected → cannot score.")
    sys.exit(0)

score = np.median(probabilities)
print(f"Fake probability score: {score:.2f}")
if score > 0.50:
    print(Fore.RED + "Likely: FAKE" + Style.RESET_ALL)
else:
    print(Fore.GREEN + "Likely: REAL" + Style.RESET_ALL)