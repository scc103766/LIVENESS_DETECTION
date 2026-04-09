import os
import cv2
import torch
import numpy as np
import faiss
import logging

# ==========================================
# 0. 导入本地模型类 (请根据实际情况修改)
# ==========================================
from Face_detection_yolo_align import YOLOv7_face_mkl 
from backbones import get_model

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class LocalFaceComparator:
    def __init__(self, yolo_path, arcface_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = self._load_detection_model(yolo_path)
        self.recognizer = self._load_recognition_model(arcface_path, architecture='r100')
        self.embedding_dim = 512
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
    def _load_detection_model(self, model_path):
        model = YOLOv7_face_mkl(model_path=model_path, device=self.device)
        return model

    def _load_recognition_model(self, model_path, architecture='r100'):
        model = get_model(architecture, fp16=False)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval().to(self.device)
        return model

    def process_image_details(self, img_bgr, img_name="temp"):
        """返回提取的特征、人脸框坐标和 YOLO 对齐后的人脸图。"""
        detect_results = self.detector.infer_batch([img_bgr], [img_name]) 
        if not detect_results or len(detect_results[0]['bbox']) == 0: 
            return None, None, None
            
        bbox = detect_results[0]['bbox'][0]
        x1, y1, x2, y2 = map(int, bbox)
        
        aligned_face_bgr = detect_results[0]['aligned_faces'][0]
        if not aligned_face_bgr.any(): 
            return None, None, None
        
        face_rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
        face_tensor = np.transpose(face_rgb, (2, 0, 1)).astype(np.float32)
        face_tensor = (face_tensor / 127.5) - 1.0 
        face_tensor = torch.from_numpy(face_tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.recognizer(face_tensor).cpu().numpy()
            
        faiss.normalize_L2(embedding)
        return embedding, (x1, y1, x2, y2), aligned_face_bgr

    def process_and_get_embedding(self, img_bgr, img_name="temp"):
        """兼容旧逻辑，只返回特征和人脸框坐标。"""
        embedding, bbox, _aligned_face_bgr = self.process_image_details(img_bgr, img_name)
        return embedding, bbox

    def compare_with_faiss(self, emb_A, emb_B):
        self.faiss_index.reset() 
        self.faiss_index.add(emb_A) 
        D, I = self.faiss_index.search(emb_B, 1) 
        return D[0][0] 


# ==========================================
# 核心验证逻辑：将测试图与底库进行一票否决比对
# ==========================================
def evaluate_against_gallery(engine, test_emb, gallery_embs, threshold):
    """
    测试特征与底库所有特征比对。
    返回: (是否为真人[bool], 库内最低分, 库内最高分)
    """
    scores = []
    for gal_emb in gallery_embs:
        score = engine.compare_with_faiss(test_emb, gal_emb)
        scores.append(score)
        
    min_score = min(scores)
    max_score = max(scores)
    
    # 核心活体规则：必须大于设定阈值 (0.49)
    is_real = min_score > threshold
    return is_real, min_score, max_score


# ==========================================
# 交互式主程序
# ==========================================
if __name__ == '__main__':
    # ---------------- 1. 基础配置 ----------------
    YOLO_PATH = '/supercloud/llm-code/scc/scc/Liveness_Detection/yolov7_face/yolov7-w6-face.pt'
    ARCFACE_PATH = '/supercloud/llm-code/scc/scc/Liveness_Detection/model_16.pt'
    
    # 【请确保这个文件夹里放了 3~5 张用户的真实免冠照片】
    GALLERY_FOLDER = './data/gallery_real_faces'  
    
    THRESHOLD = 0.49 # 一票否决的生死线
    
    print("\n" + "="*50)
    print(" 启动 FR-Based 交互式活体检测防伪系统")
    print("="*50)
    
    engine = LocalFaceComparator(YOLO_PATH, ARCFACE_PATH)

    # ---------------- 2. 加载底库 ----------------
    print(f"\n>>> 正在初始化安全底库 (Gallery)...")
    gallery_embs = []
    
    if not os.path.exists(GALLERY_FOLDER):
        os.makedirs(GALLERY_FOLDER)
        print(f" 找不到底库文件夹！已自动创建 {GALLERY_FOLDER}。请放入真人照片后重启程序。")
        exit()
        
    gallery_images = [f for f in os.listdir(GALLERY_FOLDER) if f.lower().endswith(('.jpg', '.png'))]
    if not gallery_images:
        print(f" 底库为空！请在 {GALLERY_FOLDER} 中放入绝对可信的真人照片。")
        exit()

    for img_name in gallery_images:
        path = os.path.join(GALLERY_FOLDER, img_name)
        img = cv2.imread(path)
        emb, box = engine.process_and_get_embedding(img, img_name)
        if emb is not None:
            gallery_embs.append(emb)
            
    print(f"成功加载 {len(gallery_embs)} 张底库真人面部特征！防御阵列已建立 (阈值: {THRESHOLD})。")

    # ---------------- 3. 交互式菜单循环 ----------------
    while True:
        print("\n" + "-"*40)
        print("请选择测试图像的来源模式：")
        print(" [1] 打开本地摄像头实时测试")
        print(" [2] 手动输入本地图片路径测试")
        print(" [3] 退出系统")
        choice = input("请输入序号 (1/2/3): ").strip()
        
        if choice == '3':
            print(" 退出系统，再见！")
            break
            
        elif choice == '1':
            # --- 模式 1: 摄像头实时检测 ---
            print("\n>>> 正在启动摄像头... (在视频窗口按 'q' 退出实时检测)")
            cap = cv2.VideoCapture(0) # 0为默认摄像头
            if not cap.isOpened():
                print(" 无法调用摄像头！(如果您在无 GUI 的远程服务器上运行，此模式将失效)")
                continue
                
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # 处理每一帧
                emb, box = engine.process_and_get_embedding(frame, "cam_frame")
                
                if emb is not None and box is not None:
                    is_real, min_score, max_score = evaluate_against_gallery(engine, emb, gallery_embs, THRESHOLD)
                    x1, y1, x2, y2 = box
                    
                    # 渲染 UI
                    if is_real:
                        color = (0, 255, 0) # 绿
                        text = f"REAL | Min:{min_score:.2f} > {THRESHOLD}"
                    else:
                        color = (0, 0, 255) # 红
                        text = f"SPOOF (Fake) | Min:{min_score:.2f} < {THRESHOLD}"
                        
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, text, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(frame, "No Face Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                    
                cv2.imshow("Live Face Anti-Spoofing", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
            print(" 摄像头已关闭。")

        elif choice == '2':
            # --- 模式 2: 上传指定路径 ---
            img_path = input("\n 请粘贴测试图片的绝对或相对路径: ").strip()
            # 移除用户复制路径时可能带上的双引号
            img_path = img_path.strip('"').strip("'") 
            
            if not os.path.exists(img_path):
                print(f" 找不到文件: {img_path}")
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                print(" 图片解码失败，请确认文件格式。")
                continue
                
            print(f" 正在提取 {os.path.basename(img_path)} 的人脸特征并进行底库验证...")
            emb, box = engine.process_and_get_embedding(img, "upload_img")
            
            if emb is None:
                print(" 检测失败：该图片中没有清晰的人脸！")
                continue
                
            is_real, min_score, max_score = evaluate_against_gallery(engine, emb, gallery_embs, THRESHOLD)
            
            # 控制台输出结果
            print("\n" + "="*40)
            print(f"报告分析:")
            print(f" 库内最高相似度: {max_score:.4f}")
            print(f" 库内最低相似度: {min_score:.4f}")
            
            if is_real:
                print(f"最终判决: 【真人 (Real)】 -> 最低得分跨过了 {THRESHOLD} 门槛！")
                color = (0, 255, 0)
                status_txt = f"REAL | Score:{min_score:.4f}"
            else:
                print(f"最终判决: 【假人攻击 (Fake)】 -> 突破防线失败！")
                color = (0, 0, 255)
                status_txt = f"FAKE | Score:{min_score:.4f}"
            print("="*40)
            
            # 画框并保存留档
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
            cv2.putText(img, status_txt, (x1, max(30, y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            save_path = "./result_uploaded.jpg"
            cv2.imwrite(save_path, img)
            print(f" 结果留档截图已保存至: {save_path}\n")

        else:
            print(" 输入无效，请输入 1, 2 或 3。")
