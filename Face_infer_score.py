import os
import cv2
import torch
import numpy as np
import faiss
import logging
import itertools
import matplotlib.pyplot as plt

# ==========================================
# 0. 导入你本地的模型类
# ==========================================
from Face_detection_yolo_align import YOLOv7_face_mkl 
from backbones import get_model

# 初始化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalFaceComparator:
    def __init__(self, yolo_path, arcface_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = self._load_detection_model(yolo_path)
        self.recognizer = self._load_recognition_model(arcface_path, architecture='r100')
        self.embedding_dim = 512
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
    def _load_detection_model(self, model_path):
        try:
            model = YOLOv7_face_mkl(model_path=model_path, device=self.device)
            logger.info(f"YOLOv7人脸检测模型已在 {self.device} 上成功加载。")
            return model
        except Exception as e:
            logger.error(f"加载检测模型时出错: {e}", exc_info=True)
            raise

    def _load_recognition_model(self, model_path, architecture='r100'):
        try:
            model = get_model(architecture, fp16=False)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval().to(self.device)
            logger.info(f"ArcFace识别模型已在 {self.device} 上成功加载。")
            return model
        except Exception as e:
            logger.error(f"加载识别模型时出错: {e}", exc_info=True)
            raise

    def process_and_get_embedding(self, img_bgr, img_name="temp"):
        detect_results = self.detector.infer_batch([img_bgr], [img_name]) 
        if not detect_results: return None, None
            
        result = detect_results[0]
        if len(result['bbox']) == 0: return None, None
            
        bbox = result['bbox'][0]
        x1, y1, x2, y2 = map(int, bbox)
        aligned_face_bgr = result['aligned_faces'][0]
        
        if not aligned_face_bgr.any(): 
            return None, None
        
        face_rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
        face_tensor = np.transpose(face_rgb, (2, 0, 1)).astype(np.float32)
        face_tensor = (face_tensor / 127.5) - 1.0 
        face_tensor = torch.from_numpy(face_tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.recognizer(face_tensor)
            embedding = embedding.cpu().numpy()
            
        faiss.normalize_L2(embedding)
        return embedding, (x1, y1, x2, y2)
    
    def compare_with_faiss(self, emb_A, emb_B):
        self.faiss_index.reset() 
        self.faiss_index.add(emb_A) 
        D, I = self.faiss_index.search(emb_B, 1) 
        return D[0][0] 

    def resize_for_concat(self, img1, img2, target_height=600):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        new_w1 = int(w1 * (target_height / h1))
        new_w2 = int(w2 * (target_height / h2))
        return cv2.resize(img1, (new_w1, target_height)), cv2.resize(img2, (new_w2, target_height))


# ==========================================
# 功能模块 1: 单图比对模式
# ==========================================
def run_single_image_mode(engine, img_path_A, img_path_B, output_folder, threshold=0.45):
    logger.info(">>> 进入单图 1v1 比对模式...")
    os.makedirs(output_folder, exist_ok=True)
    
    img_A = cv2.imread(img_path_A)
    img_B = cv2.imread(img_path_B)
    
    if img_A is None or img_B is None:
        logger.error("图片读取失败，请检查路径。")
        return
        
    emb_A, box_A = engine.process_and_get_embedding(img_A, "ImgA")
    emb_B, box_B = engine.process_and_get_embedding(img_B, "ImgB")
    
    render_A, render_B = engine.resize_for_concat(img_A, img_B, target_height=500)
    concat_img = cv2.hconcat([render_A, render_B])
    info_panel = np.zeros((80, concat_img.shape[1], 3), dtype=np.uint8)
    
    if emb_A is None or emb_B is None:
        text = "ERROR: Face Detect Failed!"
        color = (0, 0, 255)
    else:
        sim_score = engine.compare_with_faiss(emb_A, emb_B)
        is_same = sim_score > threshold
        res_txt = "MATCH" if is_same else "MISMATCH"
        color = (0, 255, 0) if is_same else (0, 0, 255)
        text = f"Faiss Score: {sim_score:.4f} | {res_txt}"
        
    cv2.putText(info_panel, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    final_img = cv2.vconcat([concat_img, info_panel])
    
    save_path = os.path.join(output_folder, "single_image_compare_result.jpg")
    cv2.imwrite(save_path, final_img)
    logger.info(f"✅ 单图比对完成！结果已保存至: {save_path}")


# ==========================================
# 功能模块 2: 视频流比对模式
# ==========================================
def run_video_mode(engine, ref_img_path, video_path, output_folder, threshold=0.45):
    logger.info(">>> 进入视频流比对模式...")
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. 提取基准图特征
    ref_img = cv2.imread(ref_img_path)
    if ref_img is None:
        logger.error("基准参考图读取失败！")
        return
    ref_emb, _ = engine.process_and_get_embedding(ref_img, "Reference")
    if ref_emb is None:
        logger.error("基准参考图中未检测到人脸，无法作为对比标准！")
        return
    logger.info("基准图特征提取成功。开始处理视频...")

    # 2. 打开视频并配置写入器
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 25
    
    save_path = os.path.join(output_folder, "video_compare_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # 提取当前帧的人脸特征与框坐标
        emb, box = engine.process_and_get_embedding(frame, f"frame_{frame_count}")
        
        if emb is not None and box is not None:
            sim_score = engine.compare_with_faiss(ref_emb, emb)
            x1, y1, x2, y2 = box
            
            is_same = sim_score > threshold
            color = (0, 255, 0) if is_same else (0, 0, 255)
            text = f"{sim_score:.2f} {'Real' if is_same else 'Fake'}"
            
            # 画框和写字
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            # 确保字不会写到画面外面
            text_y = max(30, y1 - 10)
            cv2.putText(frame, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            
        out_video.write(frame)
        print(f"\r正在处理视频... 第 {frame_count} 帧", end="")
        
    cap.release()
    out_video.release()
    print() # 换行
    logger.info(f"✅ 视频处理完成！带框结果已保存至: {save_path}")


# ==========================================
# 功能模块 3: 文件夹全局穷举模式 (原逻辑)
# ==========================================
def run_folder_mode(engine, root_folder, output_folder, threshold=0.45):
    logger.info(">>> 进入文件夹全局穷举、统计分析模式...")
    os.makedirs(output_folder, exist_ok=True)
    
    # 日志输出到文件
    log_file_path = os.path.join(output_folder, 'similarity_statistics.log')
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    
    all_image_paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                all_image_paths.append(os.path.join(root, file))
                
    if len(all_image_paths) < 2:
        logger.error("图片数量不足 2 张，无法进行穷举比对！")
        return

    logger.info(">>> 正在进行全局特征提取...")
    features_dict = {}
    for path in all_image_paths:
        img_bgr = cv2.imread(path)
        if img_bgr is None: continue
        img_name = os.path.basename(path)
        emb, box = engine.process_and_get_embedding(img_bgr, img_name)
        if emb is None: continue 
        
        parent_folder = os.path.basename(os.path.dirname(path))
        identifier = f"{parent_folder}_{os.path.splitext(img_name)[0]}"
        features_dict[path] = {'identifier': identifier, 'category': parent_folder, 'emb': emb}

    all_pairs = list(itertools.combinations(features_dict.keys(), 2))
    logger.info(f">>> 开始执行 {len(all_pairs)} 次穷举比对...\n")
    
    real_vs_real_scores, fake_vs_fake_scores, real_vs_fake_scores = [], [], []
    
    count = 0
    for path_A, path_B in all_pairs:
        count += 1
        data_A = features_dict[path_A]
        data_B = features_dict[path_B]
        
        sim_score = engine.compare_with_faiss(data_A['emb'], data_B['emb'])
        
        cat_A, cat_B = data_A['category'], data_B['category']
        if cat_A == '真人' and cat_B == '真人': real_vs_real_scores.append(sim_score)
        elif cat_A == '攻击' and cat_B == '攻击': fake_vs_fake_scores.append(sim_score)
        else: real_vs_fake_scores.append(sim_score)
        
        # 保存比对图
        img_A, img_B = cv2.imread(path_A), cv2.imread(path_B)
        render_A, render_B = engine.resize_for_concat(img_A, img_B, target_height=500)
        concat_img = cv2.hconcat([render_A, render_B])
        
        info_panel = np.zeros((80, concat_img.shape[1], 3), dtype=np.uint8)
        is_same = sim_score > threshold
        res_txt = "MATCH" if is_same else "MISMATCH"
        color = (0, 255, 0) if is_same else (0, 0, 255)
        text = f"Faiss Score: {sim_score:.4f} | {res_txt}"
        
        cv2.putText(info_panel, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        final_img = cv2.vconcat([concat_img, info_panel])
        
        save_name = f"{data_A['identifier']}_VS_{data_B['identifier']}.jpg"
        cv2.imwrite(os.path.join(output_folder, save_name), final_img)

    # 统计分析
    logger.info("="*50)
    logger.info(">>> 开始生成统计分析报告...")
    best_threshold, best_accuracy = 0.45, 0.0
    for th in np.arange(0.0, 1.0, 0.01):
        tp = sum(1 for s in real_vs_real_scores if s > th)
        tn = sum(1 for s in real_vs_fake_scores if s <= th)
        current_acc = (tp + tn) / (len(real_vs_real_scores) + len(real_vs_fake_scores) + 1e-9)
        if current_acc > best_accuracy:
            best_accuracy, best_threshold = current_acc, th

    def get_stats_str(scores):
        if not scores: return "无数据 (0 pairs)"
        return f"均值: {np.mean(scores):.4f} | 最大值(Max): {np.max(scores):.4f} | 最小值(Min): {np.min(scores):.4f} | 样本数: {len(scores)}"

    logger.info("【各类相似度数值统计】")
    logger.info(f" 1. 真人 VS 真人 (类内匹配): {get_stats_str(real_vs_real_scores)}")
    logger.info(f" 2. 假人 VS 假人 (头模一致): {get_stats_str(fake_vs_fake_scores)}")
    logger.info(f" 3. 真人 VS 假人 (攻击匹配): {get_stats_str(real_vs_fake_scores)}")
    
    logger.info("\n【安全阈值推荐与边界分析】")
    if real_vs_fake_scores:
        max_attack_score = np.max(real_vs_fake_scores)
        logger.info(f" [!] 危险警告: 当前假人发起的最高攻击得分为 {max_attack_score:.4f}！")
        if best_threshold <= max_attack_score:
            logger.info("     (注意: 推荐阈值低于最高攻击分，可能会漏过极个别极品头模)")
            
    logger.info(f" -> 经等错误率分析，建议设置识别阈值为: {best_threshold:.2f}")
    logger.info(f" -> 在该阈值下，准确率为: {best_accuracy*100:.2f}%")
    logger.info("="*50)

    # 画图
    plt.figure(figsize=(10, 6))
    if real_vs_real_scores: plt.hist(real_vs_real_scores, bins=20, alpha=0.6, color='green', label='Real vs Real')
    if fake_vs_fake_scores: plt.hist(fake_vs_fake_scores, bins=20, alpha=0.6, color='orange', label='Fake vs Fake')
    if real_vs_fake_scores: plt.hist(real_vs_fake_scores, bins=20, alpha=0.6, color='red', label='Real vs Fake')
        
    plt.axvline(best_threshold, color='blue', linestyle='dashed', linewidth=2, label=f'Rec Threshold: {best_threshold:.2f}')
    plt.title('Face Similarity Score Distribution', fontsize=14)
    plt.xlabel('Cosine Similarity Score (Faiss IP)', fontsize=12)
    plt.ylabel('Number of Pairs', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    chart_path = os.path.join(output_folder, 'similarity_distribution_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ 统计图表已保存至: {chart_path}")


# ==========================================
# 主控制面板
# ==========================================
if __name__ == '__main__':
    # ================= 基础模型配置 =================
    YOLO_PATH = '/supercloud/llm-code/scc/scc/Liveness_Detection/yolov7_face/yolov7-w6-face.pt'
    ARCFACE_PATH = '/supercloud/llm-code/scc/scc/Liveness_Detection/model_16.pt'
    THRESHOLD = 0.45 
    
    logger.info("="*50)
    logger.info("正在初始化本地模型引擎 (仅初始化一次)...")
    logger.info("="*50)
    engine = LocalFaceComparator(YOLO_PATH, ARCFACE_PATH)

    # ================= 工作模式选择 =================
    # 可选值: 'image', 'video', 'folder'
    RUN_MODE = 'folder'  

    if RUN_MODE == 'image':
        # [单图模式] 配置区
        IMG_A = '/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/20260407-103317.jpg'
        IMG_B = '/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/20260407-103324.jpg'
        OUT_DIR = './dataset/result_image'
        run_single_image_mode(engine, IMG_A, IMG_B, OUT_DIR, THRESHOLD)

    elif RUN_MODE == 'video':
        # [视频模式] 配置区
        REF_IMG = './dataset/reference_real_face.jpg' # 基准身份图
        TEST_VIDEO = './dataset/attack_video.mp4'     # 要测试的视频
        OUT_DIR = './dataset/result_video'
        run_video_mode(engine, REF_IMG, TEST_VIDEO, OUT_DIR, THRESHOLD)

    elif RUN_MODE == 'folder':
        # [文件夹穷举模式] 配置区
        ROOT_FOLDER = '/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_fake'
        OUT_DIR = './dataset/faiss_all_combinations_results'
        run_folder_mode(engine, ROOT_FOLDER, OUT_DIR, THRESHOLD)
        
    else:
        logger.error("未知的 RUN_MODE 设置，请修改代码末尾的 RUN_MODE 变量。")