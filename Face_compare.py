import os
import cv2
import torch
import numpy as np
import faiss
import logging
import itertools
import matplotlib.pyplot as plt

# ==========================================
# 0. 导入你本地的模型类 (请根据你的实际路径修改)
# ==========================================
from Face_detection_yolo_align import YOLOv7_face_mkl 
from backbones import get_model

# 初始化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalFaceComparator:
    def __init__(self, yolo_path, arcface_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. 加载本地检测与识别模型
        self.detector = self._load_detection_model(yolo_path)
        self.recognizer = self._load_recognition_model(arcface_path, architecture='r100')
        
        # 2. 初始化 Faiss (使用点积 IndexFlatIP，配合L2归一化计算余弦相似度)
        self.embedding_dim = 512  # ArcFace 通常输出 512 维
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
    def _load_detection_model(self, model_path):
        try:
            # 此处调用你本地的 YOLO 类
            model = YOLOv7_face_mkl(model_path=model_path, device=self.device)
            logger.info(f"YOLOv7人脸检测模型已在 {self.device} 上成功加载。")
            return model
        except Exception as e:
            logger.error(f"加载检测模型时出错: {e}", exc_info=True)
            raise

    def _load_recognition_model(self, model_path, architecture='r100'):
        try:
            # 此处调用你本地的 ArcFace 组网函数
            model = get_model(architecture, fp16=False)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval().to(self.device)
            logger.info(f"ArcFace识别模型已在 {self.device} 上成功加载。")
            return model
        except Exception as e:
            logger.error(f"加载识别模型时出错: {e}", exc_info=True)
            raise

    def process_and_get_embedding(self, img_bgr, img_name="temp"):
        """
        核心修改：完美对接你 YOLO 的 infer_batch 方法和 aligned_faces 输出
        """
        # 1. 调用你的 YOLO 推理单张图片 (infer_batch 接收 list)
        detect_results = self.detector.infer_batch([img_bgr], [img_name]) 
        
        if not detect_results:
            return None, None
            
        result = detect_results[0] # 取出第一张图片的结果字典
        
        # 判断是否检测到人脸
        if len(result['bbox']) == 0:
            return None, None
            
        # 提取第一个人脸 (通常 infer_batch 出来的是按置信度排序或原图顺序)
        bbox = result['bbox'][0]
        x1, y1, x2, y2 = map(int, bbox)
        
        # 2. 直接获取你 YOLO 内部做过关键点仿射对齐的 112x112 人脸
        aligned_face_bgr = result['aligned_faces'][0]
        
        # 你代码里写了如果对齐失败会返回全黑的图像 np.zeros，这里做个拦截
        if not aligned_face_bgr.any(): 
            logger.warning(f"图片 {img_name} 关键点对齐失败。")
            return None, None
        
        # 3. ArcFace 标准预处理
        # 你的对齐图已经是 112x112，只需转 RGB 并归一化即可
        face_rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
        face_tensor = np.transpose(face_rgb, (2, 0, 1)).astype(np.float32) # (H,W,C) -> (C,H,W)
        face_tensor = (face_tensor / 127.5) - 1.0  # 归一化到 [-1, 1]
        face_tensor = torch.from_numpy(face_tensor).unsqueeze(0).to(self.device)
        
        # 4. ArcFace 提取特征
        with torch.no_grad():
            embedding = self.recognizer(face_tensor)
            embedding = embedding.cpu().numpy() # 转回 numpy
            
        # 5. L2 归一化 (Faiss 计算余弦相似度必须的一步)
        faiss.normalize_L2(embedding)
        
        return embedding, (x1, y1, x2, y2)
    
    
    def compare_with_faiss(self, emb_A, emb_B):
        """使用 Faiss 计算两个向量的相似度"""
        self.faiss_index.reset() # 清空底库
        self.faiss_index.add(emb_A) # 把 A 放入 Faiss 底库
        
        # 搜索 B，返回距离(即相似度D)和索引(I)
        D, I = self.faiss_index.search(emb_B, 1) 
        return D[0][0] # 返回具体的相似度分值

    def resize_for_concat(self, img1, img2, target_height=600):
        """等比例缩放两张图片到相同高度，防止拼接报错"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        new_w1 = int(w1 * (target_height / h1))
        new_w2 = int(w2 * (target_height / h2))
        return cv2.resize(img1, (new_w1, target_height)), cv2.resize(img2, (new_w2, target_height))

# ==========================================
# 主程序逻辑
# ==========================================
if __name__ == '__main__':
    # --- 路径配置 ---
    YOLO_PATH = '/supercloud/llm-code/scc/scc/Liveness_Detection/yolov7_face/yolov7-w6-face.pt'
    ARCFACE_PATH = '/supercloud/llm-code/scc/scc/Liveness_Detection/model_16.pt'
    
     
    # 【修改这里】：只需要指定包含“真人”和“攻击”的上级文件夹路径即可
    ROOT_DATA_FOLDER = '/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/1' 
    OUTPUT_FOLDER = './dataset/faiss_all_combinations_results'
    
       
    # 初始化统计日志文件
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    log_file_path = os.path.join(OUTPUT_FOLDER, 'similarity_statistics.log')
    
    # 配置文件日志处理器
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    
    THRESHOLD = 0.45 # 默认画图基准阈值
    
    logger.info("="*50)
    logger.info("初始化本地 1v1 对比引擎...")
    logger.info("="*50)
    engine = LocalFaceComparator(YOLO_PATH, ARCFACE_PATH)
    
    # ==========================================
    # 步骤 1 & 2: 扫描全目录并提取特征
    # ==========================================
    all_image_paths = []
    for root, dirs, files in os.walk(ROOT_DATA_FOLDER):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                all_image_paths.append(os.path.join(root, file))
                
    total_images = len(all_image_paths)
    logger.info(f"全目录共扫描到 {total_images} 张图片。")
    if total_images < 2:
        exit()

    logger.info(">>> 正在进行全局特征提取...")
    features_dict = {}
    for path in all_image_paths:
        img_bgr = cv2.imread(path)
        if img_bgr is None: continue
        img_name = os.path.basename(path)
        emb, box = engine.process_and_get_embedding(img_bgr, img_name)
        if emb is None: continue # 如果未检测到人脸则跳过
        
        parent_folder = os.path.basename(os.path.dirname(path)) # '真人' 或 '攻击'
        identifier = f"{parent_folder}_{os.path.splitext(img_name)[0]}"
        features_dict[path] = {'identifier': identifier, 'category': parent_folder, 'emb': emb}

    # ==========================================
    # 步骤 3: 穷举比对并收集统计数据
    # ==========================================
    all_pairs = list(itertools.combinations(features_dict.keys(), 2))
    total_pairs = len(all_pairs)
    logger.info(f">>> 开始执行 {total_pairs} 次穷举比对...\n")
    
    # 统计容器
    real_vs_real_scores = []
    fake_vs_fake_scores = []
    real_vs_fake_scores = []
    
    count = 0
    for path_A, path_B in all_pairs:
        count += 1
        data_A = features_dict[path_A]
        data_B = features_dict[path_B]
        
        # 计算相似度
        sim_score = engine.compare_with_faiss(data_A['emb'], data_B['emb'])
        
        # 分类归档统计
        cat_A, cat_B = data_A['category'], data_B['category']
        if cat_A == '真人' and cat_B == '真人':
            real_vs_real_scores.append(sim_score)
        elif cat_A == '攻击' and cat_B == '攻击':
            fake_vs_fake_scores.append(sim_score)
        else:
            real_vs_fake_scores.append(sim_score)
        
        # ---------------- 渲染与保存图片 ----------------
        img_A = cv2.imread(path_A)
        img_B = cv2.imread(path_B)
        render_A, render_B = engine.resize_for_concat(img_A, img_B, target_height=500)
        concat_img = cv2.hconcat([render_A, render_B])
        
        info_panel = np.zeros((80, concat_img.shape[1], 3), dtype=np.uint8)
        is_same = sim_score > THRESHOLD
        res_txt = "MATCH" if is_same else "MISMATCH"
        color = (0, 255, 0) if is_same else (0, 0, 255)
        text = f"Faiss Score: {sim_score:.4f} | {res_txt}"
        
        cv2.putText(info_panel, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        final_img = cv2.vconcat([concat_img, info_panel])
        
        save_name = f"{data_A['identifier']}_VS_{data_B['identifier']}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, save_name), final_img)
        # ------------------------------------------------

    # ==========================================
    # 步骤 4: 统计分析与最佳阈值推荐
    # ==========================================
      # 步骤 4: 统计分析与最佳阈值推荐
    # ==========================================
    logger.info("="*50)
    logger.info(">>> 对比任务完成！开始生成统计分析报告...")
    
    # 寻找最佳精度 (Accuracy) 的阈值切分点
    best_threshold = 0.45
    best_accuracy = 0.0
    for th in np.arange(0.0, 1.0, 0.01):
        tp = sum(1 for s in real_vs_real_scores if s > th)
        tn = sum(1 for s in real_vs_fake_scores if s <= th)
        current_acc = (tp + tn) / (len(real_vs_real_scores) + len(real_vs_fake_scores) + 1e-9)
        if current_acc > best_accuracy:
            best_accuracy = current_acc
            best_threshold = th

    # 封装一个内部小函数，用于优雅地格式化均值、最大值和最小值
    def get_stats_str(scores):
        if not scores:
            return "无数据 (0 pairs)"
        return f"均值: {np.mean(scores):.4f} | 最大值(Max): {np.max(scores):.4f} | 最小值(Min): {np.min(scores):.4f} | 样本数: {len(scores)}"

    # 写入日志报告 (包含 Mean, Max, Min)
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
    logger.info(f" -> 在该阈值下，区分真实身份与伪造攻击的理论准确率为: {best_accuracy*100:.2f}%")
    logger.info("="*50)

    # ==========================================
    # 步骤 5: 绘制分布统计图 (Matplotlib)
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图，alpha=0.6 产生半透明叠加效果
    if real_vs_real_scores:
        plt.hist(real_vs_real_scores, bins=20, alpha=0.6, color='green', label='Real vs Real (Genuine)')
    if fake_vs_fake_scores:
        plt.hist(fake_vs_fake_scores, bins=20, alpha=0.6, color='orange', label='Fake vs Fake (Mask vs Mask)')
    if real_vs_fake_scores:
        plt.hist(real_vs_fake_scores, bins=20, alpha=0.6, color='red', label='Real vs Fake (Spoof Attack)')
        
    # 画出推荐的阈值线
    plt.axvline(best_threshold, color='blue', linestyle='dashed', linewidth=2, 
                label=f'Recommended Threshold: {best_threshold:.2f}')
    
    plt.title('Face Similarity Score Distribution (Live vs Spoof)', fontsize=14)
    plt.xlabel('Cosine Similarity Score (Faiss IP)', fontsize=12)
    plt.ylabel('Number of Pairs', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存统计图表
    chart_path = os.path.join(OUTPUT_FOLDER, 'similarity_distribution_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    logger.info(f"统计图表已生成并保存至: {chart_path}")
    logger.info(f"详细统计数据已写入日志: {log_file_path}")