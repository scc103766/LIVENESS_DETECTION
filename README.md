# LIVENESS_DETECTION

人脸活体检测与炫彩闪光活体研究发布仓库。这个 GitHub release 以“代码、脚本、API 和文档”为主，排除原始数据、训练输出、模型权重、本地环境和服务上传结果。

当前发布内容已同步到炫彩活体 V3/V3.1：包含 V3 主训练脚本、物理特征模块、固定协议推理脚本、V3 API 服务、网络结构图、输入输出 tensor 图和完整技术分析文档。

## 快速入口

- 项目结构说明：[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- 数据与权重边界：[DATA_ASSETS.md](DATA_ASSETS.md)
- GitHub 发布说明：[GITHUB_RELEASE_GUIDE.md](GITHUB_RELEASE_GUIDE.md)
- 炫彩 V3 文档：[docs/flash_liveness/FLASH_LIVENESS_PROJECT_V3_README.md](docs/flash_liveness/FLASH_LIVENESS_PROJECT_V3_README.md)
- 炫彩 V3 技术分析：[docs/flash_liveness/FLASH_LIVENESS_PROJECT_V3_TECHNICAL_ANALYSIS.md](docs/flash_liveness/FLASH_LIVENESS_PROJECT_V3_TECHNICAL_ANALYSIS.md)
- 炫彩 V3.1 文档：[docs/flash_liveness/FLASH_LIVENESS_PROJECT_V3_1_README.md](docs/flash_liveness/FLASH_LIVENESS_PROJECT_V3_1_README.md)
- 融合活体 API 手册：[docs/api/FUSED_FACE_LIVENESS_API_V2_README.md](docs/api/FUSED_FACE_LIVENESS_API_V2_README.md)

## 主要能力

- YOLOv7-face 人脸检测、关键点和对齐。
- ArcFace 风格人脸特征提取、图库匹配和离线比对。
- DeePixBiS/ThunderGuard 相关活体基线与融合 API。
- 炫彩闪光活体 V1/V2/V3/V3.1 训练、评测和推理脚本。
- V3 固定协议模型：RGB+Diff、颜色 token、物理特征 token、CDC texture branch、pseudo-depth、FFT auxiliary loss 和 Transformer 时序建模。
- V3 FastAPI 服务：视频/zip 上传、可选 txt 颜色协议、窗口推理、结果落盘和存储保留策略。
- 代码版历史归档 `archive_20240320_flash_liveness/`。

## V3 核心文件

- `flash_liveness_project_v3.py`
  - V3 主训练/评估/推理实现。
  - 引入固定协议数据、manifest/category split、RGB+Diff 6 通道输入、物理特征 token、FFT loss、CDC texture branch 和 pseudo-depth 辅助监督。

- `flash_liveness_project_v3_1.py`
  - V3.1 实验主线。
  - 强化完整视频读取、滑动窗口训练/推理、窗口质量融合、类均衡采样和 loss 稳定性控制。

- `flash_physical_features.py`
  - V3/V3.1 共享物理特征模块。
  - 覆盖频域/纹理、闪光响应、rPPG-like 统计、深度/法线等显式线索。

- `scripts/run_flash_liveness_v3_gpu012_training.sh`
  - 当前推荐的 V3 多 GPU 训练入口。

- `scripts/run_flash_liveness_v3_1_window_training.sh`
  - V3.1 全帧滑窗训练入口。

- `scripts/infer_flash_liveness_v3_video.py`
  - V3/V3.1 单视频和批量视频推理评测脚本。

- `flash_liveness_v3_api_service/`
  - V3 FastAPI 服务代码和调用手册。

## 推荐命令

V3 多 GPU 训练默认只使用物理 GPU `0,1,2`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 GPU_IDS=0,1,2 bash scripts/run_flash_liveness_v3_gpu012_training.sh
```

V3 单卡推理/服务默认使用物理 GPU `1`：

```bash
CUDA_VISIBLE_DEVICES=1 /home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python \
  flash_liveness_v3_api_service/app.py \
  --host 0.0.0.0 \
  --port 18131
```

项目 GPU 约定见 [docs/development/PROJECT_GPU_POLICY.md](docs/development/PROJECT_GPU_POLICY.md)。当前发布脚本不使用物理 GPU `3,4,5,6`。

## 目录结构

```text
LIVENESS_DETECTION/
├── README.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
├── flash_liveness_project.py
├── flash_liveness_project_v2.py
├── flash_liveness_project_v3.py
├── flash_liveness_project_v3_1.py
├── flash_physical_features.py
├── flash_liveness_v3_api_service/
├── fused_face_liveness_api.py
├── fused_face_liveness_api_v2.py
├── fused_face_liveness_eval.py
├── Face_detection_yolo_align.py
├── Face_compare.py
├── Face_infer_score.py
├── face_interactive_liveness.py
├── backbones/
├── scripts/
├── docs/
├── commands/
├── assets/
├── yolov7_face/
├── Face-Anti-Spoofing-using-DeePixBiS/
└── archive_20240320_flash_liveness/
```

## 不随仓库提交的内容

- 原始数据集、采集视频、图片和派生数据集。
- `flash_liveness_runs/` 训练与评测输出。
- `.pt/.pth/.onnx/.engine/.faiss/.npy/.npz` 等模型和导出资产。
- API 服务上传结果、日志、缓存和本地环境。

详见 [DATA_ASSETS.md](DATA_ASSETS.md)。
