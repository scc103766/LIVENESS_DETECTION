# LIVENESS_DETECTION GitHub Release 结构说明

更新时间：2026-05-09

本文记录当前 GitHub release 包的目录职责、主要功能和版本关系。发布包只保留代码、轻量脚本、文档和 SVG 图；数据集、视频、图片、权重、训练输出和本地服务输出不进入仓库。

## 顶层结构

```text
LIVENESS_DETECTION/
├── README.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
├── DATA_ASSETS.md
├── GITHUB_RELEASE_GUIDE.md
├── flash_liveness_project.py
├── flash_liveness_project_v2.py
├── flash_liveness_project_v3.py
├── flash_liveness_project_v3_1.py
├── flash_liveness_api.py
├── flash_liveness_infer_utils.py
├── flash_physical_features.py
├── flash_liveness_v3_api_service/
├── fused_face_liveness_api.py
├── fused_face_liveness_api_v2.py
├── fused_face_liveness_eval.py
├── Face_detection_yolo_align.py
├── Face_compare.py
├── Face_infer_score.py
├── face_interactive_liveness.py
├── LivenessDetection.py
├── server_storage_manager.py
├── backbones/
├── scripts/
├── docs/
├── commands/
├── assets/
├── yolov7_face/
├── Face-Anti-Spoofing-using-DeePixBiS/
└── archive_20240320_flash_liveness/
```

## 炫彩活体版本线

| 版本 | 代码 | 文档 | 主要内容 |
| --- | --- | --- | --- |
| V1 | `flash_liveness_project.py` | `docs/flash_liveness/FLASH_LIVENESS_PROJECT_README.md` | 基础炫彩视频活体，固定帧采样，ResNet18 时序基线 |
| V2 | `flash_liveness_project_v2.py` | `docs/flash_liveness/FLASH_LIVENESS_PROJECT_V2_README.md` | 全帧顺序读取、颜色 txt 对齐、RGB+Diff 6 通道输入、Transformer 时序建模 |
| V3 | `flash_liveness_project_v3.py` | `docs/flash_liveness/FLASH_LIVENESS_PROJECT_V3_README.md`、`docs/flash_liveness/FLASH_LIVENESS_PROJECT_V3_TECHNICAL_ANALYSIS.md` | 固定协议、manifest/category split、物理特征 token、CDC texture branch、pseudo-depth、FFT auxiliary loss |
| V3 固定协议推理 | `scripts/infer_flash_liveness_v3_video.py` | `docs/flash_liveness/FLASH_LIVENESS_V3_FIXED_PROTOCOL_INFERENCE.md` | 单视频/批量视频推理、窗口评估和新域统计 |
| V3.1 | `flash_liveness_project_v3_1.py` | `docs/flash_liveness/FLASH_LIVENESS_PROJECT_V3_1_README.md` | 全帧窗口训练、窗口质量融合、类均衡采样、辅助 loss 上限和稳定性诊断 |

## V3 相关图和技术文档

| 文件 | 内容 |
| --- | --- |
| `assets/flash_liveness_v3_architecture.svg` | V3 网络结构图 |
| `assets/flash_liveness_v3_io_tensor_flow.svg` | V3 输入输出 tensor 变化图 |
| `docs/flash_liveness/FLASH_LIVENESS_PROJECT_V3_README.md` | V3 项目说明、网络结构、输入输出和训练推理说明 |
| `docs/flash_liveness/FLASH_LIVENESS_PROJECT_V3_TECHNICAL_ANALYSIS.md` | V3 完整处理链路、技术来源、设计理由、tensor 变化和论文/项目引用 |

## 服务与 API

| 路径 | 功能 |
| --- | --- |
| `flash_liveness_v3_api_service/` | V3 checkpoint FastAPI 服务，支持视频/zip 上传、txt 颜色协议、窗口推理和存储保留策略 |
| `flash_liveness_api.py` | 炫彩活体基础 API 入口 |
| `fused_face_liveness_api.py` | 人脸库匹配 + 活体 fallback 的融合 API |
| `fused_face_liveness_api_v2.py` | 融合 API V2，包含网页上传入口和 JSON 接口 |
| `fused_face_liveness_eval.py` | 融合 API 离线评测 |

## 脚本目录

| 类型 | 代表脚本 | 功能 |
| --- | --- | --- |
| 数据集构建 | `prepare_flash_liveness_video_dataset.py`、`prepare_flash_liveness_video_dataset_v2.py`、`build_flash_liveness_fixed_protocol_dataset.py` | 构建 V1/V2/V3 训练数据 |
| V3 训练 | `run_flash_liveness_v3_gpu012_training.sh`、`run_flash_liveness_v3_1_window_training.sh` | 当前 V3/V3.1 推荐训练入口 |
| 推理评测 | `infer_flash_liveness_v3_video.py`、`evaluate_flash_liveness_checkpoint.py`、`evaluate_flash_liveness_video_v2.py` | checkpoint、单视频和批量视频推理评测 |
| 物理线索分析 | `analyze_flash_physical_cues.py`、`visualize_flash_six_channel.py` | V3 物理特征与 RGB+Diff 可视化 |
| 服务维护 | `cleanup_server_storage.py` | API 输出目录存储保留清理 |
| ThunderGuard/融合评测 | `build_thunderguard_smoke_subset.py`、`evaluate_thunderguard_tg_export_dataset.py`、`evaluate_fused_face_liveness_api_v1_local.py` | 辅助活体模块评测 |

## 文档目录

```text
docs/
├── api/
├── data/
├── development/
├── flash_liveness/
├── legacy/
├── release/
└── thunderguard/
```

| 目录 | 内容 |
| --- | --- |
| `docs/api/` | 融合 API V1/V2 操作手册 |
| `docs/data/` | 数据资产边界、数据集说明、V3 数据集和数据处理说明 |
| `docs/development/` | GPU 使用约定和开发辅助说明 |
| `docs/flash_liveness/` | 炫彩活体 V1/V2/V3/V3.1 文档、优化方案、评测记录和升级任务 |
| `docs/legacy/` | 旧采集协议反推 |
| `docs/release/` | GitHub 发布整理说明 |
| `docs/thunderguard/` | ThunderGuard 复现说明 |

## 环境与 GPU 约定

- Python 命令使用 conda 环境 `anti-spoofing_scc_175`。
- 默认单 GPU 推理使用 `CUDA_VISIBLE_DEVICES=1`。
- 默认多 GPU 训练使用 `CUDA_VISIBLE_DEVICES=0,1,2` 和 `GPU_IDS=0,1,2`。
- 当前项目只允许使用物理 GPU `0,1,2`；物理 GPU `3,4,5,6` 不作为运行卡。

## 排除内容

| 类型 | 例子 |
| --- | --- |
| 数据集和媒体 | `dataset/`、`*.avi`、`*.mp4`、`*.jpg`、`*.png` |
| 权重和导出模型 | `*.pt`、`*.pth`、`*.onnx`、`*.engine`、`*.faiss` |
| 训练和服务输出 | `flash_liveness_runs/`、`flash_liveness_v3_api_service/outputs/` |
| 本地环境和缓存 | `.env*`、`__pycache__/`、`logs/`、conda/venv 目录 |
