# Flash Liveness V3 Fixed-Protocol 推理说明

本文档对应当前基准模型：

`weights/flash_liveness_v3_fixed_protocol/best_flash_liveness_model.pth`

GitHub release 中该权重以分片形式保存。首次 clone 后先恢复：

```bash
bash weights/flash_liveness_v3_fixed_protocol/restore_best_weight.sh
```

配套推理脚本：

`/supercloud/llm-code/scc/scc/Liveness_Detection/scripts/infer_flash_liveness_v3_video.py`

## 目标

- 输入单个视频，输出真假人判断。
- 对新域协议数据集批量推理并统计准确率、AUC、EER、APCER、BPCER、ACER。
- 保留炫彩活体闪光需求：优先读取同名逐帧颜色 `txt`，缺失时按 `collect_flash` 协议自动补齐颜色时间线。
- 对长视频启用滑窗推理，避免只截取前 256 帧导致时序信息丢失。

## 炫彩闪光处理逻辑

- 逐帧读取视频，而不是稀疏抽样固定帧数。
- 优先读取同名 `txt` 中的 `frame_index,color_int` 颜色协议。
- 如果视频没有 `txt`，脚本会按训练配置自动使用 `collect_flash` 规则补协议：
  - `warmup_seconds=1.0`
  - `hold_seconds=0.35`
  - `tail_seconds=0.5`
- 每帧会构造 4 维颜色特征：`R/G/B/transition`
- 同时保留 V3 模型的：
  - RGB + diff 六通道输入
  - physical cues 特征
  - 炫彩时序信息

## 长视频推理策略

训练时该模型使用：

- `max_train_frames=256`
- `max_eval_frames=256`

为了适配真实长视频和新域协议视频，推理脚本默认使用：

- `window_size=256`
- `window_stride=128`
- `window_fusion=quality_trimmed_mean`

含义：

- 先把整段视频切成多个窗口。
- 每个窗口分别输出 `probability_live`。
- 再按窗口质量做裁剪加权融合。

这样可以减少以下问题对最终结果的干扰：

- 局部剧烈晃动
- 解码异常
- 某些片段人脸不稳定
- 炫彩闪光切换瞬间造成的局部异常窗口

## 单视频推理

建议使用训练环境：

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python \
  scripts/infer_flash_liveness_v3_video.py \
  --checkpoint weights/flash_liveness_v3_fixed_protocol/best_flash_liveness_model.pth \
  --video-path /path/to/input.mp4 \
  --txt-path /path/to/input.txt \
  --output-dir /supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/v3_fixed_protocol_single_infer
```

如果 `txt` 缺失，可以省略 `--txt-path`，脚本会自动按训练闪光协议补齐。

输出文件：

- `predictions.csv`
- `summary.json`
- `summary.md`

单视频时重点字段包括：

- `probability_live`
- `threshold`
- `prediction_name`
- `num_frames`
- `num_windows`
- `mean_window_quality`

## 新域协议数据集批量统计

数据集路径：

`/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_new_domain_video_protocol_v1v2`

批量评估命令：

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python \
  scripts/infer_flash_liveness_v3_video.py \
  --checkpoint weights/flash_liveness_v3_fixed_protocol/best_flash_liveness_model.pth \
  --data-root /supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_new_domain_video_protocol_v1v2 \
  --split all \
  --window-size 256 \
  --window-stride 128 \
  --window-fusion quality_trimmed_mean \
  --output-dir /supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/flash_liveness_v3_fixed_protocol_new_domain_eval
```

批量输出包含：

- 整体指标
- 按 `split` 的指标
- 按 `source_group` 的指标
- 按 `category` 的指标
- 每个视频的逐条预测结果

## 结果文件说明

- `predictions.csv`
  - 每个视频一行
  - 包含标签、预测、分数、窗口数、窗口质量
- `summary.json`
  - 机器可读总结果
- `summary.md`
  - 人类可读测试报告

## 说明

- 当前脚本直接复用 `flash_liveness_project_v3.py` 的模型定义和预处理。
- 推理阶段额外引入了 V3.1 的窗口融合工具函数，但不改变 V3 基准模型权重本身。
- 也就是说：模型仍然是当前 `best_flash_liveness_model.pth`，只是视频级推理方式更适合炫彩长视频和新域协议评测。

## 已产出结果

本次对新域协议数据集的 GPU 评测结果已整理到：

[`FLASH_LIVENESS_V3_FIXED_PROTOCOL_NEW_DOMAIN_RESULTS.md`](FLASH_LIVENESS_V3_FIXED_PROTOCOL_NEW_DOMAIN_RESULTS.md)
