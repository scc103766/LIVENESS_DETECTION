# Flash Liveness V3 API Service

这个服务封装当前 V3 固定协议模型：

```text
/supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/flash_liveness_v3_fixed_protocol_gpu3/best_flash_liveness_model.pth
```

服务参考 `thunderguard_facepreprocessor_service/app_v1.py` 的结构，提供 FastAPI 入口、网页上传页、`/health` 和 `/predict`。

## 输入

`POST /predict` 支持：

- 普通视频：`.mp4` / `.avi` / `.mov` / `.mkv` / `.webm` / `.m4v`
- zip：包含视频和可选同名 `.txt`
- 普通视频上传时可额外上传 `txt_file`

如果没有提供 txt，服务会按模型训练配置自动使用 `collect_flash` 固定协议补齐逐帧颜色：

```text
warmup 1.0s black
-> (255, 20, 255) hold 0.35s
-> (20, 255, 20) hold 0.35s
-> (255, 20, 20) hold 0.35s
-> repeat colors continuously
-> tail 0.5s black
```

当前 V3 best 训练集是 `dataset/flash_liveness_asset_archive_fixed_collect_protocol`，该数据集的视频时长不是固定值，颜色 txt 是按每个视频实际 `frame_count/fps` 生成的，并且使用 `restore_seconds=0.0`。因此采集端可以录制任意时长，但必须使用连续切色协议，不要使用 `restore_seconds=0.15` 的 V3.1 restore_original 协议。

## 启动

```bash
CUDA_VISIBLE_DEVICES=1 /home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python \
/supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_v3_api_service/app.py \
--host 0.0.0.0 \
--port 18131 \
--checkpoint-path /supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/flash_liveness_v3_fixed_protocol_gpu3/best_flash_liveness_model.pth \
--output-dir /raid/scc/data/liveness_v3_server_result \
--storage-max-videos 2000
```

GPU 约定：本项目后续只使用物理 `0/1/2` 三张卡，`3/4/5/6` 不作为本项目运行卡。V3 单卡推理默认使用物理 `1` 号卡；如需调整，只能改到 `0` 或 `2`。

也可以直接查看：

```bash
cat /supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_v3_api_service/run_api_command.txt
```

## 接口

健康检查：

```bash
curl http://127.0.0.1:18131/health
```

上传视频：

```bash
curl -X POST "http://127.0.0.1:18131/predict" \
  -F "file=@/path/to/input.mp4"
```

上传视频和 txt：

```bash
curl -X POST "http://127.0.0.1:18131/predict" \
  -F "file=@/path/to/input.mp4" \
  -F "txt_file=@/path/to/input.txt"
```

上传 zip：

```bash
curl -X POST "http://127.0.0.1:18131/predict" \
  -F "file=@/path/to/input.zip"
```

本机路径调用：

```bash
curl -X POST "http://127.0.0.1:18131/predict_path" \
  -H "Content-Type: application/json" \
  -d '{"video_path":"/path/to/input.mp4","txt_path":"/path/to/input.txt"}'
```

## 返回重点字段

- `result`: `live` 或 `spoof`
- `probability_live`: V3 模型输出的真人概率
- `threshold`: checkpoint 中保存的阈值，默认 `0.9375`
- `num_frames`: 解码帧数
- `num_windows`: 滑窗推理窗口数量
- `txt_source`: `provided` 或 `generated_collect_flash_protocol`
- `metadata_json`: 本次请求的完整元数据落盘路径

## threshold 来源

V3 API 默认不使用固定 `0.5` 作为最终判定阈值，而是读取 checkpoint 中保存的 `threshold`。当前线上推荐 checkpoint：

```text
flash_liveness_runs/flash_liveness_v3_fixed_protocol_gpu3/best_flash_liveness_model.pth
```

保存的阈值是：

```text
threshold = 0.9375
```

这个值来自训练脚本 `flash_liveness_project_v3.py` 的验证集评估流程：

1. 每个 epoch 结束后，`evaluate_model()` 会在验证集上收集所有样本的 `probability_live`。
2. `find_best_threshold(labels, probs)` 会把验证集里出现过的预测概率值取 6 位小数后作为候选阈值，并额外加入 `0.5`。
3. 对每个候选阈值，`compute_binary_metrics()` 计算 `APCER`、`BPCER` 和 `ACER`。
4. 选择 `balanced_accuracy = 1 - ACER` 最大的候选阈值。
5. `save_checkpoint()` 把这个验证集最优阈值写入 checkpoint 的 `threshold` 字段。

因此，`0.9375` 不是手工写死的经验值，也不是训练 loss 里的参数；它是当前 best checkpoint 在验证集上为了最小化 `ACER` 自动选出的决策边界。

当前 checkpoint 的保存指标中，验证集对应：

```text
epoch: 11
val_auc: 0.9998140976497336
val_threshold: 0.9375
val_apcer: 0.005857294994675187
val_bpcer: 0.0020597322348094747
val_acer: 0.00395851361474233
val_tp/tn/fp/fn: 969 / 1867 / 11 / 2
```

推理时判定规则是：

```text
probability_live >= threshold -> live
probability_live <  threshold -> spoof
```

所以如果一个真人视频的 `probability_live` 高于 `0.5` 但低于 `0.9375`，V3 API 仍会返回 `spoof`。这类结果通常表示当前阈值偏向降低攻击误放行率；对新采集域真人样本可能会带来更高的真人误拒风险。

## 输出目录

默认输出目录：

```text
/raid/scc/data/liveness_v3_server_result
```

每次请求会生成一个 request id 子目录，保存上传文件、解压后的 zip 内容和 `metadata.json`。

## 存储保留策略

服务默认启用视频保留上限：

```text
--storage-max-videos 2000
```

该配置在服务启动时读取，已经运行中的进程不会被热更新；修改命令后需要下次重启服务才会生效。

当输出目录内保存的视频数量超过阈值后，服务会从最旧的请求开始做归档：

- 备份 `metadata.json` 中的推理结论。
- 从请求视频中抽取 5 张代表帧，保存为 jpg。
- 删除该请求目录下的 `.mp4/.avi/.mov/.mkv/.webm/.m4v/.zip` 大文件。
- 小文件和清理标记保留在原 request 目录。

默认备份目录：

```text
/raid/scc/data/liveness_v3_server_result/_retention_backups
```

手动清理：

```bash
python /supercloud/llm-code/scc/scc/Liveness_Detection/scripts/cleanup_server_storage.py \
  --service v3 \
  --max-videos 2000
```

只查看不删除：

```bash
python /supercloud/llm-code/scc/scc/Liveness_Detection/scripts/cleanup_server_storage.py \
  --service v3 \
  --max-videos 2000 \
  --dry-run
```
