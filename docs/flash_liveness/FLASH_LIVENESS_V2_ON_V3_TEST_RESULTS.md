# Flash Liveness V2 使用 V3 Test 数据集测试结果

本文档记录 `flash_liveness_project_v2.py` / V2 checkpoint 在当前 V3 固定顺序炫光协议 test split 上的评测方式和结果归档状态。

## 当前状态

截至当前工作区检查，V2 评测脚本已经支持使用 V3 的 test 数据集与 V3 manifest/category split 加载方式，但当前仓库内没有可用的 V2 checkpoint：

```text
flash_liveness_runs/**/best_flash_liveness_model.pth: 未找到
flash_liveness_runs/**/last_flash_liveness_model.pth: 未找到
```

因此本次不能产出真实的 V2 accuracy / APCER / BPCER / ACER / AUC 数值。已有的历史 V2 评测 JSON 指向的 checkpoint 路径为：

```text
flash_liveness_runs/flash_liveness_gpu1_v2_fullframes_parallel_manual/best_flash_liveness_model.pth
```

该文件在当前工作区不存在。恢复该 checkpoint 后，可直接运行下方命令得到正式结果。

## 测试数据集

本次目标测试集使用 V3 固定顺序炫光协议数据集：

```text
dataset/flash_liveness_asset_archive_fixed_collect_protocol
```

V3 test split 由 `flash_liveness_project_v3.py` 的 `discover_dataset_splits(...)` 产生，参数固定为：

```text
val_ratio=0.2
test_ratio=0.1
seed=42
media_filter=videos
require_color_txt=True
```

当前 test split 验证统计：

| split | total | live | spoof | category coverage |
| --- | ---: | ---: | ---: | ---: |
| test | 1425 | 486 | 939 | 22/22 |

test 是普适性测试集，覆盖当前归档中的全部攻击/真人类别，而不是只针对某一种特殊攻击类型。

## V2 评测脚本改动

评测入口：

```text
scripts/evaluate_flash_liveness_video_v2.py
```

新增能力：

1. 增加 `--split-source v3`，使用 V3 manifest/category split 发现样本。
2. V3 模式下读取固定顺序炫光协议数据集中的视频和同名 `.txt`。
3. 保留 V2 原有 `FlashLivenessDataset.process_video(video_path, txt_path)` 输入格式，模型前向仍为：

```text
frames_tensor: [T, 6, H, W]
color_tensor:  [T, 4]
padding_mask:  [1, T]
```

4. 输出逐样本 `category`、`source_group`，并额外汇总每个攻击类型/真人类别的指标。

## 正式测试命令

恢复 V2 checkpoint 后运行：

```bash
conda run -n anti-spoofing_scc_175 python scripts/evaluate_flash_liveness_video_v2.py \
  --checkpoint flash_liveness_runs/flash_liveness_gpu1_v2_fullframes_parallel_manual/best_flash_liveness_model.pth \
  --data-root dataset/flash_liveness_asset_archive_fixed_collect_protocol \
  --split-source v3 \
  --split test \
  --output-dir flash_liveness_runs/v2_on_v3_fixed_protocol_test \
  --device cuda:0
```

如果 checkpoint 放在其他路径，只需要替换 `--checkpoint`。

## 结果输出文件

正式运行后，结果会写入：

```text
flash_liveness_runs/v2_on_v3_fixed_protocol_test/
```

包含：

| 文件 | 内容 |
| --- | --- |
| `v2_video_eval_summary.json` | 总体指标、阈值、样本数、失败样本数、分组指标索引 |
| `v2_video_eval_rows.csv` | 每个视频的概率、预测、标签、类别、source_group、帧数、错误信息 |
| `v2_video_eval_category_metrics.csv` | 按 V3 category 汇总的 accuracy/APCER/BPCER/ACER/AUC/EER |
| `v2_video_eval_source_metrics.csv` | 按 source_key 汇总的指标 |
| `v2_video_eval_label_metrics.csv` | live/spoof 两类分组指标 |

## 待补充正式数值

当前缺少 V2 checkpoint，以下指标等待恢复模型后由脚本自动生成：

| metric | value |
| --- | --- |
| accuracy | 待运行 |
| apcer | 待运行 |
| bpcer | 待运行 |
| acer | 待运行 |
| auc | 待运行 |
| eer | 待运行 |
| threshold | 使用 checkpoint 内阈值，除非显式传入 `--threshold` |

正式评测完成后，以 `v2_video_eval_summary.json` 中的 `overall_metrics` 为准，并用 `v2_video_eval_category_metrics.csv` 检查每一种攻击类型是否存在明显短板。
