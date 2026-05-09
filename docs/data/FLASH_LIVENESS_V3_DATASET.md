# Flash Liveness V3 数据集说明

记录时间：2026-05-07

## 当前结论

当前 V3 默认/推荐训练数据集是：

```text
dataset/flash_liveness_asset_archive_fixed_collect_protocol/
```

这是基于原始资产归档生成的“固定顺序炫光协议”派生数据集。它不会修改源视频，也不会覆盖源随机颜色 txt；派生过程是把源视频软链接到新目录，并为每个视频重新生成固定协议的同名 txt。

在未通过 `DATA_ROOT` 环境变量手动覆盖的情况下，当前训练脚本 `scripts/run_flash_liveness_v3_gpu012_training.sh` 默认使用这个数据集：

```bash
DATA_ROOT="${PROJECT_ROOT}/dataset/flash_liveness_asset_archive_fixed_collect_protocol"
```

训练命令中同时指定：

```bash
--dataset-media videos
--require-color-txt
```

因此，当前 V3 训练实际读取的是派生数据集中的视频样本，以及每个视频旁边重新生成的固定协议同名 `.txt`。

## 与原始归档数据集的关系

原始归档数据集是：

```text
dataset/flash_liveness_asset_archive_by_type/
```

V3 README 中保留了直接训练原始归档的备选方式：

```bash
conda run -n anti-spoofing_scc_175 python flash_liveness_project_v3.py train \
  --data-root dataset/flash_liveness_asset_archive_by_type \
  --dataset-media videos \
  --missing-color-protocol collect_flash
```

但这不是当前推荐主训练方式。当前推荐主训练方式是先使用固定顺序炫光协议派生集：

```bash
conda run -n anti-spoofing_scc_175 python flash_liveness_project_v3.py train \
  --data-root dataset/flash_liveness_asset_archive_fixed_collect_protocol \
  --dataset-media videos \
  --require-color-txt
```

## 固定炫光协议

固定颜色顺序来自 `scripts/collect_flash_liveness_video.py`：

```text
(255, 20, 255) -> (20, 255, 20) -> (255, 20, 20)
```

也就是以下顺序：

| 顺序 | 颜色含义 | RGB | packed int |
|---:|---|---|---:|
| 1 | 紫/品红光 | `(255, 20, 255)` | 16717055 |
| 2 | 绿光 | `(20, 255, 20)` | 1376020 |
| 3 | 红光 | `(255, 20, 20)` | 16716820 |

对应 packed int：

```text
16717055 -> 1376020 -> 16716820
```

默认时间协议：

```text
warmup: 1.0s 黑屏
hold:   0.35s 每种颜色保持时间
tail:   0.5s 黑屏收尾
```

完整时间线可以理解为：

```text
黑屏 warmup 1.0s
  -> 紫/品红光 (255, 20, 255) 保持 0.35s
  -> 绿光       (20, 255, 20) 保持 0.35s
  -> 红光       (255, 20, 20) 保持 0.35s
  -> 黑屏 tail 0.5s
```

其中黑屏阶段在同名 txt 中写为颜色值 `0`。

### 为什么前后都有黑屏

开头的 `warmup 1.0s` 主要用于建立中性基线：

- 让摄像头曝光、白平衡、自动增益先稳定下来。
- 让模型看到打光前的人脸初始状态，便于后续比较炫光刺激后的颜色/纹理/物理响应变化。
- 避免视频一开始就进入强刺激颜色，导致前几帧受启动、对焦或曝光抖动影响。

结尾的 `tail 0.5s` 主要用于收尾和恢复观察：

- 避免视频刚好结束在某一种强颜色闪光中，使最后一段响应被截断。
- 给皮肤反射、屏幕反射、纸张/面具材质响应一个回到中性状态的短窗口。
- 给帧级颜色标签和视频末尾留出缓冲，降低最后几帧对齐误差的影响。

所以这两个黑屏不是重复设计：开头黑屏是“刺激前基线”，结尾黑屏是“刺激后恢复/收尾”。它们在训练中都作为中性颜色阶段参与时序建模。

## 目录内容

数据集根目录包含：

```text
dataset/flash_liveness_asset_archive_fixed_collect_protocol/
  README.md
  manifest.tsv
  summary.tsv
  skipped.tsv
  videos/
```

其中：

- `manifest.tsv`：训练样本清单，包含 `media_type`、`label`、`category`、`source_group`、`archive_path`、`source_path`、`note`。
- `summary.tsv`：按 label/category 汇总的样本数和帧数。
- `skipped.tsv`：派生集构建时跳过的视频记录。
- `videos/`：按类别划分的视频目录。

`videos/` 下每个类别目录中包含：

- 视频软链接：指向原始归档或公共/新增来源视频。
- 同名 `.txt`：固定 collect_flash 协议颜色序列。

## 总体统计

当前派生数据集统计：

| 项目 | 数量 |
|---|---:|
| 视频样本 | 14240 |
| 同名 txt | 14240 |
| 总帧数 | 1405722 |
| live 样本 | 4855 |
| spoof 样本 | 9385 |
| category 数 | 22 |
| 构建时跳过样本 | 3 |

## 类别明细

| label | category | count | frames |
|---|---|---:|---:|
| live | live_real | 9 | 2342 |
| live | live_real_extra_angles | 5 | 1228 |
| live | live_real_flash_archive | 4569 | 438088 |
| live | live_real_lighting_pair_control_archive | 265 | 23265 |
| live | live_real_outdoor | 7 | 2523 |
| spoof | advanced_paper_attack | 14 | 1536 |
| spoof | cutout_attack | 15 | 6363 |
| spoof | cylinder_paper_attack | 8 | 1250 |
| spoof | flat_attack_flash_archive | 8101 | 787536 |
| spoof | history_head_model_attack | 374 | 32679 |
| spoof | latex_mask_attack | 10 | 3909 |
| spoof | mask_attack | 4 | 1019 |
| spoof | mask_fake_face_wig_hat_attack | 8 | 2578 |
| spoof | on_actor_paper_attack | 72 | 9993 |
| spoof | print_cut_paper_attack | 13 | 2380 |
| spoof | public_3d_attack | 10 | 1749 |
| spoof | replay_display_attack | 5 | 2288 |
| spoof | replay_mobile_attack | 20 | 6851 |
| spoof | silicone_mask_attack | 21 | 10178 |
| spoof | textile_3d_mask_attack | 23 | 6256 |
| spoof | three_d_head_model_attack_flash_archive | 641 | 54580 |
| spoof | three_d_paper_mask_attack | 46 | 7131 |

## 跳过样本

构建派生数据集时有 3 个视频因为 `no_frame_count` 被跳过：

| category | label | reason |
|---|---|---|
| flat_attack_flash_archive | spoof | no_frame_count |
| history_head_model_attack | spoof | no_frame_count |
| three_d_head_model_attack_flash_archive | spoof | no_frame_count |

## 训练读取方式

V3 数据加载时优先从 `manifest.tsv` 读取样本，并封装为：

```text
LivenessSample(
  media_path,
  txt_path,
  label,
  media_type,
  category,
  source_group,
)
```

当前训练使用 `--dataset-media videos`，所以只读取 `videos` 媒体类型。由于训练命令带 `--require-color-txt`，缺少同名 txt 的样本不会被当作正常训练样本使用。

V3 自动切分 train/val/test 时按以下组合分层：

```text
label + media_type + category
```

这样可以尽量保证 live、纸质攻击、屏幕回放、手机回放、头模、面具、硅胶等类别都进入各个 split。

## 复核命令

查看训练脚本当前默认数据集：

```bash
sed -n '1,80p' scripts/run_flash_liveness_v3_gpu012_training.sh
```

查看派生数据集说明：

```bash
sed -n '1,40p' dataset/flash_liveness_asset_archive_fixed_collect_protocol/README.md
```

查看类别统计：

```bash
sed -n '1,120p' dataset/flash_liveness_asset_archive_fixed_collect_protocol/summary.tsv
```

统计视频软链接和同名 txt：

```bash
find dataset/flash_liveness_asset_archive_fixed_collect_protocol/videos -maxdepth 2 -type l | wc -l
find dataset/flash_liveness_asset_archive_fixed_collect_protocol/videos -maxdepth 2 -type f -name '*.txt' | wc -l
```
