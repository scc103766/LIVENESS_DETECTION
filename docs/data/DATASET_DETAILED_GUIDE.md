# Liveness Dataset Detailed Guide

更新时间: 2026-04-22

## 当前目标

本文件记录 `/supercloud/llm-code/scc/scc/Liveness_Detection/dataset` 和 `/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset` 下可用于炫彩活体工作的资产。

本次整理不再只围绕 `flash_liveness_project_v2.py` 的训练格式。只要是可用的炫彩活体相关视频或图片，都会统计和归档；是否带同名 `.txt` 只作为补充信息。

## 类型归档目录

已生成按类型整理的统一归档:

```text
/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_asset_archive_by_type
├── videos/<type>/
├── images/<type>/
├── manifest.tsv
├── summary.tsv
└── README.md
```

说明:

- 源文件保持原位，归档目录中全部是软链接。
- `videos/` 下按真人/攻击类型归档视频。
- `images/` 下按真人/攻击类型归档图片。
- `manifest.tsv` 记录每个归档软链接对应的原始路径、媒体类型、标签和类别。
- `summary.tsv` 记录每个类别的视频/图片数量。
- 整理脚本: `scripts/organize_flash_liveness_assets_by_type.py`

## 总体统计

归档资产总量:

- 视频: 14,243
- 图片: 300,785
- 总计: 315,028

按标签聚合:

- live: 213,430
- spoof: 101,595
- unknown: 3

`unknown` 代表路径语义不足以可靠判断 live/spoof，已归档但不建议直接作为监督标签使用。

## 视频归档统计

| 标签 | 类型目录 | 数量 | 说明 |
| --- | --- | ---: | --- |
| live | `live_real_flash_archive` | 4568 | 20240320 归档真人炫彩活体视频 |
| live | `live_real_lighting_pair_control_archive` | 265 | `raw/remove` 中已确认的真人对照试验视频，包含曝光明亮与正常光照成对样本 |
| live | `live_real` | 9 | 当前 `dataset` 中公开真人/显式 Real 视频 |
| live | `live_real_outdoor` | 7 | 室外真人视频 |
| live | `live_real_extra_angles` | 5 | 真人额外拍摄角度视频 |
| spoof | `flat_attack_flash_archive` | 8102 | 20240320 归档平面/屏幕/打印等攻击视频 |
| spoof | `three_d_head_model_attack_flash_archive` | 642 | 20240320 归档 3D 头模攻击视频 |
| spoof | `history_head_model_attack` | 375 | `炫彩闪烁活体/20230828 头模 正常光` 历史头模攻击视频 |
| spoof | `on_actor_paper_attack` | 72 | 贴附/佩戴在真人上的纸质攻击 |
| spoof | `three_d_paper_mask_attack` | 46 | 3D 纸面头模攻击 |
| spoof | `textile_3d_mask_attack` | 23 | 纺织 3D 面具攻击 |
| spoof | `silicone_mask_attack` | 21 | 硅胶面具攻击 |
| spoof | `replay_mobile_attack` | 20 | 手机屏幕重放攻击 |
| spoof | `cutout_attack` | 15 | Cutout 攻击 |
| spoof | `advanced_paper_attack` | 14 | Advanced Paper attacks iBeta2 |
| spoof | `print_cut_paper_attack` | 13 | 打印裁剪纸质攻击 |
| spoof | `public_3d_attack` | 10 | 公开 3D 攻击 |
| spoof | `latex_mask_attack` | 10 | 乳胶面具攻击 |
| spoof | `mask_fake_face_wig_hat_attack` | 8 | iBeta3 未标注设备目录中已确认的戴面具、帽子和假头发 fake 人脸视频 |
| spoof | `cylinder_paper_attack` | 8 | 柱状纸质攻击 |
| spoof | `replay_display_attack` | 5 | 显示器/PC 屏幕重放攻击 |
| spoof | `mask_attack` | 4 | iBeta3 显式 Mask 攻击 |
| spoof | `history_flash_collect_attack` | 1 | `炫彩闪烁活体/20230607_liveness_Detection` 历史补充攻击视频 |

## 图片归档统计

| 标签 | 类型目录 | 数量 | 说明 |
| --- | --- | ---: | --- |
| live | `live_real_image_flash_archive` | 208545 | 20240320 归档真人图片/帧图 |
| live | `live_real_lighting_pair_control_archive` | 1 | `raw/remove` 真人对照试验中的图片 |
| live | `live_real_selfie_image` | 25 | 当前 `dataset/Selfies` 真人自拍图片 |
| live | `live_real_replay_display_control` | 5 | Replay display 中的真人对照图片 |
| spoof | `history_head_model_attack` | 1356 | `炫彩闪烁活体/20230828 头模 正常光` 历史头模攻击图片 |
| spoof | `history_flash_collect_attack` | 4 | `炫彩闪烁活体/20230607_liveness_Detection` 历史补充攻击图片 |
| spoof | `screen_replay_attack_flash_archive` | 62790 | 20240320 归档屏幕重放攻击图片/帧图 |
| spoof | `print_attack_flash_archive` | 16610 | 20240320 归档打印攻击图片/帧图 |
| spoof | `silicone_mask_attack` | 1 | Axon Labs 硅胶面具样例图片 |
| spoof | `pretrain_chosen_photo_print_attack_archive` | 11445 | 20240320 预训练筛选打印照片攻击图片 |
| unknown | `unknown_review` | 3 | 未匹配到明确类别的图片，以及 raw/fake、raw/3dfake 下的圆形空白图 |

## 原始数据来源概览

当前 `dataset` 下的原始公开/新增资产:

| 数据目录 | 视频 | 图片 | 备注 |
| --- | ---: | ---: | --- |
| `Public samples` | 127 | 0 | 真人、打印裁剪、柱状纸、on actor、3D、PC/mobile replay |
| `3D_paper_mask ` | 36 | 0 | 3D 纸面头模攻击 |
| `Textile 3D Face Mask Attack Sample` | 23 | 0 | 纺织 3D 面具攻击 |
| `Silicone Mask - 21 Public samples` | 21 | 0 | 公开硅胶面具攻击 |
| `Advanced Paper attacks - iBeta 2` | 14 | 0 | 高级纸质攻击 |
| `ibeta 3 dataset sample` | 15 | 0 | 部分 Real/Mask 明确，部分需人工确认 |
| `Cutout_attacks` | 15 | 0 | Cutout 攻击 |
| `Silicone_mask` | 11 | 0 | 硅胶面具攻击 |
| `Latex_mask` | 10 | 0 | 乳胶面具攻击 |
| `Replay_mobile_attacks` | 10 | 0 | 手机重放攻击 |
| `Wrapped_3D_paper_mask` | 10 | 0 | 包覆式 3D 纸面攻击 |
| `Outside environment` | 7 | 0 | 室外真人 |
| `Replay_display_attacks` | 5 | 5 | 屏幕重放与真人对照 |
| `Extra shooting angles` | 5 | 0 | 真人额外角度 |
| `Selfies` | 0 | 25 | 真人自拍图片 |
| `Axon Labs Silicone mask sample.jpg` | 0 | 1 | 硅胶面具样例图片 |

20240320 归档资产:

| 数据目录 | 视频 | 图片 | txt | 备注 |
| --- | ---: | ---: | ---: | --- |
| `h5_raw` | 625 | 287945 | 625 | 真人/打印/屏幕攻击的 h5/raw 导出资产 |
| `raw` | 13044 | 11449 | 15308 | 真人、fake、3D fake、pre_train_choose 等原始炫彩活体数据 |

`炫彩闪烁活体` 历史资产:

| 数据目录 | 原始视频 | 原始图片 | txt | 去重后纳入归档 | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| `20230607_liveness_Detection` | 312 | 4 | 313 | 1 视频 + 4 图片 | 历史补充攻击采集；多数视频与 20240320 归档重复 |
| `20230828 头模 正常光` | 480 | 1356 | 484 | 375 视频 + 1356 图片 | 历史头模正常光攻击采集 |

说明:

- `dataset/flash_liveness_video_dataset_v1`、`dataset/flash_liveness_video_dataset_v2`、`dataset/flash_liveness_new_domain_video_protocol_v1v2` 是已整理/生成的数据集，未纳入本次类型归档，避免和原始资产重复统计。
- `tg_export_from_current_dataset*` 是 ThunderGuard 导出图片协议，也未纳入本次原始资产归档。
- zip 文件只作为压缩源保留，不计入视频/图片资产数量。
- 视频归档会用文件大小与首尾内容指纹去重；如果 `炫彩闪烁活体` 中的视频与 `20240320闪光活体归档/dataset/dataset` 重复，优先保留 20240320 归档版本。

## 使用建议

如果要查找某一类攻击样本，优先从归档目录进入:

```text
dataset/flash_liveness_asset_archive_by_type/videos/<attack_type>
dataset/flash_liveness_asset_archive_by_type/images/<attack_type>
```

如果要追溯原始文件位置，查看:

```text
dataset/flash_liveness_asset_archive_by_type/manifest.tsv
```

如果要更新归档，重新运行:

```bash
python3 scripts/organize_flash_liveness_assets_by_type.py
```

## 质量控制建议

- `unknown_review` 不建议直接用于监督训练，需要先人工确认标签；当前已无 unknown 视频，仅剩 3 张 unknown 图片。
- 视频 `unknown_review` 的来源目录会输出到 `dataset/flash_liveness_asset_archive_by_type/unknown_video_source_dirs.tsv`；当前该文件只有表头，表示没有待确认视频。
- iBeta3 的 `id_1`、`id_2` 目录已确认是戴面具、帽子和假头发的 fake 人脸视频，归为 `mask_fake_face_wig_hat_attack`，标签为 spoof。
- 20240320 `raw/remove` 已确认是真人对照试验视频，包含曝光明亮和正常光照两两成对样本，归为 `live_real_lighting_pair_control_archive`，标签为 live。
- `raw/fake` 和 `raw/3dfake` 下的 2 张圆形空白图片已从攻击类别移入 `images/unknown_review`；对应视频仍保留在攻击类别中。
- 20240320 `raw/pre_train_choose` 已确认是打印照片攻击图片，归为 `pretrain_chosen_photo_print_attack_archive`，标签为 spoof。
- 20240320 `raw/fake` 已归为 `flat_attack_flash_archive`，语义上覆盖平面、屏幕、打印等攻击；如需更细粒度类别，需要进一步解析文件名或原始采集标注。
- 软链接归档不会复制大文件。如果移动或删除原始数据，对应软链接会失效。
