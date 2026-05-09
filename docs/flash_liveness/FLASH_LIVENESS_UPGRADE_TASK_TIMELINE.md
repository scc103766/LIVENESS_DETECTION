# 炫彩活体检测升级任务时间表

更新时间：2026-05-06 14:22:14 +08:00

本文档用于记录当前炫彩活体检测项目已有版本、已有结果、当前问题、后续任务时间表和一版升级截止标准。当前升级目标以 `V3.1 -> V3.2-upgrade-v1` 为主，先完成一版可复现实验闭环，再决定是否继续做更大的结构重构。

## 一、当前结论

当前项目已经不再是单纯训练一个二分类模型，而是在解决以下几个关键问题：

1. 炫光协议下的真实皮肤响应和普通假体反光差异已经能被模型学习到。
2. 高级硅胶、软质面具、真实局部遮挡等攻击仍需要更稳的数据和训练策略。
3. 不能让模型只看局部真实区域，也不能让某一个辅助 loss 压过主分类目标。
4. 当前最重要的工程目标是先产出一版稳定、可复现、有日志、有评估结果的 `V3.2-upgrade-v1`。

本轮升级截止标准：

- 训练能完整跑完，不出现 NaN/Inf 或未定义变量中断。
- 有 `best_flash_liveness_model.pth`、`last_flash_liveness_model.pth`、`metrics_history.csv/jsonl`、`summary.json`。
- 在内部 test 上输出 accuracy/APCER/BPCER/ACER/AUC/EER。
- 在新域 280 条视频集上重新评估，并给出阈值选择建议。
- 对 silicone/mask/3D/head model/replay/flat attack 等类别做分组结果。
- 写出一版升级结论：是否优于 V3、是否可进入推理接口集成、剩余风险是什么。

## 二、当前已有版本与结果

| 版本 | 代码/模型 | 主要能力 | 当前结果 | 主要问题 |
|---|---|---|---|---|
| ThunderGuard 历史 ONNX | `20240320闪光活体归档/ThunderGuard/resources/MoEA_score.onnx` | 早期闪光活体模型，已有 ONNX 推理链路 | test_fake 模型单独评估：accuracy=0.6154，APCER=0.0，BPCER=1.0，ACER=0.5；与人脸比对融合后 accuracy=0.8462，ACER=0.2 | 对真人召回弱，模型老，难覆盖当前高级攻击 |
| V1 | `flash_liveness_project.py` | 固定 16 帧，CNN+Transformer 基础时序建模 | 新域 280 条 all：accuracy=0.2786，APCER=0.7710，BPCER=0.0，ACER=0.3855，AUC=0.7788；flashed 预处理后 accuracy=0.7250，ACER=0.4056，AUC=0.6250 | 固定帧策略弱，不能充分利用完整炫光时序 |
| V2 | `flash_liveness_project_v2.py` | 全帧读取，逐帧颜色 `.txt` 对齐，支持变长视频和 padding mask | 当前仓库缺少可用 V2 checkpoint，V2-on-V3 正式指标未产出 | checkpoint 缺失；物理特征主要是设计文档，还不是主线稳定结果 |
| V3 | `flash_liveness_project_v3.py` | CDC/伪深度/频域辅助监督/物理特征，固定协议训练 | 内部 test：accuracy=0.9986，APCER=0.0011，BPCER=0.0021，ACER=0.0016，AUC=0.99997；新域 280 条：accuracy=0.95，APCER=0.0076，BPCER=0.6667，ACER=0.3372，AUC=0.9559 | 内部指标强，但新域阈值下真人召回不足；外部泛化和阈值策略需要优化 |
| V3.1 fullframe/window | `flash_liveness_project_v3_1.py`，run=`flash_liveness_v3_1_fullframes_window_restore_original_gpu3` | 全帧窗口训练与推理，尝试更接近完整视频评估 | test：accuracy=0.9116，APCER=0.0831，BPCER=0.0988，ACER=0.0909，AUC=0.9654 | `test_loss/cls_loss` 出现 NaN，说明 loss 稳定性不足；整体不如 V3 内部最佳 |
| V3.1 best protocol 初始训练 | run=`flash_liveness_v3_1_best_protocol_nan_guard_20260429_stream_gpu5` | 使用 best protocol v1，加入 NaN 诊断 | 训练在 eval 阶段停止，诊断显示 physical 特征最大值约 468627，引发 logits/loss 非有限 | 物理特征尺度爆炸，需要稳定化 |
| V3.1 best protocol 修复版 | run=`v3_1_best_protocol_fix_batchidx_20260506_135716` | 物理特征 log 稳定化、loss 平衡、class-balanced sampler、`pos_weight=1.0`、NaN guard、修复 `batch_idx` | 当前正在训练：Epoch 1 已推进到 Batch 1620/8124，avg_loss=0.7136，weighted_aux 约 0.009~0.020，未再出现 `batch_idx` 错误 | 尚未完成 epoch，暂无最终 checkpoint 和 val/test 指标 |

## 三、当前数据状态

当前主训练数据集：

```text
dataset/flash_liveness_best_protocol_v1
materialized samples: 19143
skipped samples: 0
```

当前 split 统计：

| split | total | live | spoof | 说明 |
|---|---:|---:|---:|---|
| train | 16247 | 3872 | 12375 | hard spoof 只在 train 中重复采样 |
| val | 1459 | 480 | 979 | 不重复，用于阈值和调参 |
| test | 1437 | 503 | 934 | 不重复，用于最终评估 |

对当前升级最有帮助的类别：

- `silicone_mask_attack`
- `latex_mask_attack`
- `public_3d_attack`
- `three_d_head_model_attack_flash_archive`
- `three_d_paper_mask_attack`
- `textile_3d_mask_attack`
- `replay_display_attack`
- `replay_mobile_attack`
- `flat_attack_flash_archive`
- `live_real_flash_archive`
- `live_real_lighting_pair_control_archive`

当前数据短板：

- 硅胶样本仍然少，test/val 中 silicone 类别样本数很低，分组指标容易不稳定。
- 很多 hard spoof 没有真实炫光 `.txt`，只能用 neutral color protocol，不能完全学习真实闪光响应。
- 真实人脸新域样本少，新域阈值容易偏向拦截 spoof，导致 BPCER 高。

## 四、已经完成的 V3.1 修复

| 时间 | 内容 | 状态 |
|---|---|---|
| 2026-04-29 | 建立 best protocol v1 数据集，整合 flash archive、hard spoof、neutral protocol | 已完成 |
| 2026-04-29 | 启动 V3.1 best protocol NaN guard 训练 | 已完成，训练被 NaN/Inf 守护中断 |
| 2026-04-29 | 定位 NaN 原因：physical 特征尺度爆炸，最大值约 468627 | 已完成 |
| 2026-05-06 | 增加 physical 特征稳定化：finite 替换、`sign(x)*log1p(abs(x))`、clip 到 `[-6,6]` | 已完成 |
| 2026-05-06 | 移除上下翻转增强，改为左右 yaw perspective 轻量增强 | 已完成 |
| 2026-05-06 | 增加 loss 平衡：单项辅助 loss 和总辅助 loss cap | 已完成 |
| 2026-05-06 | 增加 `quality_lower_trimmed_mean` 窗口融合，避免局部高 live 窗口掩盖 spoof 风险 | 已完成 |
| 2026-05-06 | 增加 class-balanced sampler，并在开启时将 `pos_weight=auto` 转为 `1.0` | 已完成 |
| 2026-05-06 | 修复 `evaluate_model()` 中 `batch_idx` 未定义问题 | 已完成 |
| 2026-05-06 | 重新启动修复版训练，session=`v31_fix_batchidx_gpu6` | 进行中 |

## 五、后续任务时间表

| 日期 | 阶段 | 要完成的事情 | 产出物 | 验收标准 |
|---|---|---|---|---|
| 2026-05-06 | 训练稳定性恢复 | 继续观察当前 `v3_1_best_protocol_fix_batchidx_20260506_135716` 训练，确认第 1 个 epoch 能完成 train/val/test | `train.log`、`metrics_history.jsonl/csv` 初始记录 | 不再出现 `NameError`、NaN/Inf；第 1 个 epoch 有 val/test 指标 |
| 2026-05-07 | 完成 V3.1 修复版主训练 | 跑完 30 epoch，或至少跑到 val AUC/ACER 稳定且无继续提升迹象 | `best_flash_liveness_model.pth`、`last_flash_liveness_model.pth`、`summary.json` | 内部 test AUC 不低于 0.96，ACER 尽量低于 0.10；若达不到，需要记录失败原因 |
| 2026-05-08 | 内部测试复盘 | 汇总 train/val/test 曲线，检查 loss 平衡、class-balanced sampler 后 batch loss 是否更稳定 | `V3_1_BALANCED_TRAINING_REVIEW.md` | 说明是否存在过拟合、是否某个辅助 loss 主导、是否类别分布导致阈值偏移 |
| 2026-05-09 | 新域 280 条视频评估 | 使用 V3.1 best checkpoint 在新域数据上跑 window 推理和 full-sequence 对比 | 新域 `summary.json`、分组 csv/jsonl | 必须给出 accuracy/APCER/BPCER/ACER/AUC/EER；重点看 BPCER 是否低于 V3 的 0.6667 |
| 2026-05-10 | hard spoof 分组评估 | 对 silicone/latex/3D/head model/replay/flat attack 分组统计 | category metrics 表 | silicone/mask 类必须单独列出；样本过少时标注“不足以形成稳定结论” |
| 2026-05-11 | 阈值策略选择 | 对内部 test、新域数据分别扫描阈值，给出安全优先阈值和均衡阈值 | `threshold_report.md` | 至少给出两套阈值：安全优先低 APCER、均衡低 ACER |
| 2026-05-12 | 推理链路整理 | 固定 V3.1/V3.2 推理命令、输入格式、window 参数、fusion 策略 | `FLASH_LIVENESS_V3_2_INFERENCE.md` | 单视频推理和批量评估命令都能复现 |
| 2026-05-13 | 数据补强方案冻结 | 明确哪些外部数据只做预训练/hard spoof 辅助，哪些需要重采炫光协议 | 数据补强清单 | WMCA/CASIA/3DMAD 等不能直接替代炫光数据，只能作为 hard spoof 预训练或对比 |
| 2026-05-14 | V3.2-upgrade-v1 代码冻结 | 只保留必要修复，不再追加大结构变更 | 代码、脚本、README 更新 | `py_compile`、脚本 `bash -n`、一次 smoke eval 全部通过 |
| 2026-05-15 | 结果归档 | 将 V1/V2/V3/V3.1/V3.2-upgrade-v1 指标写成统一对比表 | `FLASH_LIVENESS_VERSION_RESULT_COMPARISON.md` | 每个版本都有 checkpoint、数据集、阈值、主要指标、问题 |
| 2026-05-16 | 升级版验收 | 判断是否达到“一版升级”截止标准 | `FLASH_LIVENESS_V3_2_UPGRADE_ACCEPTANCE.md` | 达标则结束本轮升级；未达标则只允许补 bug 和阈值，不再扩大目标 |
| 2026-05-17 | 预留缓冲 | 处理训练中断、显存占用、坏样本、日志补充 | 修正后的最终归档 | 最晚 2026-05-17 结束本轮升级 |

## 六、V3.2-upgrade-v1 的目标指标

目标分为三个层级：

| 层级 | 指标目标 | 说明 |
|---|---|---|
| 必须达到 | 训练完整结束，无 NaN/Inf；内部 test AUC >= 0.96；输出完整 checkpoint 和 summary | 这是工程可用底线 |
| 期望达到 | 内部 test ACER <= 0.08；新域 BPCER 明显低于 V3 的 0.6667 | 说明对真人召回和泛化有改善 |
| 优秀目标 | 新域 ACER < 0.25，且 hard spoof APCER 不明显恶化 | 说明升级不仅改善真人召回，也没有牺牲攻击拦截 |

不建议只用 accuracy 判断升级成败。当前新域数据 live/spoof 很不平衡，accuracy 容易被 spoof 数量主导。后续汇报必须同时看：

- APCER：攻击通过率，越低越安全。
- BPCER：真人误拒率，越低体验越好。
- ACER：安全和体验的平均错误。
- AUC/EER：阈值无关的整体可分性。
- category metrics：尤其是 silicone、latex、3D mask、head model、replay。

## 七、当前训练观察

当前运行：

```text
tmux session: v31_fix_batchidx_gpu6
run: flash_liveness_runs/v3_1_best_protocol_fix_batchidx_20260506_135716
GPU: 6
```

截至 2026-05-06 14:22:14：

```text
Epoch [1/30] Batch [1620/8124]
processed_items=3240/16247
batch_loss=0.7065
avg_loss=0.7136
cls=0.6977
depth=0.1526
contrast=0.0092
fft=0.0048
weighted_aux=0.0088
```

当前观察：

- `avg_loss` 从约 0.756 降到约 0.714，较之前未均衡采样时更平稳。
- `weighted_aux` 保持在较小范围，辅助 loss 没有压过分类 loss。
- `pos_weight=1.0` 后，batch loss 不再反复出现由类别权重造成的 1.4/2.1 型跳变。
- 当前还未完成第 1 个 epoch，不能下最终结论。

查看日志：

```bash
tail -f /supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/v3_1_best_protocol_fix_batchidx_20260506_135716/train.log
```

## 八、风险与处理原则

| 风险 | 表现 | 处理原则 |
|---|---|---|
| loss 再次 NaN/Inf | 训练中断并输出诊断 JSON | 先定位输入张量、physical、logits；必要时关闭 AMP 或进一步裁剪 physical |
| 新域真人召回仍差 | BPCER 高，live 大量被拒 | 不直接降低全局安全阈值；先做阈值双方案和 live domain 数据补强 |
| silicone 样本太少 | category 指标波动极大 | 不夸大结论；只作为风险提示，同时规划采集真实炫光 silicone 数据 |
| 模型只关注局部真实区域 | 局部窗口 live 概率掩盖 spoof | 继续使用 `quality_lower_trimmed_mean`，必要时增加低分位/最小窗口策略对比 |
| 辅助 loss 主导 | depth/fft/contrast 指标大幅影响 total loss | 保留 aux cap；所有结果同时记录 raw aux 和 weighted aux |
| fullframe 滑窗误差 | 解码失败、坏窗口、局部高分 | 记录 window quality；用稳健融合，不直接平均所有窗口 |

## 九、本轮截止后的决策

到 2026-05-16 或最晚 2026-05-17，本轮升级只做下面三种结论之一：

1. 通过：`V3.2-upgrade-v1` 指标和稳定性均优于当前 V3/V3.1，可以进入推理接口集成。
2. 条件通过：内部 test 达标，新域仍有短板，允许用于受控场景，但需要继续采集新域 live 和 silicone 炫光数据。
3. 不通过：训练稳定但指标不优，保留当前 V3 为主版本，V3.1/V3.2 作为实验分支继续优化。

本轮不继续无边界扩展模型结构。完成一版升级后，下一轮再单独规划：

- 真实 silicone 炫光采集协议。
- WMCA/CASIA/3DMAD 等外部数据的 hard spoof 预训练策略。
- 多模态 depth/IR/thermal 与炫光 RGB 的迁移方式。
- 端侧/服务端推理速度优化。
