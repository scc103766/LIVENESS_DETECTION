# Flash Liveness Project V3_1 说明

V3_1 保存于 `flash_liveness_project_v3_1.py`，目标是在不丢弃长视频后半段信息的前提下避免显存爆炸。

当前这一版 V3_1 训练默认配套“fixed collect protocol restore-original” 数据组织：

```text
neutral/original -> flash color -> neutral/original -> next flash color
```

也就是说，固定顺序协议不再把非闪光阶段描述为“黑屏”，而是视作“保持原视频原色/无额外闪光”。对离线协议生成和缺省 txt 补齐来说，`color=0` 表示原色视频段；对真正的 fullscreen 采集脚本而言，该阶段仍只能用暗屏近似显示，但训练标签语义已经统一为“无额外闪光”。

核心方案：

```text
完整读取视频全部帧到 CPU
按 window_size / window_stride 切成滑动窗口
每次只把一个窗口送入 GPU
窗口概率通过质量稳健融合得到视频级概率
训练时每个窗口继承视频标签并累计梯度
```

默认窗口：

```text
window_size=256
window_stride=128
eval_window_size=256
eval_window_stride=128
window_fusion=quality_lower_trimmed_mean
window_trim_ratio=0.2
```

`quality_lower_trimmed_mean` 会先估计每个窗口的质量。剧烈运动、异常 diff、突发亮度波动的窗口会降低权重；同时默认丢弃最高 live 分的 20% 窗口，再做质量加权平均。这样更偏安全：高级面具或局部伪造只要在某些窗口暴露低 live 证据，就不会被局部真实特征或高 live 窗口轻易平均掉。

loss 默认加入动态上限：

```text
aux_loss_max_ratio=0.35
aux_loss_total_max_ratio=0.70
```

也就是任一辅助 loss 和全部辅助 loss 的加权贡献都不会无限压过分类主损失，避免模型只盯着 pseudo-depth、FFT 或某一类物理 cue。训练增强不做上下翻转，也不做简单左右镜像；默认只在训练集对整段人脸序列施加轻量左右 yaw 透视增强，验证/测试保持原始正脸分布。

当前 best training 脚本还启用了类均衡采样。原数据中 spoof 明显多于 live，若继续用小 batch 随机采样和 `pos_weight=auto`，batch loss 会随类别组成剧烈跳动；类均衡采样启用后，`pos_weight=auto` 会自动退化为 `1.0`，避免“过采样 live + BCE 正类加权”的双重补偿。

## 固定颜色协议调整

V3_1 训练时，固定协议的 collect-flash 时间线建议使用：

```text
warmup(original):  1.0s
flash hold:        0.35s
restore(original): 0.15s
tail(original):    0.5s
```

这里的重点不是插入黑色帧，而是：

```text
保持原色视频
-> 打一段固定颜色闪光
-> 恢复原色视频
-> 再进入下一种闪光颜色
```

这样更贴近“无额外闪光”和“有闪光刺激”之间的真实切换，也能减轻模型把黑屏边界学成伪特征的风险。

推荐先用下列脚本重建一版派生数据集：

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python scripts/build_flash_liveness_fixed_protocol_dataset.py \
  --output-root dataset/flash_liveness_asset_archive_fixed_collect_protocol_restore_original \
  --warmup-seconds 1.0 \
  --hold-seconds 0.35 \
  --restore-seconds 0.15 \
  --tail-seconds 0.5 \
  --overwrite
```

然后再启动 V3_1 训练。

## 物理线索优先级

V3_1 补充物理线索时，优先接入当前分析里最稳定的那几类特征，而不是一次性把所有 cue 都堆进去。

第一优先级：

```text
freq_high_energy_mean
freq_mid_energy_mean
freq_lap_var_mean
freq_lap_var_std
freq_row_periodicity_mean
```

这些频域/纹理特征在 val/test 上最稳定，优先用于拦截 replay、打印、头模、面具等物理呈现攻击。

第二优先级：

```text
flash_stable_delta_abs_mean
flash_transition_delta_abs_mean
flash_color_intensity_range
flash_color_green_range
```

这些特征和炫彩协议本身最相关，用来描述“同样的打光下，人脸响应是否稳定、是否像真人皮肤”。

第三优先级：

```text
rppg_delta_g_abs_mean
rppg_std_g
rppg_forehead_std_g
rppg_cheek_std_g
```

rPPG 先作为辅助项，不建议在第一轮把它当主判据。

启动训练：

```bash
bash scripts/run_flash_liveness_v3_1_window_training.sh
```

默认只使用本项目允许的物理 0/1/2 卡：

```text
GPU_IDS=0,1,2
DEVICE=cuda:0
BATCH_SIZE=2
MAX_TRAIN_FRAMES=0
MAX_EVAL_FRAMES=0
FLASH_RESTORE_SECONDS=0.15
```

项目 GPU 约定：后续运行只使用物理 `0/1/2`，`3/4/5/6` 不作为本项目运行卡。

注意：这里 `MAX_*_FRAMES=0` 表示完整读取视频，不再截断前 256 帧。显存由窗口长度控制，而不是由整段视频长度控制。

默认训练脚本已切到：

```text
dataset/flash_liveness_asset_archive_fixed_collect_protocol_restore_original
flash_liveness_runs/flash_liveness_v3_1_fullframes_window_restore_original_gpu012
```

并且把 batch size 提到 `2`，作为当前“尽量快但仍偏保守”的默认值。如果物理 `0/1/2` 中当前可用显存不足，则回退到 `1`。

单视频推理：

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python flash_liveness_project_v3_1.py infer \
  --checkpoint flash_liveness_runs/flash_liveness_v3_1_fullframes_window_restore_original_gpu012/best_flash_liveness_model.pth \
  --video-path sample.mp4 \
  --txt-path sample.txt \
  --window-size 256 \
  --window-stride 128 \
  --window-fusion quality_lower_trimmed_mean \
  --aux-loss-max-ratio 0.35 \
  --aux-loss-total-max-ratio 0.70 \
  --device cuda:0
```

输出会包含 `num_frames`、`num_windows`、`mean_window_quality` 和 `min_window_quality`，便于确认完整视频被窗口覆盖，以及窗口质量是否异常。
