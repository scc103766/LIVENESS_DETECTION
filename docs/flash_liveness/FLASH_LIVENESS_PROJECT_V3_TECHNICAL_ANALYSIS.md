# Flash Liveness V3 完整处理链路技术分析

更新时间：2026-05-09

本文面向 `flash_liveness_project_v3.py` 的当前实现，逐步拆解 V3 从数据发现、帧读取、颜色协议、显式物理特征、FFT 辅助监督、模型分支、token fusion、Transformer 时序建模到最终训练/推理输出的完整链路。重点回答每一步：

- 输入是什么，输出是什么，tensor 形状如何变化。
- 使用了什么技术或工程方法。
- 为什么希望发生这样的变化。
- 相关论文、开源项目或本项目代码来源是什么。

配套图：

- [V3 网络结构图](../../assets/flash_liveness_v3_architecture.svg)
- [V3 输入输出变化图](../../assets/flash_liveness_v3_io_tensor_flow.svg)

核心代码：

- [flash_liveness_project_v3.py](../../flash_liveness_project_v3.py)
- [flash_physical_features.py](../../flash_physical_features.py)
- [Face_detection_yolo_align.py](../../Face_detection_yolo_align.py)

## 1. V3 的设计定位

V3 不是普通静默 RGB 活体模型，也不是只做颜色闪烁分类的规则系统。它的目标是把“已知炫彩光刺激”和“人脸在光刺激下的视觉/物理响应”合成一个可训练的多模态时序模型。

V2 已有能力：

- 顺序读取视频帧。
- 构造 RGB + frame diff 的 6 通道视觉输入。
- 读取逐帧颜色 txt，形成 `[r, g, b, transition]` 颜色 token。
- 使用 CNN + Transformer 做时序二分类。

V3 在 V2 上新增：

- `manifest.tsv` 数据发现与 `label + media_type + category` 分层切分。
- 图片样本兼容。
- 缺少颜色 txt 时可使用 `collect_flash` 固定协议补齐颜色标签。
- `flash_physical_features.py` 的 frequency / flash-response / rPPG 显式物理 token。
- CDC texture branch，用 Central Difference Convolution 强化局部纹理/材质差异。
- pseudo-depth map 和 depth/contrast depth 辅助损失。
- FFT target 和 FFT head，借鉴 Fourier auxiliary supervision 思路约束 texture branch。
- 输出从单一 `logits` 扩展为 `logits + depth_maps + fft_maps`。
- Loss 从单一 BCE 扩展为 `BCE + depth + contrast depth + FFT`。

## 2. 总体数据流

```text
manifest.tsv / live|spoof 目录
  -> LivenessSample(media_path, txt_path, label, media_type, category, source_group)
  -> 分层 train/val/test split
  -> 视频逐帧读取或图片读取
  -> 逐帧颜色 txt 读取，缺失时生成 collect_flash 协议
  -> 人脸检测/对齐，失败则中心裁剪
  -> BGR -> RGB, resize 224x224, normalize [0,1]
  -> frame diff
  -> concat RGB + diff 得到 6 通道视频 tensor
  -> 颜色整数编码为 [r,g,b,transition]
  -> frequency / flash response / rPPG physical token
  -> FFT target [T,1,32,32]
  -> batch 内 padding 到 max_len，生成 padding_mask
  -> ResNet18 frame embedding
  -> CDC texture embedding + pseudo-depth map
  -> color projection
  -> physical projection
  -> concat token fusion
  -> positional encoding + Transformer Encoder
  -> masked temporal pooling
  -> classifier logits
  -> BCE + depth + contrast depth + FFT joint training
```

## 3. 逐步骤技术分析

### 3.1 数据发现与样本元信息

对应代码：

- `LivenessSample`
- `collect_samples_from_manifest`
- `collect_samples_from_label_dirs`
- `stratified_split`
- `discover_dataset_splits`

输入来源：

```text
data_root/
  manifest.tsv
  videos/...
  images/...
```

或旧格式：

```text
data_root/
  train/live
  train/spoof
  val/live
  val/spoof
  test/live
  test/spoof
```

样本被封装为：

```text
LivenessSample(
  media_path: str,
  txt_path: str | None,
  label: int,          # live=1, spoof=0
  media_type: str,     # videos/images
  category: str,       # 攻击类型或真实类型
  source_group: str,
)
```

使用的技术：

- manifest 驱动的数据索引。
- 元信息保留。
- 按 `(label, media_type, category)` 分层切分。

为什么这样做：

- 活体检测不是单纯二分类，安全性取决于各类攻击是否覆盖。只按 `live/spoof` 随机切分，可能让某些攻击类型只出现在 train 或只出现在 test。
- `category` 保证纸张、屏幕回放、3D 头模、硅胶、乳胶、真实人脸等类别尽量进入 train/val/test。
- `source_group` 保留来源信息，后续可以检查设备、采集批次或来源域偏差。

tensor 变化：

这一阶段还没有 tensor，输出是样本列表：

```text
samples: list[LivenessSample]
splits:
  train: list[LivenessSample]
  val:   list[LivenessSample]
  test:  list[LivenessSample]
```

希望获得的变化：

- 从无结构文件树变成带标签、类型、攻击类别、来源分组的结构化样本。
- 为后续可解释评估、category coverage 和新域测试打基础。

### 3.2 颜色协议读取与缺失补齐

对应代码：

- `parse_color_txt`
- `color_int_to_feature`
- `FlashLivenessDataset._build_missing_video_color_map`
- `build_frame_color_labels` from `scripts.collect_flash_liveness_video`

输入：

```text
同名 txt:
  frame_idx,color_int

或 tg_export 简短格式:
  color_int
```

输出：

```text
color_map: dict[int, int]
color_features per frame: [r, g, b, transition]
color_tensor: [T, 4]
```

`color_int_to_feature` 的含义：

```text
r = ((color >> 16) & 0xFF) / 255
g = ((color >>  8) & 0xFF) / 255
b = ( color        & 0xFF) / 255
transition = 1 if 当前帧颜色 != 前一帧颜色 else 0
```

使用的技术：

- 逐帧条件编码。
- 颜色刺激协议作为 side-channel token。
- 缺失 txt 时使用固定 `collect_flash` 协议合成标签。

为什么这样做：

- 炫彩活体依赖“当前帧是什么光色”和“人脸对该光色的响应”。没有颜色 token，模型只能看到亮度/颜色变化，但不知道变化是否来自协议、环境光、屏幕回放、压缩噪声或采集设备。
- `transition` 显式标记切光瞬间，使模型知道某些帧处在光色变化边界，便于 Transformer 学习稳定阶段和过渡阶段的不同响应。
- 缺失 txt 时使用固定协议补齐，是为了让历史视频仍可进入训练/评测；但这要求视频确实来自相同采集协议，否则会引入错误条件。

tensor 变化：

```text
txt / protocol
  -> color_map: frame_idx -> color_int
  -> color_features: [T, 4]
  -> color_tensor: torch.float32 [T, 4]
```

希望获得的变化：

- 从不可学习的外部采集条件，变成逐帧可学习的条件 embedding。
- 让后续 `frame_emb + color_emb` 表示“在当前光色条件下看到的人脸响应”。

### 3.3 视频/图片读取

对应代码：

- `_read_all_frames_with_color`
- `_read_image_with_color`
- `_sample_frames`

输入：

```text
video_path: .mp4/.avi/.mov/.mkv/...
image_path: .jpg/.png/.bmp/.webp/...
txt_path: optional
```

输出：

```text
frames_bgr:     list[np.ndarray]  # 每帧 BGR
color_features: list[np.ndarray]  # 每帧 [r,g,b,transition]
color_values:   list[int]         # 每帧 packed RGB int
```

使用的技术：

- OpenCV sequential decode。
- `frame_stride` 控制跳帧。
- `max_frames` 控制最长序列。
- 单张图片按 `T=1` 兼容。

为什么这样做：

- 顺序读取保留完整时序，比固定抽帧更适合学习切光前后响应、迟滞、稳定阶段和短期扰动。
- `frame_stride` 与 `max_frames` 给训练资源留出口：全帧训练更完整，但内存/耗时更高。
- 图片兼容能利用静态样本，但图片没有真实时序，主要补充材质和静态纹理，不应替代炫彩视频主线。

tensor 变化：

这一阶段仍是 Python list：

```text
frames_bgr: T x H_raw x W_raw x 3
color_features: T x 4
color_values: T
```

希望获得的变化：

- 把原始媒体统一成“有序帧 + 对齐颜色条件”的样本序列。

### 3.4 人脸检测/对齐与兜底裁剪

对应代码：

- `FacePreprocessor`
- `YOLOv7_face_mkl` from [Face_detection_yolo_align.py](../../Face_detection_yolo_align.py)

输入：

```text
frames_bgr: list[np.ndarray]
target_size: (224, 224)
```

输出：

```text
processed_frames: list[np.ndarray]
每帧 shape: [224, 224, 3] BGR
```

使用的技术：

- 可选 YOLOv7-face 检测与关键点对齐。
- 多脸时选择 confidence 最大的人脸。
- 对齐失败或未配置 detector 时使用中心裁剪 + resize。

相关项目/论文：

- YOLOv7 原论文：Wang, Bochkovskiy, Liao, [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)。
- YOLOv7-face 项目：[derronqi/yolov7-face](https://github.com/derronqi/yolov7-face)，其 README 标注了 face detection with landmark，并列出 WIDERFace 测试结果和 YOLOv7 参考。

为什么这样做：

- 活体线索主要在脸部区域。背景、衣服、屏幕边框、手持设备等容易成为捷径特征。
- 对齐或裁剪到固定尺寸后，CNN 分支能学习更稳定的人脸纹理、反光、颜色响应和局部差分。
- fallback 中心裁剪保证数据加载不中断，避免 detector 单点失败导致训练样本大面积丢失。

tensor 变化：

```text
frames_bgr: [T, H_raw, W_raw, 3]
  -> processed_frames: [T, 224, 224, 3]
```

希望获得的变化：

- 空间尺度统一。
- 人脸位置尽量稳定。
- 降低非人脸区域对分类的干扰。

### 3.5 RGB 归一化与 RGB+Diff 6 通道构造

对应代码：

- `FlashLivenessDataset.process_sample`

输入：

```text
processed_frames: [T, 224, 224, 3] BGR uint8
```

处理：

```text
BGR -> RGB
uint8 -> float32 / 255.0
frames: [T, 224, 224, 3]

diff_frames[1:] = frames[1:] - frames[:-1]
diff_frames[0] = diff_frames[1]  # 当 T > 1

multi_modal_frames = concat([frames, diff_frames], axis=-1)
```

输出：

```text
multi_modal_frames: [T, 224, 224, 6]
tensor_frames:      [T, 6, 224, 224]
```

使用的技术：

- RGB 归一化。
- 相邻帧差分，作为短时局部时间导数。
- channel-wise concat。

为什么这样做：

- RGB 保留人脸静态外观、肤色、材质和空间纹理。
- Diff 直接给出短期变化：切光后的亮度变化、屏幕刷新/反射变化、抖动、迟滞。
- 把 RGB 与 diff 放在同一像素位置的 6 维通道中，CNN 可以在局部卷积里同时看到“当前外观”和“刚发生的变化”。
- Diff 让 Transformer 不必完全从多帧 RGB 中自己推断短期差异，可以把容量更多用于长时序关系。

tensor 变化：

```text
frames:      [T, 224, 224, 3]
diff_frames: [T, 224, 224, 3]
concat:      [T, 224, 224, 6]
permute:     [T, 6, 224, 224]
```

希望获得的变化：

- 从纯空间输入变成“空间 + 短时动态”输入。
- 为 ResNet18 和 CDC texture branch 提供更强的局部动态信号。

### 3.6 显式物理特征 PhysicalCueExtractor

对应代码：

- [flash_physical_features.py](../../flash_physical_features.py)
- `FrequencyArtifactExtractor`
- `FlashResponseFeatureExtractor`
- `RPPGFeatureExtractor`
- `PhysicalCueExtractor.per_frame`

输入：

```text
frames:       [T, 224, 224, 3] RGB float32 in [0,1]
color_values: [T]
```

输出：

```text
physical_np:     [T, P]
tensor_physical: [T, P]
```

当前默认：

```text
P = 5 frequency features
  + 8 flash response features
  + 6 rPPG features
  = 19
```

#### 3.6.1 Frequency / Texture Features

每帧 5 维：

```text
freq_high_energy
freq_mid_energy
freq_lap_var
freq_row_periodicity
freq_col_periodicity
```

技术：

- 灰度图 2D FFT。
- 高频/中频能量比例。
- Laplacian 方差。
- 行/列均值 profile 的相邻差分标准差。

理由：

- 屏幕回放可能出现刷新纹、摩尔纹、像素栅格和压缩频域异常。
- 打印/纸张/面具可能出现过平滑、异常锐化或材质纹理不同。
- CNN 可以自己学这些线索，但显式 token 能把频域/纹理统计直接交给时序模型。

#### 3.6.2 Flash Response Features

每帧 8 维：

```text
flash_r_mean
flash_g_mean
flash_b_mean
flash_intensity
flash_delta_intensity
flash_transition
flash_response_decay
flash_chroma_ratio
```

技术：

- 每帧 RGB 均值。
- 强度与强度差分。
- 与颜色标签对齐的 transition 标记。
- 简单迟滞/衰减代理。
- 色度比例。

理由：

- `color_tensor` 表示“发出了什么光”；flash response 表示“脸实际怎么响应”。
- 真人皮肤、屏幕、纸张、硅胶、头模对同一颜色刺激的反射与衰减不同。
- 这些统计特征能帮助模型关注光学响应，而不是只看静态纹理。

#### 3.6.3 rPPG-like Green-channel Statistics

每帧 6 维：

```text
rppg_mean_g
rppg_delta_g
rppg_cheek_g
rppg_forehead_g
rppg_local_energy
rppg_fft_peak_score
```

技术：

- 绿色通道均值。
- 绿色通道差分。
- 上半脸/下半脸区域统计。
- 局部时间窗口能量。
- 简单频域峰值分数。

相关论文/项目：

- rPPG 背景参考：de Haan and Jeanne, [Robust pulse-rate from chrominance-based rPPG](https://research.tue.nl/en/publications/robust-pulse-rate-from-chrominance-based-rppg/)。
- 视觉微弱生理信号背景参考：Wu et al., [Eulerian Video Magnification for Revealing Subtle Changes in the World](https://people.csail.mit.edu/mrub/evm/)。

注意：

- 当前实现不是完整心率估计算法，也没有做严格频带滤波、CHROM/POS 解混或心率回归。
- 它只是把绿色通道时序稳定性作为辅助物理线索。

理由：

- 真人脸部可能存在微弱血液/皮肤响应，而屏幕/纸张/头模通常没有真实生理响应。
- 在短视频、强闪光和运动场景下，rPPG 不稳定，所以 V3 只把它作为 19 维 physical token 的一部分，不把它作为单独判定依据。

tensor 变化：

```text
frames [T,224,224,3] + color_values [T]
  -> frequency [T,5]
  -> flash_response [T,8]
  -> rppg [T,6]
  -> physical_tensor [T,19]
```

希望获得的变化：

- 从端到端 CNN 隐式学习，变成“隐式视觉特征 + 显式物理统计”的互补建模。
- 提高对未知材质、未知屏幕、未知采集设备的泛化可能。

### 3.7 FFT 辅助监督 Target

对应代码：

- `_fft_target_from_frames`

输入：

```text
frames: [T, 224, 224, 3] RGB float32
```

处理：

```text
RGB -> grayscale
resize -> [32,32]
2D FFT
fftshift
log1p(abs(spectrum))
min-max normalize
```

输出：

```text
fft_tensor: [T, 1, 32, 32]
```

使用的技术：

- 2D Fourier spectrum target。
- 训练期辅助监督。

相关项目：

- MiniVision [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) 项目说明中使用傅里叶频谱图辅助监督，并开源 MiniFASNetV1/V2 模型。V3 借鉴的是“频域辅助监督”思想，不是直接复刻其网络。

为什么这样做：

- 只用 live/spoof BCE，texture branch 可能学到背景、设备、压缩、采集来源等捷径。
- FFT target 强迫 texture branch 的 embedding 通过 `fft_head` 预测每帧频谱结构，从而把局部纹理与频域伪影纳入训练约束。
- 推理时不需要外部 FFT 文件；FFT target 只在训练/评测 loss 中在线生成。

tensor 变化：

```text
frames [T,224,224,3]
  -> fft_targets [T,1,32,32]
```

希望获得的变化：

- 给 texture branch 一个比二分类标签更密集的 per-frame 监督。
- 让模型在材质和频域层面学习攻击差异。

### 3.8 Dataset 输出

`FlashLivenessDataset.__getitem__` 返回：

```text
tensor_frames:   [T, 6, 224, 224]
tensor_colors:   [T, 4]
tensor_physical: [T, P]
tensor_fft:      [T, 1, 32, 32]
label:           scalar float32
```

如果 `transform=True`，会随机水平翻转：

```text
tensor_frames = flip(width)
tensor_fft    = flip(width)
```

注意：

- color/physical 不做水平翻转，因为它们是逐帧条件或统计，不依赖左右像素坐标。
- 当前增强较保守，避免破坏颜色协议和时序关系。

## 4. Batch Collate 与 Padding

对应代码：

- `collate_skip_none`

输入：

```text
batch = [
  ([T1,6,224,224], [T1,4], [T1,P], [T1,1,32,32], label1),
  ([T2,6,224,224], [T2,4], [T2,P], [T2,1,32,32], label2),
  ...
]
```

输出：

```text
videos:       [B, max_len, 6, 224, 224]
colors:       [B, max_len, 4]
physical:     [B, max_len, P]
fft_targets:  [B, max_len, 1, 32, 32]
padding_mask: [B, max_len]  # True 表示 padding
labels:       [B]
```

使用的技术：

- batch 内动态 padding。
- padding mask。

为什么这样做：

- 视频长度天然不同，强行截断会丢失时序；强行固定读取会浪费大量算力。
- Transformer 支持 `src_key_padding_mask`，可以让 padding 帧不参与注意力。
- loss 也使用 `valid_mask = ~padding_mask`，只统计有效帧。

tensor 变化：

```text
不同 T 的样本
  -> max_len = max(T_i)
  -> 不足 max_len 的位置填 0
  -> padding_mask 标记无效帧
```

希望获得的变化：

- 保留变长时序信息。
- 训练 batch 化。
- 保证 Transformer 和辅助 loss 不被 padding 污染。

## 5. 模型结构分析

模型类：

- `CNNTransformerLiveness`

默认配置：

```text
embed_dim = 512
num_heads = 8
num_layers = 2
physical_dim = 19
cdc_theta = 0.7
lambda_depth = 0.1
lambda_contrast = 0.1
lambda_fft = 0.05
```

### 5.1 ResNet18 主视觉分支

对应代码：

- `models.resnet18`
- `self.cnn_backbone`
- `self.frame_proj`

输入：

```text
videos: [B, T, 6, 224, 224]
reshape -> [B*T, 6, 224, 224]
```

结构：

```text
ResNet18 without final FC
conv1: Conv2d(6 -> 64, kernel=7, stride=2, padding=3)
Adaptive pooling from ResNet final block
frame_proj: Identity if E=512 else Linear(512 -> E)
```

预训练处理：

```text
original RGB conv1 weight -> first 3 channels
same RGB conv1 weight     -> diff 3 channels
freeze cnn_backbone[0:5]
```

输出：

```text
frame_emb: [B, T, E]
```

使用的技术：

- ResNet18 残差网络。
- ImageNet 预训练权重迁移到 6 通道输入。
- 早期层冻结。

相关论文：

- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)。

为什么这样做：

- ResNet18 足够轻，适合视频逐帧编码。
- 预训练权重提供低层边缘、纹理和中层视觉表征，减少小数据训练不稳定。
- 把 RGB 权重复制到 diff 通道，是一种工程初始化：让 diff 通道从类似 RGB 的边缘/纹理滤波器起步，而不是完全随机。
- 冻结早期层降低过拟合和显存/梯度开销。

### 5.2 CDC Texture Branch

对应代码：

- `Conv2dCD`
- `CDCTextureBranch`

输入：

```text
x_cnn: [B*T, 6, 224, 224]
```

`Conv2dCD`：

```text
out = normal_conv(x) - theta * center_difference_conv(x)
```

当前结构：

```text
Conv2dCD(6 -> 64, stride=2) + BN + ReLU
Conv2dCD(64 -> 128, stride=2) + BN + ReLU
Conv2dCD(128 -> E, stride=2) + BN + ReLU
```

输出：

```text
feature_map
  -> AdaptiveAvgPool2d(1,1)
  -> texture_emb: [B*T, E] -> [B, T, E]

feature_map
  -> depth_head
  -> interpolate 32x32
  -> depth_maps: [B*T,1,32,32] -> [B,T,1,32,32]
```

使用的技术：

- Central Difference Convolution。
- texture branch 与主视觉分支并联。
- pseudo-depth head。

相关论文：

- Yu et al., [Searching Central Difference Convolutional Networks for Face Anti-Spoofing](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.html)。

为什么这样做：

- 反欺骗任务中，局部纹理、边缘、材质和细粒度差异常常比全局语义更关键。
- CDC 强化中心差分，对纸张纹理、屏幕栅格、硅胶/乳胶材质、头模表面更敏感。
- 主 ResNet 分支关注稳定语义，CDC branch 补充局部微分纹理。
- pseudo-depth 不是精确深度估计，而是弱结构先验，给 CDC 分支额外约束。

### 5.3 Color Projection

对应代码：

- `self.color_proj`

输入：

```text
colors: [B, T, 4]
```

结构：

```text
Linear(4 -> E)
LayerNorm(E)
ReLU
```

输出：

```text
color_emb: [B, T, E]
```

为什么这样做：

- 把低维颜色协议映射到和视觉 token 相同的 embedding 维度。
- 后续 `frame_emb + color_emb` 让每帧视觉特征带上“当前光色条件”。
- LayerNorm 让颜色 embedding 尺度更稳定，避免颜色 token 直接压过视觉 token。

### 5.4 Physical Projection

对应代码：

- `self.physical_proj`

输入：

```text
physical: [B, T, P]
```

结构：

```text
Linear(P -> E)
LayerNorm(E)
ReLU
```

输出：

```text
physical_emb: [B, T, E]
```

如果禁用 physical features：

```text
physical_dim = 0
physical_emb = zeros_like(frame_emb)
```

为什么这样做：

- physical token 是手工统计特征，尺度和 CNN embedding 完全不同，需要投影到同一表示空间。
- LayerNorm 降低不同物理特征量纲差异造成的训练不稳定。
- 与视觉/纹理 token 并列融合，而不是简单相加，避免含义混杂。

### 5.5 Token Fusion

对应代码：

- `self.fusion`

输入：

```text
frame_emb:    [B,T,E]
color_emb:    [B,T,E]
texture_emb:  [B,T,E]
physical_emb: [B,T,E]
```

融合：

```text
visual_color = frame_emb + color_emb
concat = cat([visual_color, texture_emb, physical_emb], dim=-1)
concat: [B,T,3E]

fusion MLP:
  Linear(3E -> E)
  LayerNorm(E)
  ReLU

fused: [B,T,E]
```

为什么这样做：

- `frame_emb + color_emb` 表示“当前视觉响应是在什么光刺激下发生的”。
- `texture_emb` 保持 CDC 分支独立贡献。
- `physical_emb` 保持显式物理统计独立贡献。
- concat 后用 MLP 学习三类 token 的权重和交互，比直接全部相加更可控。

希望获得的变化：

```text
多路异构信息
  -> 同维 embedding
  -> 单个逐帧 fused token
  -> 交给 Transformer 建模时间关系
```

### 5.6 FFT Head

对应代码：

- `self.fft_head`

输入：

```text
texture_emb: [B,T,E]
```

结构：

```text
Linear(E -> 256)
ReLU
Linear(256 -> 32*32)
Sigmoid
reshape -> [B,T,1,32,32]
```

输出：

```text
fft_maps: [B,T,1,32,32]
```

为什么这样做：

- FFT head 只接 `texture_emb`，使 CDC branch 必须保留可预测频谱/材质的信息。
- 它不直接参与推理结果，但通过训练 loss 改造 texture embedding。
- 与 MiniVision 静默活体项目中的 Fourier auxiliary supervision 思路一致：频域差异能辅助区分真假脸。

### 5.7 Positional Encoding + Transformer Encoder

对应代码：

- `PositionalEncoding`
- `nn.TransformerEncoderLayer`
- `nn.TransformerEncoder`

输入：

```text
fused: [B,T,E]
```

处理：

```text
features = pos_encoder(fused)
trans_out = transformer(features, src_key_padding_mask=padding_mask)
```

输出：

```text
trans_out: [B,T,E]
```

使用的技术：

- 正弦/余弦位置编码。
- Multi-head self-attention。
- Feed-forward network。
- padding mask。

相关论文：

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

为什么这样做：

- 炫彩活体不是单帧问题，关键在固定协议下的时间响应。
- Transformer 可以让每一帧关注其他帧，学习切光前后、稳定阶段、迟滞和周期模式。
- padding mask 保证变长 batch 中补齐帧不参与注意力。

### 5.8 Masked Temporal Pooling + Classifier

对应代码：

- masked pooling in `forward`
- `self.fc`

输入：

```text
trans_out: [B,T,E]
padding_mask: [B,T]
```

处理：

```text
valid_mask = ~padding_mask
pooled_out = sum(trans_out * valid_mask) / sum(valid_mask)

fc:
  Linear(E -> 128)
  ReLU
  Dropout(0.5)
  Linear(128 -> 1)
```

输出：

```text
logits: [B]
probability_live = sigmoid(logits)
```

为什么这样做：

- 不同样本有效帧数不同，masked mean pooling 可避免 padding 影响视频级表示。
- 平均池化让模型输出一个视频级活体判断，而不是逐帧判断。
- Dropout 降低小数据场景下的过拟合。

## 6. Loss 设计

对应代码：

- `compute_v3_loss`
- `contrast_depth_loss`

总损失：

```text
total_loss =
  cls_loss
  + lambda_depth * depth_loss
  + lambda_contrast * contrast_depth_loss
  + lambda_fft * fft_loss
```

默认：

```text
lambda_depth = 0.1
lambda_contrast = 0.1
lambda_fft = 0.05
```

### 6.1 分类损失 BCEWithLogitsLoss

输入：

```text
logits: [B]
labels: [B]
```

输出：

```text
cls_loss: scalar
```

技术：

- 二分类 logits + BCE。
- `pos_weight=auto` 时，按 train split 中 `spoof/live` 比例设置正类权重。

理由：

- 最终目标仍是视频级 live/spoof 二分类。
- 类别不均衡时，自动正类权重能缓解 live/spoof 比例偏差。

### 6.2 Pseudo-depth MSE

输入：

```text
depth_maps: [B,T,1,32,32]
labels:     [B]
valid_mask: [B,T]
```

target 构造：

```text
depth_targets = labels.view(B,1,1,1,1).expand_as(depth_maps)
live  -> 1
spoof -> 0
```

输出：

```text
depth_loss: scalar
```

理由：

- 这不是精确深度图，而是弱监督结构先验。
- 它迫使 CDC branch 不只给出分类相关 embedding，还产生空间结构响应。
- 对平面攻击有直接约束；对头模这类 3D 攻击，则依赖 texture/physical/FFT 与分类 loss 联合区分。

### 6.3 Contrast Depth Loss

输入：

```text
pred_depth:   [B,T,1,32,32]
target_depth: [B,T,1,32,32]
valid_mask:   [B,T]
```

处理：

- 使用 8 个局部差分 kernel。
- 对预测 depth 和 target depth 做局部差分。
- 计算 MSE。
- 只统计有效帧。

理由：

- 单纯 MSE 可能让 depth map 只学成全局 0 或 1。
- contrast depth loss 让局部结构差异也受到约束，借鉴 CDCN 中对局部深度/差分监督的思想。

### 6.4 FFT Loss

输入：

```text
fft_maps:    [B,T,1,32,32]
fft_targets: [B,T,1,32,32]
valid_mask:  [B,T]
```

输出：

```text
fft_loss: scalar
```

理由：

- 提供 per-frame 密集辅助监督。
- 约束 texture branch 保留频域材质信息。
- 训练期使用，推理期不要求用户提供 FFT 文件。

## 7. 训练、评测与推理

### 7.1 训练

训练 batch：

```text
batch_videos:       [B,T,6,224,224]
batch_colors:       [B,T,4]
batch_physical:     [B,T,P]
batch_fft:          [B,T,1,32,32]
batch_padding_mask: [B,T]
batch_labels:       [B]
```

前向输出：

```text
outputs = {
  "logits":     [B],
  "depth_maps": [B,T,1,32,32],
  "fft_maps":   [B,T,1,32,32],
}
```

训练配置：

- 默认 AMP 混合精度。
- 默认 `pin_memory=True`。
- 默认 `persistent_workers=True`。
- 可见多 GPU 时可用 `nn.DataParallel`。
- checkpoint 保存 config、threshold、split_counts、train/val/test metrics。

为什么保存 config：

- 推理时需要知道 `target_size`、`physical_dim`、`cdc_theta`、loss 权重、缺失颜色协议等训练时配置。
- 避免推理模型结构与 checkpoint 不一致。

### 7.2 阈值与指标

评测输出：

- accuracy
- AUC
- EER
- threshold
- cls/depth/contrast/fft loss

阈值：

- 训练/验证阶段通过验证数据寻找较优 threshold。
- checkpoint 保存该 threshold。
- 推理时默认使用 checkpoint threshold，也可手动覆盖。

为什么不固定 0.5：

- 数据集类别比例、攻击强度、模型校准和损失权重都会影响概率分布。
- 安全场景通常更关注 APCER/BPCER 或业务阈值，需要允许阈值调优。

### 7.3 单视频推理

推理输入：

```text
checkpoint
video_path
txt_path optional
threshold optional
```

推理时在线构造：

```text
frames_tensor:   [1,T,6,224,224]
color_tensor:    [1,T,4]
physical_tensor: [1,T,P]
padding_mask:    [1,T] all False
```

输出：

```json
{
  "video_path": "...",
  "probability_live": 0.xxxxxx,
  "threshold": 0.xxxxxx,
  "prediction": "live|spoof"
}
```

注意：

- 推理不需要外部 FFT target。
- 如果缺少 txt，会按 checkpoint config 中的 `missing_color_protocol` 生成颜色协议；这只适用于视频采集确实符合该协议的情况。

## 8. V2 到 V3 的关键变化

| 维度 | V2 | V3 | 为什么要变 |
| --- | --- | --- | --- |
| 数据索引 | live/spoof 目录 | manifest + label/media/category/source_group | 保证攻击类型覆盖和可解释评估 |
| 媒体类型 | 主要视频 | 视频 + 图片 | 利用更多静态材质样本，但视频仍是主线 |
| 颜色条件 | txt -> `[T,4]` | txt 或 collect_flash -> `[T,4]` | 保持条件输入，兼容缺失 txt 样本 |
| 视觉输入 | RGB + Diff 6 通道 | RGB + Diff 6 通道 | 保留 V2 有效短时动态建模 |
| 主视觉分支 | ResNet18 | ResNet18 + 冻结早期层 | 稳定视觉语义，减少过拟合 |
| 纹理分支 | 无 | CDC texture branch | 强化局部差分、材质、边缘 |
| 显式物理线索 | 无 | frequency/flash/rPPG token | 给 Transformer 直接物理统计证据 |
| 伪深度 | 无 | depth_maps + depth loss | 提供结构先验 |
| 局部深度差分 | 无 | contrast depth loss | 避免 depth map 只学全局常数 |
| 频域监督 | 无 | FFT target + FFT head | 约束 texture branch 学频域材质 |
| 模型输出 | logits | logits + depth_maps + fft_maps | 分类 + 辅助监督 |
| Loss | BCE | BCE + depth + contrast + FFT | 多任务约束，降低捷径学习 |

## 9. Tensor 变化总表

| 阶段 | 输入 | 输出 | 形状变化 | 变化目的 |
| --- | --- | --- | --- | --- |
| 样本发现 | 文件树 / manifest | `LivenessSample` | 无 tensor | 保留 label/category/source |
| 视频读取 | video + txt | frames/color | `T x raw frame` | 保留顺序帧和颜色对齐 |
| 人脸预处理 | BGR frames | face frames | `[T,H,W,3] -> [T,224,224,3]` | 统一人脸区域 |
| RGB normalize | BGR uint8 | RGB float | `[0,255] -> [0,1]` | 稳定数值范围 |
| Frame diff | RGB sequence | diff | `[T,224,224,3]` | 显式短期动态 |
| 6ch concat | RGB + diff | video tensor | `[T,224,224,6] -> [T,6,224,224]` | CNN 同时看外观和变化 |
| Color encode | color int | color tensor | `[T] -> [T,4]` | 光色条件可学习 |
| Physical extractor | frames + colors | physical tensor | `[T,224,224,3] -> [T,19]` | 显式物理统计 |
| FFT target | frames | fft tensor | `[T,224,224,3] -> [T,1,32,32]` | 频域辅助监督 |
| Collate | variable T | padded batch | `T_i -> max_len` | batch 化变长序列 |
| ResNet | videos | frame_emb | `[B*T,6,224,224] -> [B,T,E]` | 视觉语义 |
| CDC | videos | texture/depth | `[B*T,6,224,224] -> [B,T,E] + [B,T,1,32,32]` | 材质纹理 + 结构先验 |
| Color proj | colors | color_emb | `[B,T,4] -> [B,T,E]` | 光色条件 embedding |
| Physical proj | physical | physical_emb | `[B,T,P] -> [B,T,E]` | 物理 token embedding |
| Fusion | 3 路 embedding | fused | `[B,T,3E] -> [B,T,E]` | 融合逐帧多模态证据 |
| Transformer | fused + mask | trans_out | `[B,T,E] -> [B,T,E]` | 长时序建模 |
| Masked pooling | trans_out | pooled | `[B,T,E] -> [B,E]` | 视频级表示 |
| Classifier | pooled | logits | `[B,E] -> [B]` | live/spoof 判定 |

## 10. 设计假设与风险

### 10.1 缺失 txt 的协议假设

`collect_flash` 补齐颜色标签只有在视频确实按该协议采集时才合理。如果视频来自其他打光时序，颜色 token 和 flash-response token 会错位。

建议：

- 正式训练优先使用 `--require-color-txt`。
- 对历史无 txt 数据单独评估，不要和严格协议数据混在一起得出结论。

### 10.2 Pseudo-depth 是弱监督

当前 depth target 由 label 构造：

```text
live=1, spoof=0
```

它不是精确几何深度。对 3D 头模、硅胶面具等 3D 攻击，单靠 pseudo-depth 不足以区分，必须依赖材质、颜色响应、频域、rPPG-like 线索和分类 loss。

### 10.3 rPPG-like 特征不是医学级心率检测

当前 rPPG 特征只是绿色通道统计和简单频域峰值，不是完整 rPPG pipeline。它适合作为辅助证据，不适合作为独立活体判据。

### 10.4 全帧训练的内存压力

V3 支持全帧读取，`[B,T,6,224,224]` 会随 `T` 线性增长。实际训练需要结合：

- `frame_stride`
- `max_train_frames`
- `max_eval_frames`
- batch size
- AMP

进行资源控制。

### 10.5 Detector 失败与中心裁剪

中心裁剪保证鲁棒加载，但如果视频中人脸不居中，可能引入背景和非脸区域。正式训练建议配置检测模型，并检查 corrupted sample 记录。

## 11. 相关论文与项目

| 技术点 | 引用/项目 | 本项目使用方式 |
| --- | --- | --- |
| ResNet18 主干 | He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | 使用 ResNet18 去掉 FC 的 backbone，conv1 改 6 通道 |
| Transformer 时序建模 | Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 使用 PyTorch `TransformerEncoder` 建模帧间关系 |
| Central Difference Convolution / CDCN | Yu et al., [Searching Central Difference Convolutional Networks for Face Anti-Spoofing](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.html) | 实现 `Conv2dCD` 与 CDC texture branch，加入 pseudo-depth/contrast depth 思想 |
| Fourier auxiliary supervision / MiniFASNet | MiniVision, [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) | 借鉴傅里叶频谱图辅助监督思想，V3 用 `fft_head` 预测每帧频谱 target |
| YOLOv7 检测 | Wang et al., [YOLOv7](https://arxiv.org/abs/2207.02696) | 人脸预处理依赖 YOLOv7-face 类检测/对齐封装 |
| YOLOv7-face 工程 | [derronqi/yolov7-face](https://github.com/derronqi/yolov7-face) | 本仓库 `yolov7_face/` 与 `Face_detection_yolo_align.py` 的人脸检测/关键点对齐来源 |
| rPPG 概念背景 | de Haan and Jeanne, [Robust pulse-rate from chrominance-based rPPG](https://research.tue.nl/en/publications/robust-pulse-rate-from-chrominance-based-rppg/) | V3 只使用 green-channel 统计作为辅助物理 token，不复现完整 rPPG |
| 微弱颜色变化背景 | Wu et al., [Eulerian Video Magnification](https://people.csail.mit.edu/mrub/evm/) | 解释面部微弱颜色变化可作为生理线索的背景，不直接实现 EVM |

## 12. 本项目内的实现位置

| 模块 | 文件 | 关键函数/类 |
| --- | --- | --- |
| 数据发现 | [flash_liveness_project_v3.py](../../flash_liveness_project_v3.py) | `collect_samples_from_manifest`, `stratified_split`, `discover_dataset_splits` |
| 颜色读取 | [flash_liveness_project_v3.py](../../flash_liveness_project_v3.py) | `parse_color_txt`, `color_int_to_feature` |
| 缺失协议生成 | [scripts/collect_flash_liveness_video.py](../../scripts/collect_flash_liveness_video.py) | `build_frame_color_labels`, `COLOR_SEQUENCE_RGB` |
| 人脸预处理 | [flash_liveness_project_v3.py](../../flash_liveness_project_v3.py), [Face_detection_yolo_align.py](../../Face_detection_yolo_align.py) | `FacePreprocessor`, `YOLOv7_face_mkl` |
| Dataset | [flash_liveness_project_v3.py](../../flash_liveness_project_v3.py) | `FlashLivenessDataset` |
| 物理特征 | [flash_physical_features.py](../../flash_physical_features.py) | `PhysicalCueExtractor`, `FrequencyArtifactExtractor`, `FlashResponseFeatureExtractor`, `RPPGFeatureExtractor` |
| FFT target | [flash_liveness_project_v3.py](../../flash_liveness_project_v3.py) | `_fft_target_from_frames` |
| 模型 | [flash_liveness_project_v3.py](../../flash_liveness_project_v3.py) | `Conv2dCD`, `CDCTextureBranch`, `CNNTransformerLiveness` |
| Loss | [flash_liveness_project_v3.py](../../flash_liveness_project_v3.py) | `compute_v3_loss`, `contrast_depth_loss` |
| 训练/评测 | [flash_liveness_project_v3.py](../../flash_liveness_project_v3.py) | `train_one_epoch`, `evaluate_model`, `train_model` |
| 推理 | [flash_liveness_project_v3.py](../../flash_liveness_project_v3.py) | `predict_video`, `load_checkpoint` |

## 13. 总结

V3 的核心不是简单堆叠更多模块，而是把炫彩活体拆成四类互补证据：

1. 视觉语义：人脸外观、纹理、局部结构。
2. 短时动态：相邻帧 diff 和切光瞬间响应。
3. 已知刺激条件：逐帧颜色协议。
4. 显式物理统计：频域、闪光响应、rPPG-like 线索。

这些证据先在逐帧层面融合，再交给 Transformer 建模长时序关系。辅助监督则让模型在二分类之外学习 pseudo-depth 和频域结构，降低只依赖背景、设备或采集来源捷径的风险。最终推理接口仍保持简单：输入视频和可选颜色 txt，输出 `probability_live` 与 `live/spoof` 结果。
