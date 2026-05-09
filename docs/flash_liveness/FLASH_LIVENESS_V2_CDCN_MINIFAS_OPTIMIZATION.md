# Flash Liveness V2 引入 CDCN 与 MiniFASNet V2 技术的优化方案

## 1. 目标

本文档用于整理 `CDCN-Face-Anti-Spoofing.pytorch` 与 `face-antispoof-onnx` 中的算法思想，分析它们与当前 `flash_liveness_project_v2.py` 炫彩视频活体模型的关系，并给出哪些方法适合加入 V2、为什么适合、如何逐步加入。

目标不是把 CDCN 或 MiniFASNet V2 直接替换为炫彩活体主模型，而是以当前 Flash Liveness V2 为基准，吸收它们对静默活体、纹理、频域、伪深度和部署工程的优势，让 V2 在视频炫彩活体检测中更好覆盖：

- 真人人脸。
- 打印照片攻击。
- 手机/屏幕重放攻击。
- 纸面/剪裁攻击。
- 硅胶头模/3D 头模攻击。
- 面具、包裹式假脸、非皮肤材质攻击。
- 新设备、新光照、新相机、新采集环境下的未知攻击。

## 2. 当前 V2 基线

当前 V2 的核心文件是：

```text
flash_liveness_project_v2.py
```

现有输入形式：

```text
video frames: T x H x W x 3
diff frames:  T x H x W x 3
concat:       T x H x W x 6
model input:  T x 6 x H x W
color txt:    T x 4 = [R, G, B, transition]
```

现有模型结构可以概括为：

```text
RGB + Diff 6通道
-> ResNet18 frame backbone
-> frame embedding

txt color
-> color_proj
-> color embedding

frame embedding + color embedding
-> positional encoding
-> Transformer Encoder
-> temporal pooling with padding_mask
-> live/spoof classification
```

V2 的优势：

- 使用完整视频帧，不只抽 16 帧。
- 使用 txt 中逐帧颜色信息。
- 使用 Diff 通道捕捉光照切换和运动变化。
- 使用 Transformer 建模长时序关系。
- 使用 padding mask 支持不同长度视频。

V2 当前仍可加强的方向：

- 局部纹理和材质伪影。
- 屏幕/打印频域模式。
- 伪深度和 3D 结构约束。
- 对头模、硅胶、面具等 3D 攻击的物理线索。
- 让模型更明确地区分“光学响应”和“普通亮度变化/裁剪抖动”。

## 3. CDCN 与 MiniFASNet V2 的异同

| 维度 | CDCN / CDCN++ | MiniFASNet V2 SE |
| --- | --- | --- |
| 项目 | `CDCN-Face-Anti-Spoofing.pytorch` | `face-antispoof-onnx` |
| 任务类型 | 静默活体 | 静默活体 |
| 输入 | RGB 人脸图，常用 256x256 | RGB 人脸图，常用 128x128 |
| 输出 | 32x32 伪深度图 | real/spoof logits |
| 核心算子 | Central Difference Convolution | Depthwise separable conv + SE |
| 训练监督 | pseudo-depth map + contrast depth loss | classification + Fourier auxiliary loss |
| 主要线索 | 局部梯度、纹理、伪深度 | 纹理、频域、通道注意力 |
| 适合迁移到 V2 的点 | CDC 卷积、伪深度、contrast depth loss、空间注意力 | FFT 辅助监督、SE 通道注意力、轻量部署 |
| 不适合直接迁移的点 | 单帧 depth score 直接做最终判断 | 单帧二分类直接做最终判断 |

共同点：

- 都不是炫彩模型。
- 都不读取闪光颜色 txt。
- 都不建模绿光血红蛋白吸收、rPPG、光学迟滞等炫彩时序线索。
- 都更擅长从单帧纹理、局部材质、屏幕/打印伪影中寻找攻击迹象。

核心差别：

- CDCN 更强调“局部差分 + 伪深度结构”。
- MiniFASNet V2 更强调“轻量分类 + 频域纹理 + 通道注意力”。

## 4. 哪个更有利于炫彩活体检测

如果只选一个作为 V2 的增强方向，优先级应是：

```text
CDCN 思想 > MiniFASNet V2 整体模型
```

理由：

- 炫彩活体最终要区分的不只是单帧真假，而是真人皮肤/血液/三维结构在不同光色下的响应。
- CDCN 的 Central Difference Convolution 对局部梯度和材质差异更敏感，适合增强光照切换下的皮肤、硅胶、屏幕、打印表面差别。
- CDCN 的伪深度监督与 3D 结构假设更接近，适合头模、纸片、屏幕等攻击。
- Contrast Depth Loss 可以让模型关注局部深度差分，而不是只学全局亮度。

但 MiniFASNet V2 中的 Fourier 辅助监督非常值得加入：

- 屏幕重放的摩尔纹、像素栅格通常在频域更明显。
- 打印攻击的半色调网点、纸纹也在频域有特征。
- 它可以作为训练期辅助任务，不一定增加推理成本。

因此最合理结论不是二选一，而是：

```text
V2 主干保留
+ CDCN 的局部差分/伪深度思想
+ MiniFASNet 的 Fourier 频域监督
+ 现有 flash_physical_features.py 的频域/闪光/rPPG/深度线索
```

## 5. 推荐加入 V2 的方法

### 5.1 Central Difference Convolution

来源：

```text
CDCN-Face-Anti-Spoofing.pytorch/models/CDCNs.py -> Conv2d_cd
```

推荐等级：

```text
高
```

加入理由：

- 当前 V2 输入是 RGB + Diff 6通道，Diff 已经强调帧间变化，但普通卷积对局部中心差异没有显式约束。
- CDC 卷积可以增强局部梯度、边缘、反射纹理和材质变化。
- 对屏幕、打印、纸片、头模等攻击，局部纹理与真实皮肤差异明显。
- 炫彩打光下，假体材质常出现不自然的边缘高光、反射块和局部梯度突变，CDC 更容易捕捉。

建议加入位置：

```text
flash_liveness_project_v2.py
CNNTransformerLiveness.__init__
self.cnn_backbone[0] = nn.Conv2d(6, 64, ...)
```

可尝试替换为：

```text
Conv2d_cd(6, 64, kernel_size=7, stride=2, padding=3, theta=0.7)
```

注意：

- 原 CDCN 的 `Conv2d_cd` 默认 kernel 3x3，需要适配 7x7 或把第一层改成 3x3 stack。
- 对 6 通道输入要修改 `in_channels=6`。
- 初期建议只替换第一层或增加一个并联 CDC 分支，不要大规模替换整个 ResNet，避免训练不稳定。

预期提升：

- 增强局部材质差异。
- 提升对打印、屏幕、头模表面纹理的泛化。
- 减少模型只依赖全局颜色/亮度的风险。

### 5.2 CDCN-style 局部纹理分支

推荐等级：

```text
高
```

加入理由：

直接替换 ResNet 第一层可能影响预训练权重兼容性。更稳妥的方法是给 V2 增加一个轻量 CDC 分支：

```text
RGB/Diff frame
-> CDC texture branch
-> texture embedding

ResNet frame branch
-> frame embedding

color_proj
-> color embedding

frame + texture + color
-> Transformer
```

这样 V2 原有路径保持稳定，CDC 分支只补充局部差分信息。

建议输出：

```text
texture_emb: [B, T, embed_dim]
```

融合方式：

```python
features = frame_emb + color_emb + texture_emb
```

或：

```python
features = fusion_mlp(torch.cat([frame_emb, color_emb, texture_emb], dim=-1))
```

预期提升：

- 局部纹理能力增强。
- 不破坏原 V2 主干。
- 更适合做 ablation 对比。

### 5.3 Pseudo-depth Head

来源：

```text
CDCN 输出 32x32 depth map
```

推荐等级：

```text
高
```

加入理由：

活体检测不能只靠分类标签。伪深度可以给模型一个更物理的约束：

```text
真人脸 -> 应有稳定的面部 3D 结构
平面屏幕/打印 -> 深度异常或趋近平面
头模/硅胶 -> 有 3D 结构但材质/光学响应异常
```

对头模来说，仅靠 depth 不一定够，但 depth + 闪光响应 + 频域纹理会更强。

建议加入方式：

```text
frame feature map
-> small decoder
-> pseudo_depth_map [B*T, 1, 32, 32]
```

监督标签初期可用弱监督：

```text
live  -> 全 1 depth map
spoof -> 全 0 depth map
```

如果后续能接入人脸深度估计/3DMM/normal cue，可换成更真实的深度标签。

Loss：

```text
depth_loss = MSE(pred_depth, pseudo_label)
```

预期提升：

- 让模型不只学颜色，还学结构。
- 对平面攻击更敏感。
- 与 ThunderGuard 的 normal/depth 思路形成互补。

### 5.4 Contrast Depth Loss

来源：

```text
CDCN-Face-Anti-Spoofing.pytorch/models/loss.py
```

推荐等级：

```text
高
```

加入理由：

单纯 MSE 只约束每个像素值，容易让模型学成全局亮度差。Contrast Depth Loss 用 8 个方向的中心差分约束深度图局部结构，更适合活体检测：

```text
预测深度图局部差分
vs
标签深度图局部差分
```

对 V2 的价值：

- 约束模型关注局部结构，而不是只靠整脸平均亮度。
- 对打印/屏幕的平面结构有帮助。
- 对光照变化导致的假边缘也有抑制作用。

建议加入总 loss：

```text
total_loss =
    cls_loss
  + lambda_depth * depth_mse
  + lambda_contrast * contrast_depth_loss
```

初始建议：

```text
lambda_depth = 0.1
lambda_contrast = 0.1
```

之后根据验证集 APCER/BPCER 调整。

### 5.5 Fourier / FFT Auxiliary Loss

来源：

```text
face-antispoof-onnx/src/minifasv2/model.py -> FTGenerator
face-antispoof-onnx/src/minifasv2/data.py -> generate_FT
```

推荐等级：

```text
高
```

加入理由：

频域线索对屏幕和打印攻击非常有效：

- 屏幕像素栅格。
- 摩尔纹。
- 打印半色调网点。
- 视频翻拍压缩纹。
- 人造材料表面的周期纹理。

当前 `flash_physical_features.py` 已经有 `FrequencyArtifactExtractor`，说明这个方向和当前项目已有分析一致。

两种加入方式：

训练期辅助监督：

```text
backbone feature
-> FT head
-> predicted FFT map
-> MSE with input face FFT map
```

显式物理特征 token：

```text
flash_physical_features.py
FrequencyArtifactExtractor
-> T x 5 frequency feature
-> physical_proj
-> Transformer fusion
```

推荐先做第二种，因为项目里已有 `flash_physical_features.py`，实现成本更低。

预期提升：

- 提升屏幕/打印攻击拦截。
- 降低模型对训练集具体设备的过拟合。
- 增强未知攻击泛化。

### 5.6 SE / Channel Attention

来源：

```text
MiniFASNet V2 SE
```

推荐等级：

```text
中高
```

加入理由：

V2 的输入不仅有 RGB，还有 Diff 和颜色 token。不同攻击类型下，不同通道的重要性不同：

- 绿光下可能更关注皮肤血液响应。
- 红光下可能更关注深层散射。
- 紫红光下可能更关注材质反射差异。
- Diff 通道在光色切换瞬间更关键。

SE 可以帮助模型动态调整通道权重。

建议加入位置：

- 6通道输入后的浅层卷积后。
- CDC texture branch 后。
- RGB/Diff 分支融合前。

预期提升：

- 多通道信息利用更灵活。
- 减少无关通道噪声。
- 对不同采集光色和设备有更好适应性。

### 5.7 Spatial Attention

来源：

```text
CDCNpp -> SpatialAttention
```

推荐等级：

```text
中高
```

加入理由：

活体线索不是全图平均分布，真正有价值的区域通常是：

- 面颊。
- 额头。
- 鼻梁。
- 眼周皮肤。
- 嘴角周围皮肤。

而背景、衣服、头发、屏幕边框可能是干扰。

空间注意力可以让模型把权重放在人脸皮肤区域。

注意：

- 如果人脸 crop 不稳定，空间注意力可能学到错误位置。
- 建议先保证 FacePreprocessor/YOLO crop 稳定，再加 spatial attention。

### 5.8 Physical Feature Token

来源：

```text
flash_physical_features.py
```

推荐等级：

```text
高
```

现有模块已经包含：

```text
FrequencyArtifactExtractor
FlashResponseFeatureExtractor
RppgFeatureExtractor
DepthNormalFeatureLoader
PhysicalCueExtractor
```

这些线索与 CDCN/MiniFAS 的优势互补：

- CDCN 强局部梯度/伪深度。
- MiniFAS 强频域纹理。
- PhysicalCueExtractor 强闪光响应、rPPG、深度/法线。

建议将 physical features 正式接入 V2：

```text
physical_np: T x physical_dim
physical_proj: Linear -> embed_dim
features = frame_emb + color_emb + physical_emb
```

预期提升：

- 对头模和硅胶材质增强。
- 对真人血液/绿光响应增强。
- 对屏幕/打印频域伪影增强。
- 对新攻击类型更有解释性。

## 6. 不建议加入或不建议直接使用的方法

### 6.1 直接使用 MiniFASNet V2 单帧输出作为最终结果

不建议。

原因：

- 它不读颜色 txt。
- 它不建模闪光时序。
- 它容易受炫彩帧颜色偏移影响。
- 我们之前测试中它在当前数据集上对 spoof 放过偏多。

适合用途：

```text
静默初筛、辅助分支、频域监督参考
```

### 6.2 直接使用 CDCN 单帧 depth score 作为最终结果

不建议。

原因：

- CDCN 不知道光色切换。
- 头模也可能有 3D 结构，单纯伪深度无法完全区分。
- 如果没有针对当前炫彩数据训练，域偏移会很大。

适合用途：

```text
局部差分 backbone、伪深度辅助监督、contrast depth loss
```

### 6.3 只增加更多分类层

不建议只堆分类头。

炫彩活体需要物理约束，而不是更复杂的黑盒分类器。应优先增加：

- 频域监督。
- 伪深度监督。
- rPPG/绿光响应。
- 闪光响应统计。
- spoof type 辅助任务。

## 7. 面向全部攻击类型的优化矩阵

| 攻击类型 | 主要破绽 | 推荐加入模块 | 理由 |
| --- | --- | --- | --- |
| 打印照片 | 平面、纸纹、半色调 | CDC Conv、FFT、pseudo-depth | 局部纹理和频域明显，深度趋近平面 |
| 手机/屏幕重放 | 摩尔纹、像素栅格、屏幕反光 | FFT、SE、Spatial Attention、Diff | 屏幕有周期纹理和反光，Diff 可捕捉刷新/光照异常 |
| 纸面剪裁 | 边缘异常、平面结构 | CDC Conv、contrast depth、face-boundary attention | 中心差分对边缘和局部梯度敏感 |
| 硅胶头模 | 有3D但无真实皮肤血流 | rPPG、flash response、green-channel features、CDC texture | 需要区分“有形状”和“有活体生物响应” |
| 3D 头模 | 形状像真人但材质不同 | flash response、FFT、CDC texture、depth/normal | 材质反射和颜色迟滞不同 |
| 屏幕录制/翻拍视频 | 频域伪影、时序不自然 | FFT、temporal Transformer、Diff、spoof type loss | 需要空间频域 + 时间模式共同判断 |
| 未知攻击 | 分布外材质或设备 | 多辅助任务、quality score、threshold gray zone | 避免模型只记住训练集攻击样式 |

## 8. 推荐的 V2 优化架构

建议目标结构：

```text
video frames + txt colors
       |
       v
RGB frames + Diff frames: T x 6 x H x W
       |
       +--> ResNet frame backbone ------------------+
       |                                            |
       +--> CDC texture branch ---------------------+--> fusion
       |                                            |
       +--> pseudo-depth head + contrast depth loss |
                                                    |
color txt: T x 4 -> color_proj ---------------------+
                                                    |
physical features: T x D -> physical_proj ----------+
                                                    |
FFT auxiliary head / frequency token ---------------+
                                                    |
                 fused tokens + PE
                         |
                  Transformer Encoder
                         |
                valid-frame temporal pooling
                         |
          live/spoof + spoof_type + quality score
```

推荐输出：

```text
main_logits: live/spoof
spoof_type_logits: print/screen/head_model/mask/unknown
depth_map: pseudo-depth auxiliary
quality_score: face quality / color protocol quality
```

## 9. 推荐分阶段实验路线

### Stage 1：先接入 physical feature token

优先级最高，因为项目已有 `flash_physical_features.py`。

改动：

```text
Dataset process_video 返回 physical_features
collate_fn padding physical_features
CNNTransformerLiveness 增加 physical_proj
features = frame_emb + color_emb + physical_emb
```

理由：

- 实现成本最低。
- 可解释性强。
- 能直接利用频域、闪光响应、rPPG。

验证指标：

- APCER 是否下降。
- 尤其观察 spoof 中屏幕/打印/头模的分项结果。

### Stage 2：增加 FFT / frequency auxiliary loss

改动：

```text
backbone middle feature -> FT head -> FFT map
loss += lambda_fft * MSE(pred_fft, target_fft)
```

理由：

- 提升屏幕/打印泛化。
- 与 MiniFASNet V2 的有效经验一致。
- 训练期增强，不一定增加推理复杂度。

### Stage 3：增加 CDC texture branch

改动：

```text
新增 Conv2d_cd
RGB/Diff -> CDC shallow branch -> texture_emb
texture_emb 融合 Transformer token
```

理由：

- 强化局部梯度/材质差异。
- 保留原 ResNet 主干，降低训练风险。

### Stage 4：增加 pseudo-depth + contrast depth loss

改动：

```text
frame feature map -> depth decoder -> 32x32 map
loss += depth_mse + contrast_depth_loss
```

理由：

- 引入结构约束。
- 对平面攻击、纸片、屏幕有帮助。
- 与 CDCN 的核心优势一致。

### Stage 5：增加 spoof type 辅助分类

改动：

数据集整理时给 spoof 样本增加类型标签：

```text
print
screen_replay
head_model
silicone
paper_mask
unknown
```

模型增加：

```text
spoof_type_head
```

理由：

- 让模型学习“为什么假”，而不是只学“真假”。
- 对未知攻击泛化更好。
- 便于分析每类攻击是否被防住。

## 10. Loss 设计建议

建议总 loss：

```text
total_loss =
    cls_loss
  + lambda_depth * depth_mse
  + lambda_contrast * contrast_depth_loss
  + lambda_fft * fft_loss
  + lambda_type * spoof_type_loss
  + lambda_quality * quality_loss
```

初始权重建议：

```text
lambda_depth = 0.1
lambda_contrast = 0.1
lambda_fft = 0.05
lambda_type = 0.2
lambda_quality = 0.05
```

调参原则：

- 如果 spoof 放过多，优先提高 `lambda_depth/lambda_contrast/lambda_fft` 或提高最终阈值。
- 如果真人误杀多，检查 rPPG/flash response 特征是否噪声过大，降低辅助 loss 或加入质量门控。
- 如果某一类攻击漏检，增加 spoof type loss 和该类采样权重。

## 11. 数据与评估建议

为了让 V2 防护住全部类型并具备强泛化，训练和评估不能只看总体 accuracy。

必须保留：

```text
overall metrics:
accuracy / APCER / BPCER / ACER / AUC / EER

per attack type:
print APCER
screen APCER
head_model APCER
silicone APCER
paper_mask APCER
unknown APCER

per source domain:
camera/device
lighting
video format
collection protocol
```

评估目标建议：

```text
安全优先:
APCER 尽量低，尤其 head_model/screen/print 不能放过

体验约束:
BPCER 不应过高，避免真人大量误杀

泛化:
新增设备/新增攻击类型单独做 hold-out test
```

## 12. 关键风险

### 风险 1：辅助任务过多导致训练不稳定

解决：

- 分阶段加入。
- 每次只加一个模块。
- 保留 ablation。

### 风险 2：模型学到采集设备而不是活体线索

解决：

- 按设备/来源划分 train/val/test。
- 做跨域 hold-out。
- 加强颜色/亮度增强，但不要破坏 txt 颜色对应关系。

### 风险 3：pseudo-depth 标签过粗

初期用 live=1/spoof=0 的伪深度是弱监督，不是真实深度。

解决：

- 先低权重使用。
- 后续引入离线深度/normal cue。
- 与 contrast depth loss 搭配，不让它只学全局常数。

### 风险 4：rPPG 在短视频或强运动下不稳定

解决：

- 将 rPPG 作为辅助 token，不作为硬规则。
- 增加 face quality / motion quality 判断。
- 对低质量片段降低 rPPG 权重。

## 13. 最终建议

以当前 V2 为基准，最推荐的优化组合是：

```text
第一优先级:
physical feature token
FFT/frequency features
CDCN-style texture branch

第二优先级:
pseudo-depth head
contrast depth loss
SE/channel attention
spatial attention

第三优先级:
spoof type auxiliary classification
quality score / gray-zone decision
ONNX/INT8 部署优化
```

如果只做一版最有性价比的增强，建议：

```text
V2 + physical_proj + frequency features + CDC texture branch
```

如果做一版更完整的强泛化模型，建议：

```text
V2
+ physical feature token
+ CDC texture branch
+ FFT auxiliary loss
+ pseudo-depth head
+ contrast depth loss
+ spoof type head
```

## 14. 结论

CDCN 更适合作为炫彩 V2 的局部结构和伪深度增强来源，MiniFASNet V2 更适合作为频域监督和轻量部署参考。

面向炫彩视频活体检测，最优路线不是直接替换模型，而是保留 V2 的颜色时序和 Transformer 主干，同时吸收：

```text
CDCN: Central Difference Conv + pseudo-depth + contrast depth
MiniFASNet V2: Fourier auxiliary loss + SE/channel attention + ONNX部署经验
Flash physical features: frequency + flash response + rPPG + depth/normal
```

这样才能让模型既理解“炫彩光色时序”，又理解“真实皮肤/血液/三维结构/材质纹理”，从而更有机会防护住数据集中的全部攻击类型，并提升对未知攻击和新采集环境的泛化能力。
