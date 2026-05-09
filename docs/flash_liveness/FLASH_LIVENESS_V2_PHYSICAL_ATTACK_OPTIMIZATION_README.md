# V2 闪光活体物理攻击优化思路

本文档用于记录 `flash_liveness_project_v2.py` 当前对物理呈现攻击的处理能力，以及训练结束后可以继续加入的生物信号、光学响应、纹理频域和深度几何优化方向。

物理呈现攻击，Physical Presentation Attack / Face Anti-Spoofing，常见形式包括打印照片、硅胶头模、3D 面具、屏幕播放真实人脸视频等。它们可能在外观上接近真人，但通常缺少真实皮肤、血液、三维几何和成像链路共同产生的细微信号。

## 当前 V2 已有能力

当前 V2 训练代码位于：

```text
/supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_project_v2.py
```

V2 当前输入由两部分组成：

```text
video_tensor: B x T x 6 x 224 x 224
color_tensor: B x T x 4
padding_mask: B x T
```

其中：

```text
B = batch size
T = 当前 batch 内 padding 后的视频最大帧数
6 = RGB 3通道 + 相邻帧差分 3通道
4 = 当前闪光颜色 R/G/B + 是否发生颜色切换 transition
```

当前核心流程：

```text
视频逐帧读取
  -> 人脸裁剪或中心裁剪
  -> RGB 归一化
  -> 计算相邻帧差分 diff_frames[t] = frame[t] - frame[t-1]
  -> 拼接为 6 channel 输入
  -> ResNet18 提取逐帧空间特征
  -> color_proj 注入每帧闪光颜色信息
  -> Transformer 建模完整时序
  -> padding mask 忽略补齐帧
  -> 输出 live/spoof 分数
```

因此，V2 已经具备以下隐式能力：

```text
1. 利用 RGB 外观判断人脸纹理、材质、边缘和色彩异常。
2. 利用帧间差分感知闪光切换瞬间的亮度/颜色响应。
3. 利用 color_tensor 知道每一帧对应的打光颜色。
4. 利用 Transformer 学习不同光色下的人脸响应顺序。
5. 利用全帧读取保留完整光色切换瞬间和迟滞效应。
```

但这些能力主要是“让模型自己从数据里学”，并没有显式计算下面这些物理攻击判据。

## 当前 V2 没有显式处理的线索

当前代码没有单独实现以下模块：

```text
1. rPPG 心率/血流波动检测。
2. 皮肤次表面散射 SSS 响应建模。
3. 屏幕摩尔纹、刷新纹、高频纹理异常检测。
4. 单目深度图、法线图、3D 曲面一致性检测。
5. 眼部、脸颊、额头等区域的局部生理信号对比。
6. 光色切换后的响应延迟、恢复曲线、迟滞曲线显式特征。
```

所以当前 V2 可以“间接学习”这些现象，但还不是一个显式的物理先验模型。

## 四类物理攻击线索

### 1. rPPG 血液流动

真人皮肤下有血液周期性流动，尤其绿光通道对血红蛋白吸收更敏感。真实人脸在连续视频中会出现微弱、周期性的颜色变化。

攻击样本常见问题：

```text
打印照片：没有真实血流周期。
硅胶头模：材质反射可能变化，但不具备真实脉搏周期。
屏幕 replay：可能播放了人脸视频，但屏幕刷新、压缩和二次拍摄会破坏真实 rPPG。
```

可加入特征：

```text
face_mean_rgb[t]       每帧人脸区域 RGB 均值
cheek_mean_rgb[t]      脸颊区域 RGB 均值
forehead_mean_rgb[t]   额头区域 RGB 均值
green_signal[t]        绿色通道时序
rppg_fft               频谱峰值、主频能量、信噪比
rppg_consistency       左右脸颊、额头之间的周期一致性
```

建议在 V2 中新增：

```text
PhysiologyFeatureExtractor
  -> 输入: T x H x W x 3 人脸帧序列
  -> 输出: T x C_rppg 或 video-level C_rppg
```

训练融合方式：

```text
CNN frame feature + color embedding + rPPG embedding -> Transformer
```

### 2. SSS 次表面散射

真人皮肤不是单纯镜面反射。光进入皮肤后会发生吸收、散射和再出射，不同颜色光照下，皮肤的反射变化存在生物组织特征。

攻击样本常见问题：

```text
打印纸：反射更接近平面材料反射。
硅胶头模：可能有皮肤色，但血液吸收和组织散射不真实。
屏幕 replay：发光来自屏幕像素，不是皮肤反射闪光。
```

可加入特征：

```text
response_purple[t]       紫红光下的区域响应
response_green[t]        绿光下的区域响应
response_red[t]          红光下的区域响应
response_ratio_rg        红/绿响应比
response_ratio_pg        紫红/绿响应比
transition_decay         光色切换后的恢复速度
hysteresis_curve         光学迟滞曲线
```

建议在 V2 中新增：

```text
FlashResponseFeatureExtractor
  -> 根据 txt 中的颜色标注，把帧按光色分组
  -> 统计每个光色下 ROI 的均值、方差、变化率
  -> 显式输出光色响应特征
```

训练融合方式：

```text
color-aware frame feature + flash response embedding -> Transformer/MLP
```

### 3. 屏幕摩尔纹和高频纹理

屏幕 replay 攻击通常经过二次成像：屏幕像素网格、刷新频率、摄像头采样频率和压缩共同作用，容易产生摩尔纹、高频条纹、局部彩边和频域异常。

可加入特征：

```text
fft_energy_high_freq        高频能量比例
moire_band_energy           特定频带能量
laplacian_texture_strength  局部锐度/纹理强度
screen_scanline_score       横向或纵向周期纹得分
compression_artifact_score  块状压缩伪影强度
```

建议在 V2 中新增：

```text
FrequencyArtifactExtractor
  -> 对每帧或关键 ROI 做 FFT/DCT/Laplacian
  -> 输出逐帧频域特征
```

训练融合方式：

```text
RGB/diff CNN feature + frequency embedding -> Transformer
```

### 4. 3D 深度和法线一致性

真人脸是连续三维曲面，鼻梁、眼窝、脸颊、下巴具有自然深度变化。打印照片接近平面，屏幕 replay 也是平面显示，头模虽然有 3D 结构，但皮肤材质和微表面响应仍不真实。

可加入特征：

```text
depth_map                单目深度估计
normal_map               法线图
face_depth_variance      人脸区域深度方差
nose_cheek_depth_ratio   鼻梁和脸颊深度关系
planarity_score          平面性得分
temporal_depth_stability 深度时序稳定性
```

建议在 V2 中新增：

```text
DepthGeometryExtractor
  -> 可先使用已有深度估计模型离线生成 depth/normal
  -> 或训练时读取预生成 depth/normal 文件
```

训练输入可以扩展为：

```text
原始 V2:
B x T x 6 x 224 x 224

加入 depth:
B x T x 7 x 224 x 224

加入 depth + normal:
B x T x 10 x 224 x 224

加入 depth + normal + 光响应图:
B x T x C x 224 x 224
```

## 推荐的代码改造路径

### 阶段一：不破坏当前 V2，增加可解释特征日志

目标是先验证这些物理线索是否真的有区分度，不急着改变模型。

建议新增脚本：

```text
scripts/analyze_flash_physical_cues.py
```

功能：

```text
1. 读取 live/spoof 视频和对应 txt。
2. 按光色分组统计 RGB 响应。
3. 提取脸颊/额头区域的绿色通道时序。
4. 计算 FFT 主频和频域能量。
5. 统计帧间差分强度和光色切换响应。
6. 输出 csv/json/可视化曲线。
```

产物：

```text
physical_cues_summary.csv
rppg_curves/*.png
flash_response_curves/*.png
frequency_spectrum/*.png
```

这样可以先回答一个关键问题：当前数据里，真人和攻击样本在哪些物理特征上真的分得开。

### 阶段二：扩展 Dataset，返回辅助特征

当前 `FlashLivenessDataset.__getitem__` 返回：

```python
return tensor_frames, tensor_colors, torch.tensor(label, dtype=torch.float32)
```

可以扩展为：

```python
return tensor_frames, tensor_colors, tensor_physical_features, torch.tensor(label, dtype=torch.float32)
```

其中 `tensor_physical_features` 可以是：

```text
T x C_phys
```

例如：

```text
C_phys = [
  mean_r,
  mean_g,
  mean_b,
  cheek_g,
  forehead_g,
  diff_energy,
  high_freq_energy,
  flash_response_delta,
  transition_decay_score
]
```

### 阶段三：扩展模型融合层

当前模型融合方式：

```python
features = cnn_backbone(frame_6ch)
color_emb = color_proj(color_features)
features = pos_encoder(features + color_emb)
trans_out = transformer(features)
```

可以改为：

```python
features = cnn_backbone(frame_6ch)
color_emb = color_proj(color_features)
phys_emb = physical_proj(physical_features)
features = pos_encoder(features + color_emb + phys_emb)
trans_out = transformer(features)
```

新增：

```python
self.physical_proj = nn.Sequential(
    nn.Linear(C_phys, embed_dim),
    nn.LayerNorm(embed_dim),
    nn.ReLU(),
)
```

模型 forward 改为：

```python
def forward(self, x, color_features, physical_features=None, padding_mask=None):
    ...
```

这样不会推翻当前 V2，而是在同一条时序建模链路中加入物理先验。

### 阶段四：增加多任务辅助监督

如果后续能得到更多标注，可以让模型不仅判断 live/spoof，还学习辅助任务：

```text
主任务: live/spoof BCE loss
辅助任务1: attack type classification，print/replay/mask/head_model
辅助任务2: depth consistency score
辅助任务3: rPPG quality score
辅助任务4: flash response consistency score
```

总损失：

```text
loss = loss_live_spoof
     + lambda_rppg * loss_rppg_quality
     + lambda_depth * loss_depth_consistency
     + lambda_attack * loss_attack_type
```

这能让模型学到更稳定的物理中间表征，而不是只依赖某个数据集里的背景或设备偏差。

## 显式物理模块代码模板

下面是一份可以在训练结束后逐步接入 V2 的代码模板。它不是为了立刻替换当前 baseline，而是说明每个专门模块应该如何实现、如何接入当前 V2、以及接入后如何生效。

为了先验证这些线索是否真的有区分度，已新增一个独立分析脚本：

```text
/supercloud/llm-code/scc/scc/Liveness_Detection/scripts/analyze_flash_physical_cues.py
```

该脚本不调用训练模型、不修改 V2 训练代码，只做三件事：

```text
1. 逐视频提取 rPPG、闪光响应/SSS、高频摩尔纹等显式物理特征。
2. 输出每个视频的一行特征记录。
3. 按 live/spoof 标签统计每个特征的区分度，包含 AUC、Cohen's d、最佳平衡准确率。
```

推荐正式运行命令：

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python \
  /supercloud/llm-code/scc/scc/Liveness_Detection/scripts/analyze_flash_physical_cues.py \
  --input-root /supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_video_dataset_v2/test \
  --output-dir /supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/physical_cue_analysis_v2_test \
  --frame-stride 1 \
  --max-frames 0 \
  --max-frequency-frames 96
```

如果全量数据较慢，可以先抽样验证：

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python \
  /supercloud/llm-code/scc/scc/Liveness_Detection/scripts/analyze_flash_physical_cues.py \
  --input-root /supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_video_dataset_v2/test \
  --output-dir /supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/physical_cue_analysis_v2_smoke \
  --samples-per-class 10 \
  --max-frames 128 \
  --max-frequency-frames 32
```

如果要固定使用 1 个真人视频和 1 个头模/3D 攻击视频做 smoke test 对比，可以使用：

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python \
  /supercloud/llm-code/scc/scc/Liveness_Detection/scripts/analyze_flash_physical_cues.py \
  --input-root /supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_video_dataset_v2/test \
  --output-dir /supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/physical_cue_live_vs_headmodel_smoke \
  --smoke-live-headmodel-pair \
  --max-frames 128 \
  --max-frequency-frames 32
```

该模式会自动选择：

```text
1 个 live 目录下的真人视频。
1 个 spoof 目录下文件名包含 toumo/headmodel/head_model/head/mask/3dfake 的头模或 3D 攻击视频。
```

输出文件：

```text
physical_cues_per_video.csv
  每个视频一行，记录 rPPG、闪光响应、频域纹理等特征。

physical_cues_feature_separation.csv
  每个特征一行，记录 live/spoof 均值、AUC、Cohen's d、最佳阈值和平衡准确率。

physical_cues_summary.json
  汇总样本数、成功/失败数量、Top 区分特征。

physical_cues_key_meanings.md
  用 Markdown 表格解释每个 key 的含义，适合直接查看。

physical_cues_key_meanings.csv
  用 CSV 格式解释每个 key 的含义，适合表格软件查看。

physical_cues_key_meanings.json
  用 JSON 格式解释每个 key 的含义，适合程序读取。
```

结果解读：

```text
separability_auc 越接近 1，说明该特征越能区分 live/spoof。
direction=live_higher 表示真人该特征更大。
direction=live_lower 表示真人该特征更小。
best_balanced_accuracy 是单个特征用最佳阈值做二分类时的平衡准确率。
```

注意：小样本 smoke test 的 AUC 可能虚高，只能证明流程跑通。真正判断模块是否有用，应至少在 test 或 val 全量上看 `physical_cues_feature_separation.csv`。

建议新建文件：

```text
/supercloud/llm-code/scc/scc/Liveness_Detection/flash_physical_features.py
```

然后在 `flash_liveness_project_v2.py` 中导入这些模块。

### 1. rPPG 绿色通道时序模块

作用：

```text
显式提取脸部区域随时间变化的绿色通道波动、帧间变化幅度、频谱主峰和频谱质量。
```

为什么有效：

```text
真人皮下血液容积会随心跳周期变化，绿光对血红蛋白吸收敏感。
打印照片、头模、屏幕 replay 通常缺少稳定的真实血流周期。
```

示例代码：

```python
import cv2
import numpy as np


class RPPGFeatureExtractor:
    """从人脸序列中提取简单 rPPG 统计特征。

    输入 frames_rgb:
        T x H x W x 3, float32, range [0, 1]

    输出:
        T x 6
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def _fft_peak_score(self, signal: np.ndarray) -> float:
        if len(signal) < 8:
            return 0.0
        centered = signal - signal.mean()
        spectrum = np.abs(np.fft.rfft(centered))
        if len(spectrum) <= 2:
            return 0.0
        spectrum[0] = 0.0
        peak = float(spectrum.max())
        total = float(spectrum.sum()) + self.eps
        return peak / total

    def __call__(self, frames_rgb: np.ndarray) -> np.ndarray:
        if frames_rgb.size == 0:
            return np.zeros((0, 6), dtype=np.float32)

        height = frames_rgb.shape[1]
        mean_g = frames_rgb[..., 1].mean(axis=(1, 2))
        delta_g = np.zeros_like(mean_g)
        if len(mean_g) > 1:
            delta_g[1:] = mean_g[1:] - mean_g[:-1]

        forehead = frames_rgb[:, : height // 3, :, 1]
        cheek = frames_rgb[:, height // 2 :, :, 1]
        forehead_g = forehead.mean(axis=(1, 2))
        cheek_g = cheek.mean(axis=(1, 2))

        local_energy = np.zeros_like(mean_g)
        for idx in range(len(mean_g)):
            left = max(idx - 2, 0)
            right = min(idx + 3, len(mean_g))
            local_energy[idx] = np.std(mean_g[left:right])

        fft_peak_score = np.full_like(mean_g, self._fft_peak_score(mean_g))
        features = np.stack(
            [mean_g, delta_g, cheek_g, forehead_g, local_energy, fft_peak_score],
            axis=1,
        )
        return features.astype(np.float32)
```

在当前 V2 中如何添加：

```python
# FlashLivenessDataset.__init__
self.rppg_extractor = RPPGFeatureExtractor()

# FlashLivenessDataset.process_video
frames = np.asarray(frames_rgb, dtype=np.float32) / 255.0
rppg_features = self.rppg_extractor(frames)
```

如何生效：

```text
rPPG 特征会作为 T x C_phys 的一部分输入模型。
Transformer 在时间维度上看到绿色通道周期性、局部能量、频谱质量。
如果真人存在稳定血流微波动，而头模/打印/屏幕缺失该模式，模型可以学习到更稳定的生理差异。
```

### 2. 闪光响应与 SSS 次表面散射模块

作用：

```text
利用 txt 中逐帧颜色标注，把人脸对紫红、绿、红三种光的响应显式转成特征。
```

示例代码：

```python
class FlashResponseFeatureExtractor:
    """提取光色响应和光色切换迟滞特征。

    输入:
        frames_rgb: T x H x W x 3, float32, range [0, 1]
        color_values: T, int

    输出:
        T x 8
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def __call__(self, frames_rgb: np.ndarray, color_values: np.ndarray) -> np.ndarray:
        if frames_rgb.size == 0:
            return np.zeros((0, 8), dtype=np.float32)

        rgb_mean = frames_rgb.mean(axis=(1, 2))
        intensity = rgb_mean.mean(axis=1)

        delta_intensity = np.zeros_like(intensity)
        if len(intensity) > 1:
            delta_intensity[1:] = intensity[1:] - intensity[:-1]

        transition = np.zeros_like(intensity)
        if len(color_values) > 1:
            transition[1:] = color_values[1:] != color_values[:-1]

        response_decay = np.zeros_like(intensity)
        for idx in range(1, len(intensity)):
            if transition[idx] > 0:
                response_decay[idx] = abs(delta_intensity[idx])
            else:
                response_decay[idx] = response_decay[idx - 1] * 0.8

        red_over_green = rgb_mean[:, 0] / (rgb_mean[:, 1] + self.eps)
        blue_over_green = rgb_mean[:, 2] / (rgb_mean[:, 1] + self.eps)
        chroma_range = rgb_mean.max(axis=1) - rgb_mean.min(axis=1)

        features = np.stack(
            [
                rgb_mean[:, 0],
                rgb_mean[:, 1],
                rgb_mean[:, 2],
                intensity,
                delta_intensity,
                transition,
                response_decay,
                red_over_green + blue_over_green + chroma_range,
            ],
            axis=1,
        )
        return features.astype(np.float32)
```

在当前 V2 中如何添加：

```python
# _read_all_frames_with_color 建议同时返回 raw color_values
frames_bgr, color_features, color_values = self._read_all_frames_with_color(video_path, txt_path)

# process_video 中
flash_features = self.flash_response_extractor(frames, np.asarray(color_values))
```

如何生效：

```text
模型直接得到当前光色下的反射强度、光色切换、切换后的响应衰减、红绿/蓝绿响应比例。
这些特征能帮助区分真实皮肤、硅胶、纸张和屏幕发光材料。
```

### 3. 屏幕摩尔纹和高频纹理模块

作用：

```text
提取 replay 攻击中常见的屏幕像素网格、扫描线、高频条纹和压缩伪影。
```

示例代码：

```python
class FrequencyArtifactExtractor:
    """提取每帧频域和边缘纹理特征。

    输入:
        frames_rgb: T x H x W x 3, float32, range [0, 1]

    输出:
        T x 5
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def _frame_features(self, frame_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor((frame_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_f = gray.astype(np.float32) / 255.0

        fft = np.fft.fftshift(np.fft.fft2(gray_f))
        mag = np.log1p(np.abs(fft))
        h, w = mag.shape
        cy, cx = h // 2, w // 2

        yy, xx = np.ogrid[:h, :w]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        high_mask = radius > min(h, w) * 0.25
        mid_mask = (radius > min(h, w) * 0.10) & (radius <= min(h, w) * 0.25)

        high_energy = float(mag[high_mask].mean())
        mid_energy = float(mag[mid_mask].mean())
        total_energy = float(mag.mean()) + self.eps

        lap = cv2.Laplacian(gray_f, cv2.CV_32F)
        lap_var = float(lap.var())

        row_profile = gray_f.mean(axis=1)
        col_profile = gray_f.mean(axis=0)
        row_periodicity = float(np.std(np.diff(row_profile)))
        col_periodicity = float(np.std(np.diff(col_profile)))

        return np.asarray(
            [
                high_energy / total_energy,
                mid_energy / total_energy,
                lap_var,
                row_periodicity,
                col_periodicity,
            ],
            dtype=np.float32,
        )

    def __call__(self, frames_rgb: np.ndarray) -> np.ndarray:
        if frames_rgb.size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.stack([self._frame_features(frame) for frame in frames_rgb], axis=0)
```

在当前 V2 中如何添加：

```python
# FlashLivenessDataset.__init__
self.frequency_extractor = FrequencyArtifactExtractor()

# FlashLivenessDataset.process_video
frequency_features = self.frequency_extractor(frames)
```

如何生效：

```text
模型获得每一帧的高频能量、周期纹、边缘异常强度。
对于屏幕播放攻击，Transformer 可以结合时序判断这些纹理是否持续存在或随闪光异常变化。
```

### 4. Depth / Normal 几何模块

作用：

```text
显式引入单目深度图和法线图，帮助模型判断人脸是否具有合理的 3D 几何结构。
```

推荐先做离线生成，不建议在 DataLoader 里实时跑深度模型，否则训练速度会明显下降。

示例读取代码：

```python
class DepthNormalFeatureLoader:
    """读取预生成 depth/normal，并转成逐帧统计特征。

    depth: T x H x W
    normal: T x H x W x 3
    输出: T x 6
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def _match_length(self, array: np.ndarray, target_len: int) -> np.ndarray:
        if len(array) == target_len:
            return array
        if len(array) == 0:
            shape = (target_len,) + array.shape[1:]
            return np.zeros(shape, dtype=np.float32)
        indices = np.linspace(0, len(array) - 1, target_len).astype(np.int64)
        return array[indices]

    def __call__(self, depth_path: str, normal_path: str, target_len: int) -> np.ndarray:
        if not depth_path or not normal_path:
            return np.zeros((target_len, 6), dtype=np.float32)

        depth = np.load(depth_path).astype(np.float32)
        normal = np.load(normal_path).astype(np.float32)
        depth = self._match_length(depth, target_len)
        normal = self._match_length(normal, target_len)

        depth_mean = depth.mean(axis=(1, 2))
        depth_std = depth.std(axis=(1, 2))
        depth_range = depth.max(axis=(1, 2)) - depth.min(axis=(1, 2))
        normal_mean = normal.mean(axis=(1, 2))
        normal_strength = np.linalg.norm(normal_mean, axis=1)
        planarity_score = 1.0 / (depth_std + self.eps)

        features = np.stack(
            [
                depth_mean,
                depth_std,
                depth_range,
                normal_strength,
                planarity_score,
                depth_std / (depth_mean + self.eps),
            ],
            axis=1,
        )
        return features.astype(np.float32)
```

在当前 V2 中如何添加：

```python
# collect_samples_from_label_dirs 可以从 3 元组扩展到 5 元组:
# (video_path, txt_path, depth_path, normal_path, label)

# FlashLivenessDataset.__getitem__
video_path, txt_path, depth_path, normal_path, label = self.samples[idx]

# process_video 返回 frames/colors 后
depth_features = self.depth_loader(depth_path, normal_path, target_len=tensor_frames.shape[0])
```

如何生效：

```text
模型获得每帧的深度均值、深度方差、平面性、法线强度等几何线索。
打印照片/屏幕 replay 的 planarity_score 往往更高，真实人脸 depth_std 和局部曲面变化更自然。
```

### 5. 合并物理特征并接入 Dataset

统一提取器：

```python
class PhysicalCueExtractor:
    """统一提取 V2 物理先验特征。

    第一版推荐 C_phys = 19:
        rPPG 6维 + flash response 8维 + frequency artifact 5维

    加 depth/normal 后 C_phys = 25。
    """

    def __init__(self, use_depth: bool = False) -> None:
        self.rppg = RPPGFeatureExtractor()
        self.flash = FlashResponseFeatureExtractor()
        self.frequency = FrequencyArtifactExtractor()
        self.depth = DepthNormalFeatureLoader() if use_depth else None

    def __call__(
        self,
        frames_rgb: np.ndarray,
        color_values: np.ndarray,
        depth_path: str | None = None,
        normal_path: str | None = None,
    ) -> np.ndarray:
        features = [
            self.rppg(frames_rgb),
            self.flash(frames_rgb, color_values),
            self.frequency(frames_rgb),
        ]

        if self.depth is not None:
            features.append(self.depth(depth_path or "", normal_path or "", len(frames_rgb)))

        return np.concatenate(features, axis=1).astype(np.float32)
```

接入当前 V2：

```python
# FlashLivenessDataset.__init__
self.physical_extractor = PhysicalCueExtractor(use_depth=False)

# FlashLivenessDataset.process_video
physical_features = self.physical_extractor(frames, np.asarray(color_values))
tensor_physical = torch.from_numpy(physical_features).float()
return tensor_frames, tensor_colors, tensor_physical

# FlashLivenessDataset.__getitem__
return tensor_frames, tensor_colors, tensor_physical, torch.tensor(label, dtype=torch.float32)
```

### 6. 修改 collate，使物理特征支持 padding

加入物理特征后 batch 结构：

```text
videos: B x T x 6 x H x W
colors: B x T x 4
physical_features: B x T x C_phys
padding_mask: B x T
labels: B
```

示例代码：

```python
def collate_skip_none_with_physical(batch):
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return None

    max_len = max(item[0].shape[0] for item in valid_batch)
    channels, height, width = valid_batch[0][0].shape[1:]
    color_dim = valid_batch[0][1].shape[1]
    phys_dim = valid_batch[0][2].shape[1]

    videos = torch.zeros((len(valid_batch), max_len, channels, height, width), dtype=torch.float32)
    colors = torch.zeros((len(valid_batch), max_len, color_dim), dtype=torch.float32)
    physical = torch.zeros((len(valid_batch), max_len, phys_dim), dtype=torch.float32)
    padding_mask = torch.ones((len(valid_batch), max_len), dtype=torch.bool)
    labels = torch.stack([item[3] for item in valid_batch])

    for idx, (video_tensor, color_tensor, physical_tensor, _) in enumerate(valid_batch):
        seq_len = video_tensor.shape[0]
        videos[idx, :seq_len] = video_tensor
        colors[idx, :seq_len] = color_tensor
        physical[idx, :seq_len] = physical_tensor
        padding_mask[idx, :seq_len] = False

    return videos, colors, physical, padding_mask, labels
```

如何生效：

```text
不同长度视频仍按 batch 内最长视频 padding。
padding_mask 继续告诉 Transformer 哪些是真实帧、哪些是补齐帧。
物理特征和每帧图像严格对齐，同步进入模型。
```

### 7. 修改模型，融合显式物理特征

当前 V2 模型融合：

```python
features = self.cnn_backbone(x_cnn)
color_emb = self.color_proj(color_features)
features = self.pos_encoder(features + color_emb)
trans_out = self.transformer(features, src_key_padding_mask=padding_mask)
```

加入物理特征后：

```python
class CNNTransformerLivenessWithPhysical(nn.Module):
    def __init__(self, embed_dim: int = 512, physical_dim: int = 19) -> None:
        super().__init__()
        resnet = models.resnet18(weights=None)
        original_conv1 = resnet.conv1
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_backbone[0] = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            self.cnn_backbone[0].weight[:, :3, :, :] = original_conv1.weight
            self.cnn_backbone[0].weight[:, 3:, :, :] = original_conv1.weight

        self.color_proj = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.physical_proj = nn.Sequential(
            nn.Linear(physical_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.3,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        color_features: torch.Tensor,
        physical_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = x.size()
        x_cnn = x.view(batch_size * num_frames, channels, height, width)
        features = self.cnn_backbone(x_cnn)
        features = features.view(batch_size, num_frames, -1)

        color_emb = self.color_proj(color_features)
        physical_emb = self.physical_proj(physical_features)
        features = self.pos_encoder(features + color_emb + physical_emb)
        trans_out = self.transformer(features, src_key_padding_mask=padding_mask)

        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            pooled_out = (trans_out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled_out = trans_out.mean(dim=1)

        return self.fc(pooled_out).squeeze(-1)
```

如何生效：

```text
CNN 分支负责看图像外观和帧差。
color_proj 负责告诉模型当前是哪种闪光颜色。
physical_proj 负责显式注入 rPPG、SSS、摩尔纹、深度几何这些物理判据。
Transformer 负责学习这些线索在完整视频时序中的组合模式。
```

### 8. 修改训练循环

当前 batch：

```python
batch_videos, batch_colors, batch_padding_mask, batch_labels = batch
outputs = model(batch_videos, batch_colors, batch_padding_mask)
```

加入物理特征后：

```python
batch_videos, batch_colors, batch_physical, batch_padding_mask, batch_labels = batch

batch_videos = batch_videos.to(device, non_blocking=non_blocking)
batch_colors = batch_colors.to(device, non_blocking=non_blocking)
batch_physical = batch_physical.to(device, non_blocking=non_blocking)
batch_padding_mask = batch_padding_mask.to(device, non_blocking=non_blocking)
batch_labels = batch_labels.to(device, non_blocking=non_blocking)

outputs = model(batch_videos, batch_colors, batch_physical, batch_padding_mask)
loss = criterion(outputs, batch_labels)
```

如何生效：

```text
训练目标仍然是 live/spoof BCE。
但模型获得了更明确的物理解释特征，因此不必完全依赖 CNN 从像素中自行学习这些细微信号。
```

### 9. 推荐实际落地顺序

不要一次性把所有模块都加入正式训练。推荐按下面顺序做：

```text
第一版:
Baseline V2 + FlashResponseFeatureExtractor

第二版:
Baseline V2 + FlashResponseFeatureExtractor + RPPGFeatureExtractor

第三版:
Baseline V2 + FlashResponseFeatureExtractor + RPPGFeatureExtractor + FrequencyArtifactExtractor

第四版:
Baseline V2 + FlashResponseFeatureExtractor + RPPGFeatureExtractor + FrequencyArtifactExtractor + DepthNormalFeatureLoader
```

每加入一个模块，都记录：

```text
1. train/val/test accuracy
2. APCER
3. BPCER
4. ACER
5. AUC
6. EER
7. 对打印照片、屏幕 replay、头模分别的错误样本
```

如果某个模块让训练集提升但测试集下降，说明它可能学到了采集设备、背景或压缩格式偏差，需要重新检查数据划分和跨设备泛化。

## 建议优先级

第一优先级：

```text
光色响应特征 + 光色切换迟滞曲线
```

原因：当前项目本身就是闪光活体，txt 已经有每帧颜色标注，改造成本最低，和业务目标最一致。

第二优先级：

```text
rPPG 绿色通道时序 + 频谱质量
```

原因：真人和硅胶头模、打印照片之间通常有生理差异，但需要足够长、稳定、少运动的视频片段。

第三优先级：

```text
屏幕摩尔纹/高频纹理检测
```

原因：对 replay 攻击很有用，但容易受摄像头、压缩、分辨率影响，需要做跨设备验证。

第四优先级：

```text
depth/normal 几何分支
```

原因：对打印照片、屏幕平面攻击有效，但对 3D 头模不一定足够，需要和材质/光响应联合判断。

## 训练结束后的实验建议

训练完成当前 V2 后，建议按下面顺序做 ablation：

```text
Baseline V2:
RGB + diff + color + Transformer

Experiment A:
Baseline + flash response features

Experiment B:
Baseline + rPPG features

Experiment C:
Baseline + frequency artifact features

Experiment D:
Baseline + depth/normal features

Experiment E:
Baseline + flash response + rPPG + frequency + depth
```

评估时不要只看 accuracy，还要重点看：

```text
APCER: 攻击样本被误判为真人的比例，越低越好。
BPCER: 真人样本被误判为攻击的比例，越低越好。
ACER: APCER 和 BPCER 平均值，越低越好。
AUC: 分数排序能力，越高越好。
EER: 等错误率，越低越好。
```

对于物理攻击课题，最关键的是 APCER。因为攻击样本被放行为真人，风险最高。

## 结论

当前 V2 已经具备一定的物理攻击识别基础：它保留全帧时序、显式输入闪光颜色、使用帧间差分表达光照变化，并用 Transformer 建模时间关系。

但当前 V2 还没有显式实现 rPPG、SSS、摩尔纹、高频频域、depth/normal 等物理先验模块。下一步最稳妥的优化方式不是直接大改模型，而是先新增物理线索分析脚本，验证这些特征在当前数据集上的区分度；确认有效后，再把光色响应、rPPG、频域纹理和深度几何作为辅助特征分支融合进 V2。
