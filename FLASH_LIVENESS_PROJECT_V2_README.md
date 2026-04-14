# Flash Liveness Project V2 说明

`flash_liveness_project_v2.py` 是闪光活体检测训练与推理脚本的 V2 版本。相比 V1 固定抽取 `16` 帧，V2 的主要变化是顺序读取视频全部可解码帧，并读取同名 `.txt` 中记录的逐帧闪光颜色，把颜色信息作为额外时序特征输入模型。

核心目标是让模型不只看到人脸图像本身，还能知道“当前帧是什么闪光颜色、是否刚发生颜色切换”，从而更好地区分真人皮肤、屏幕翻拍、纸张、头模等介质在闪光变化下的响应差异。

## 模型结构图

### V2 结构图

![Flash Liveness V2 Architecture](assets/flash_liveness_v2_architecture.svg)

### 对照参考：V1 结构图

![Flash Liveness V1 Architecture](assets/flash_liveness_v1_architecture.svg)

## 核心数据流

V2 的完整处理链路如下：

```text
视频文件 + 同名 txt 颜色文件
        |
顺序读取全部视频帧
        |
读取每一帧对应的闪光颜色
        |
人脸检测/对齐，失败则中心裁剪
        |
resize 到 224x224
        |
BGR 转 RGB，归一化到 [0, 1]
        |
计算相邻帧差分 diff
        |
拼接 6 通道图像特征: RGB 3通道 + diff 3通道
        |
将颜色整数转成 [r, g, b, transition]
        |
DataLoader 对不同长度视频做 padding
        |
ResNet18 提取逐帧图像特征
        |
color_proj 提取逐帧颜色特征
        |
图像特征 + 颜色特征 + 位置编码
        |
Transformer Encoder 建模时序关系
        |
只平均有效帧，忽略 padding
        |
BCE 二分类输出 live/spoof
```

模型输入包含两部分：

```text
video tensor: [B, T, 6, H, W]
color tensor: [B, T, 4]
```

其中：

```text
B = batch size
T = 当前视频帧数，V2 中不同视频可以不同
6 = RGB 3通道 + 帧间差分 3通道
4 = 闪光颜色 r,g,b + transition 切换标记
```

其中 V2 的 `color tensor` 来自同名 `.txt`，每一帧都会被编码成：

```text
[r, g, b, transition]
```

含义是：

- `r, g, b`: 当前帧对应闪光颜色的归一化 RGB 值
- `transition`: 当前帧颜色是否相对上一帧发生切换，未切换为 `0`，切换为 `1`

而 `video tensor` 的 6 个通道仍然与 V1 一致：

```text
[R, G, B, dR, dG, dB]
```

也就是：

- 前 3 个通道：当前帧 RGB
- 后 3 个通道：相邻帧差分

所以 V2 不是替换掉 V1 的视觉输入，而是在 V1 的视觉时序建模基础上，再显式加入“闪光协议颜色信息”。

## DataLoader 如何对不同长度视频做 padding

V2 和 V1 的一个本质差异是：V2 不把所有视频强行截成固定 `16` 帧，而是顺序读取全部可解码帧。因此，同一个 batch 里的不同视频长度通常不一样，不能直接用普通 `torch.stack(...)` 叠起来。

V2 的 padding 逻辑在 [`flash_liveness_project_v2.py`](/supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_project_v2.py) 中由 [`collate_skip_none(...)`](/supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_project_v2.py#L796) 完成。

每个样本在 `__getitem__` 阶段返回的是：

```text
video_tensor: [T, 6, 224, 224]
color_tensor: [T, 4]
label: []
```

其中 `T` 是当前视频的真实有效帧数，不同视频可以不同。

padding 的具体步骤如下：

1. 先找出当前 batch 里最长的视频长度

```python
max_len = max(item[0].shape[0] for item in valid_batch)
```

如果一个 batch 中 3 个视频长度分别是：

```text
120, 95, 143
```

那么当前 batch 会以：

```text
max_len = 143
```

为统一长度。

2. 预先创建全 0 的大张量

```python
videos = torch.zeros((B, max_len, channels, height, width), dtype=torch.float32)
colors = torch.zeros((B, max_len, color_dim), dtype=torch.float32)
padding_mask = torch.ones((B, max_len), dtype=torch.bool)
```

也就是先得到：

```text
videos:       [B, max_len, 6, 224, 224]
colors:       [B, max_len, 4]
padding_mask: [B, max_len]
```

初始化时：

- `videos` 全 0
- `colors` 全 0
- `padding_mask` 全 `True`

这里 `True` 表示“这个时间位置目前是 padding，不是真实帧”。

3. 把每个视频自己的真实帧拷贝到前半段

```python
for idx, (video_tensor, color_tensor, _) in enumerate(valid_batch):
    seq_len = video_tensor.shape[0]
    videos[idx, :seq_len] = video_tensor
    colors[idx, :seq_len] = color_tensor
    padding_mask[idx, :seq_len] = False
```

也就是说：

- 真实帧放在前面
- 后面不足的部分保持为 0
- 真实帧位置标成 `False`
- padding 位置保持 `True`

4. 返回给模型的是 4 个量

```text
videos, colors, padding_mask, labels
```

其 shape 分别是：

```text
videos:       [B, max_len, 6, 224, 224]
colors:       [B, max_len, 4]
padding_mask: [B, max_len]
labels:       [B]
```

5. Transformer 和池化都会显式忽略 padding

在模型前向中：

```python
trans_out = self.transformer(features, src_key_padding_mask=padding_mask)
```

说明 Transformer 在时序建模时就已经知道哪些位置是补齐帧。

后面在时间池化时，又进一步只统计有效帧：

```python
valid_mask = (~padding_mask).unsqueeze(-1).float()
pooled_out = (trans_out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
```

这意味着：

- padding 帧不会参与 Transformer 的有效时序建模
- padding 帧也不会参与最终平均池化

所以 V2 的 padding 不是“简单补 0 然后假装它是真实帧”，而是“补 0 + 显式 mask 掉”，这是可变长视频输入能稳定训练的关键。

一个直观例子如下：

假设一个 batch 里有两个视频：

```text
Video A: T = 5
Video B: T = 8
```

padding 后：

```text
videos.shape       = [2, 8, 6, 224, 224]
colors.shape       = [2, 8, 4]
padding_mask.shape = [2, 8]
```

对应的 `padding_mask` 可以理解成：

```text
Video A: [False, False, False, False, False, True,  True,  True]
Video B: [False, False, False, False, False, False, False, False]
```

其中：

- `False` = 真实帧
- `True` = padding 补齐帧

因此，V2 的 DataLoader padding 机制可以概括为：

```text
按 batch 内最长视频长度补齐
视频张量和颜色张量都补 0
padding_mask 显式标记哪些位置是补齐
Transformer 和最终池化都忽略补齐位置
```

## 与 V1 的主要区别

| 项目 | V1 | V2 |
| --- | --- | --- |
| 帧读取方式 | 从视频中采样固定 `num_frames` 帧 | 顺序读取全部可解码帧 |
| 输入长度 | 固定长度，默认 16 帧 | 变长序列 |
| DataLoader | 直接 stack | padding 到 batch 内最大帧数 |
| mask | 不需要 | 使用 `padding_mask` 忽略补齐帧 |
| 颜色信息 | 没有显式输入 | 读取同名 `.txt` 逐帧颜色 |
| 模型额外分支 | 无 | `color_proj`: `[r,g,b,transition] -> embed_dim` |
| 推理要求 | 视频路径即可 | 视频路径 + 同名颜色 `.txt`，或手动传 `--txt-path` |

## 数据目录格式

推荐目录结构：

```text
dataset/flash_liveness_video_dataset_v2/
  train/
    live/
      sample_001.mp4
      sample_001.txt
    spoof/
      sample_002.mp4
      sample_002.txt
  val/
    live/
    spoof/
  test/
    live/
    spoof/
```

也可以只提供：

```text
data_root/
  live/
  spoof/
```

脚本会按 `--val-ratio` 和 `--test-ratio` 自动分层切分。

注意：V2 扫描样本时要求视频旁边存在同名 `.txt` 文件，否则该视频不会加入样本列表。

例如：

```text
abc.mp4
abc.txt
```

支持的视频后缀：

```text
.mp4, .avi, .mov, .mkv, .webm, .m4v
```

支持的正类目录名：

```text
live, real, bonafide, bona_fide, genuine, 真人
```

支持的负类目录名：

```text
spoof, fake, attack, imposter, print, replay, mask, 头模, 攻击
```

## 颜色 txt 格式

每个视频需要有一个同名 `.txt` 文件，记录逐帧闪光颜色。

支持两种格式。

格式一：显式帧号和颜色值

```text
0,16711680
1,16711680
2,65280
3,255
```

格式二：每行只有颜色值，帧号按行号自动递增

```text
16711680
16711680
65280
255
```

颜色值按整数 RGB 解析：

```text
0xRRGGBB
```

解析逻辑：

```python
r = ((color_value >> 16) & 0xFF) / 255.0
g = ((color_value >> 8) & 0xFF) / 255.0
b = (color_value & 0xFF) / 255.0
```

额外会生成一个 `transition` 特征：

```text
当前帧颜色与上一帧不同 => transition = 1.0
当前帧颜色与上一帧相同 => transition = 0.0
第一帧或无上一帧       => transition = 0.0
```

因此每帧颜色特征为：

```text
[r, g, b, transition]
```

## 图像预处理

每帧视频图像先经过 `FacePreprocessor`：

1. 如果提供 `--detector-model`，使用 YOLOv7 人脸检测/对齐。
2. 如果没有检测模型，使用中心裁剪。
3. 如果检测器加载失败或推理失败，也回退中心裁剪。
4. 输出统一 resize 到 `--image-size x --image-size`，默认 `224x224`。

预处理后：

```text
BGR -> RGB
像素值 / 255.0
计算 diff_frames
拼接 RGB + diff
```

最终每帧图像特征为：

```text
[6, 224, 224]
```

## Dataset 逻辑

`FlashLivenessDataset` 的每个样本格式是：

```python
(video_path, txt_path, label)
```

`label` 约定：

```text
1 = live / 真人
0 = spoof / 假人、攻击
```

读取一个样本时：

1. 用 OpenCV 顺序读取全部视频帧。
2. 用同名 `.txt` 解析逐帧颜色。
3. 如果某一帧没有颜色记录，沿用上一帧颜色；如果上一帧也没有，则使用 `0`。
4. 对每一帧做人脸裁剪和 resize。
5. 计算 RGB + diff 的 6 通道图像 tensor。
6. 计算 `[r,g,b,transition]` 颜色 tensor。
7. 返回：

```python
tensor_frames, tensor_colors, label
```

其中：

```text
tensor_frames: [T, 6, 224, 224]
tensor_colors: [T, 4]
label: scalar float
```

训练时 `transform=True`，会以 50% 概率水平翻转 `tensor_frames`。

## 变长序列 padding

V2 读取全部帧，因此不同视频的 `T` 不一样。`collate_skip_none` 会把一个 batch 内的视频 padding 到最长视频长度。

输出 batch：

```text
videos:       [B, max_T, 6, H, W]
colors:       [B, max_T, 4]
padding_mask: [B, max_T]
labels:       [B]
```

`padding_mask` 的含义：

```text
False = 有效帧
True  = padding 帧
```

Transformer 会使用 `src_key_padding_mask=padding_mask` 忽略 padding 帧。最后做时序平均时，也只平均有效帧。

## 模型结构

模型类：

```python
CNNTransformerLiveness
```

结构：

```text
6通道帧图像
    |
ResNet18 backbone
    |
逐帧视觉特征 [B,T,embed_dim]

颜色特征 [B,T,4]
    |
color_proj: Linear + LayerNorm + ReLU
    |
逐帧颜色嵌入 [B,T,embed_dim]

视觉特征 + 颜色嵌入
    |
PositionalEncoding
    |
TransformerEncoder
    |
有效帧平均池化
    |
Linear classifier
    |
logit
```

ResNet18 第一层卷积被改成 6 通道输入：

```python
nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
```

初始化方式：

```text
前 3 通道使用原 ResNet conv1 权重
后 3 通道也复制原 ResNet conv1 权重
```

颜色分支：

```python
self.color_proj = nn.Sequential(
    nn.Linear(4, embed_dim),
    nn.LayerNorm(embed_dim),
    nn.ReLU(),
)
```

融合方式：

```python
features = visual_features + color_emb
features = pos_encoder(features)
```

## 训练命令

示例：

```bash
cd /supercloud/llm-code/scc/scc/Liveness_Detection

/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python flash_liveness_project_v2.py train \
  --data-root /supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_video_dataset_v2 \
  --output-dir /supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/flash_liveness_v2_run \
  --epochs 20 \
  --batch-size 2 \
  --num-workers 4 \
  --image-size 224 \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --log-interval 20 \
  --pos-weight auto \
  --device cuda:0 \
  --use-imagenet-pretrained \
  --imagenet-pretrained-path /supercloud/llm-code/scc/scc/Liveness_Detection/resnet18-f37072fd.pth
```

如果使用人脸检测：

```bash
--detector-model /supercloud/llm-code/scc/scc/Liveness_Detection/yolov7_face/yolov7-w6-face.pt \
--detector-device cuda:0
```

V2 中 `--num-frames` 已不再控制抽帧，脚本默认读取视频全部可解码帧。

## 续训

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python flash_liveness_project_v2.py train \
  --data-root /supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_video_dataset_v2 \
  --output-dir /supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/flash_liveness_v2_run \
  --resume-checkpoint /supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/flash_liveness_v2_run/last_flash_liveness_model.pth \
  --epochs 30 \
  --batch-size 2 \
  --num-workers 4 \
  --device cuda:0
```

注意：`--epochs` 是总 epoch 数，不是额外训练轮数。

如果 checkpoint 已经训练到第 20 轮，而你想再训练 10 轮，应设置：

```text
--epochs 30
```

## 单视频推理

如果视频旁边有同名 `.txt`：

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python flash_liveness_project_v2.py infer \
  --checkpoint /path/to/best_flash_liveness_model.pth \
  --video-path /path/to/input.mp4 \
  --device cuda:0
```

如果 `.txt` 不在同名路径，可以显式传入：

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python flash_liveness_project_v2.py infer \
  --checkpoint /path/to/best_flash_liveness_model.pth \
  --video-path /path/to/input.mp4 \
  --txt-path /path/to/input_color.txt \
  --device cuda:0
```

输出示例：

```json
{
  "video_path": "/path/to/input.mp4",
  "probability_live": 0.982311,
  "threshold": 0.512345,
  "prediction": "live"
}
```

判断规则：

```text
probability_live >= threshold => live
probability_live <  threshold => spoof
```

## 训练输出文件

训练输出目录中常见文件：

```text
best_flash_liveness_model.pth
last_flash_liveness_model.pth
run_config.json
metrics_history.csv
metrics_history.jsonl
training_summary.json
skipped_corrupted_samples.txt
```

含义：

```text
best_flash_liveness_model.pth      验证集 AUC 最优模型
last_flash_liveness_model.pth      最后一个 epoch 的模型
run_config.json                    训练命令、参数、类别数量
metrics_history.csv                每个 epoch 的指标表
metrics_history.jsonl              每个 epoch 的指标 JSONL
training_summary.json              最优模型对应的训练/验证/测试汇总
skipped_corrupted_samples.txt      无法解码或被跳过的视频
```

checkpoint 中保存：

```text
epoch
model_state_dict
optimizer_state_dict
threshold
config
resolved_pos_weight
best_val_auc
split_counts
train_metrics
val_metrics
test_metrics
```

## 指标含义

```text
accuracy = 总准确率
apcer    = 假人被判为真人的比例
bpcer    = 真人被判为假人的比例
acer     = (apcer + bpcer) / 2
auc      = ROC AUC
eer      = 等错误率
tp       = 真人判真人
tn       = 假人判假人
fp       = 假人判真人
fn       = 真人判假人
```

其中 `apcer` 对安全性很关键，表示攻击样本通过率。

## 关键参数

`--data-root`

数据根目录。

`--output-dir`

训练输出目录。

`--epochs`

训练总轮数。

`--batch-size`

批大小。V2 读取全部帧，显存压力比 V1 大，显存不足时应优先减小该值。

`--num-workers`

DataLoader worker 数。

`--image-size`

输入人脸尺寸，默认 `224`。

`--learning-rate`

学习率。

`--weight-decay`

AdamW 权重衰减。

`--pos-weight`

BCE 正类权重，默认 `auto`，按训练集中的 `spoof/live` 比例自动设置。

`--device`

运行设备，例如 `cuda:0`、`cuda:1`、`cpu`。

`--detector-model`

YOLOv7 人脸检测权重路径。不传则使用中心裁剪。

`--txt-path`

推理时使用的逐帧颜色 txt 路径。只在 `infer` 子命令中使用。

## 已知注意点与可优化方向

1. V2 的样本结构是三元组

   当前样本格式为：

   ```python
   (video_path, txt_path, label)
   ```

   如果某些统计函数仍按 V1 的二元组 `(video_path, label)` 解包，就会出错。尤其需要检查：

   ```text
   print_split_stats
   summarize_split_counts
   resolve_pos_weight
   ```

2. 视频过长时显存压力较大

   V2 默认读取全部帧，再在 batch 内 padding 到最长视频长度。如果某个视频很长，会明显增加 CPU 内存和 GPU 显存占用。

   可优化方向：

   ```text
   限制最大帧数
   按闪光阶段采样
   滑窗推理
   分段 Transformer
   ```

3. 位置编码最大长度为 512

   当前 `PositionalEncoding(max_len=512)`。如果视频有效帧超过 512，可能出现位置编码长度不够的问题。

4. txt 缺失时推理会失败

   V2 推理默认找视频同名 `.txt`。如果没有颜色文件，应增加更友好的错误提示，或提供默认颜色策略。

5. 颜色特征目前直接相加

   当前融合方式是：

   ```python
   visual_features + color_emb
   ```

   可以尝试更强的融合方式，例如 concat 后 linear、FiLM 调制、cross-attention、阶段 token 等。

6. 水平翻转只作用于图像

   训练增强中只翻转 `tensor_frames`，颜色特征不变。这通常合理，因为颜色与空间左右无关。

7. 阈值来自验证集

   checkpoint 保存的 `threshold` 来自验证集最佳 ACER。实际业务中可能需要按攻击通过率或真人误拒率重新选择阈值。

## 推荐排查流程

如果训练效果异常，建议按顺序检查：

1. 数据集中视频和 `.txt` 是否一一对应。
2. `.txt` 的帧号是否和视频帧号一致。
3. `skipped_corrupted_samples.txt` 是否记录大量坏视频。
4. 是否开启了人脸检测，以及检测结果是否稳定。
5. 视频帧数是否过长导致 padding 浪费或显存不足。
6. `metrics_history.csv` 中 val/test 指标是否同步提升。
7. `apcer` 是否过高，避免假人过多被判真人。

## 最小可用命令

查看帮助：

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python flash_liveness_project_v2.py --help
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python flash_liveness_project_v2.py train --help
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python flash_liveness_project_v2.py infer --help
```

训练：

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python flash_liveness_project_v2.py train \
  --data-root /path/to/dataset \
  --output-dir /path/to/output \
  --epochs 20 \
  --batch-size 2 \
  --device cuda:0
```

推理：

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python flash_liveness_project_v2.py infer \
  --checkpoint /path/to/best_flash_liveness_model.pth \
  --video-path /path/to/input.mp4 \
  --txt-path /path/to/input.txt \
  --device cuda:0
```
