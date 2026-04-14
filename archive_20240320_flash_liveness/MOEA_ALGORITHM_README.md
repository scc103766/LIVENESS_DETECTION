# ThunderGuard 中 MoEA 算法详解

本文档针对归档目录中 ThunderGuard 实际使用的 `MoEA` 模型进行单独解读，目标是回答四个问题：

1. `MoEA` 的算法原理是什么
2. 归档项目里是如何实现它的
3. 它相比普通单分支模型的优势是什么
4. 它更适合什么样的业务场景

对应代码主入口在：

- [model.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/networks/MoEA/model.py)
- [loss.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/networks/MoEA/loss.py)
- [train.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/train.py)
- [tg_dataset.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/dataset/tg_dataset.py)
- [convert_2_onnx.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/convert_2_onnx.py)

## 1. 一句话理解 MoEA

MoEA 可以理解成：

```text
带可训练门控的多专家闪光活体模型
```

它不是只用一个网络统一处理所有攻击类型，而是显式维护了多个“专家”分支，再由一个 `gating net` 去决定当前样本更应该相信哪个专家。

在这个项目里，这些专家主要对应三类攻击形态：

- 印刷品攻击 `print`
- 屏幕/电子屏攻击 `screen`
- 3D 头模攻击 `3d model`

因此，MoEA 的核心思想不是“所有攻击都长得差不多”，而是：

```text
不同攻击介质的闪光响应规律不同，
应该让不同专家分别建模，再通过门控融合。
```

## 2. MoEA 的核心原理

从算法角度看，MoEA 是把下面几件事组合在一起：

1. 多专家 `Mixture of Experts`
2. 可训练门控网络 `Gating Net`
3. 多帧注意力融合
4. 伪深度回归
5. 活体二分类
6. 闪光颜色一致性学习

也就是说，它不是一个单一损失、单一输出的网络，而是一个多任务、多分支联合优化的闪光活体框架。

### 2.1 多专家机制

MoEA 在 [model.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/networks/MoEA/model.py) 中定义了三个专家分支：

```python
self.u_net_print = ResUNet()
self.u_net_screen = ResUNet()
self.u_net_3d = ResUNet()
```

这三个专家不是做最终分类，而是都去预测“深度相关表示”。

直观理解：

- `print expert` 负责更擅长识别纸张/平面介质
- `screen expert` 负责更擅长识别屏幕翻拍
- `3d expert` 负责更擅长识别头模类攻击

然后模型不是硬切换使用某一个专家，而是用软门控权重把三者融合。

### 2.2 Gating Net

门控网络定义在：

```python
self.gate = nn.Sequential(
    DownConvNormAct(3, 32),
    DownConvNormAct(32, 64),
    DownConvNormAct(64, 64, kernel_size=7),
    AdaptiveAvgPool2d((1, 1)),
    Reshape(64),
    nn.Linear(64, 32),
    nn.Linear(32, 3),
)
```

输入是单张 normal cue 图像，输出是 3 维 gate logits，对应三类专家。

前向中：

```python
gate = self.gate(x)
gate_soft_max = torch.softmax(gate, dim=1)
```

这表示模型会为每个样本自动学出：

```text
[w_print, w_screen, w_3d]
```

并满足：

```text
w_print + w_screen + w_3d = 1
```

后续三个专家的输出会按这个权重进行加权融合，而不是简单平均。

### 2.3 多帧注意力融合

MoEA 处理的输入不是单帧图，而是 `6` 张 normal cue 图构成的序列。

训练数据在 [tg_dataset.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/dataset/tg_dataset.py) 中会把每个样本整理成：

```text
6 张 RGB normal cue + 6 张对应深度图
```

视觉输入最终是：

```text
[6, 3, 256, 256]
```

模型里另外有一条注意力分支：

```python
self.attention_stem = nn.Sequential(...)
self.easy_u_net_attention = EasyResUNet()
```

它先预测每一帧的注意力深度图，然后在 `infer_depth_map_attention(...)` 中得到跨 6 帧的注意力权重：

```python
attention_map = torch.softmax(torch.reshape(attention_map, [-1, self.frame_num, 64, 64]), dim=1)
```

随后在 `infer_depth_map(...)` 里，用这个注意力把 6 帧深度图融合成单帧深度图：

```python
single_depth_map = torch.sum(
    torch.reshape(depth_map, [-1, self.frame_num, 64, 64]) * depth_map_attention, dim=1, keepdim=True)
```

这一步的意义是：

```text
不是默认 6 帧等权，而是让模型自己学习哪些帧更值得信任。
```

### 2.4 伪深度建模

每个专家分支 `ResUNet` 都输出一个类似“像素级深度概率”的结果，随后通过 softmax 和深度系数表，恢复为连续深度图：

```python
depth_soft_max = self.pixel_wise_softmax(depth_x)
depth_map = torch.unsqueeze(torch.sum(self.depth_map_cof * depth_soft_max, dim=-1), dim=1)
```

这里的 `depth_map_cof` 是：

```text
0 ~ 1 的 256 个离散深度等级
```

也就是说，MoEA 不是直接回归一个实数深度，而是：

- 先预测每个像素属于哪一个深度桶
- 再把它变成连续深度图

这类做法通常比直接回归更稳定。

### 2.5 活体二分类

融合出的单帧深度图再进入：

```python
self.f_net = nn.Sequential(...)
```

得到 2 类 logits：

```python
single_p = torch.reshape(self.f_net(single_depth_map), [-1, 2])
single_pred = torch.softmax(single_p, dim=1)[:, 1]
```

这里：

- `single_p` 是真假二分类 logits
- `single_pred` 是样本为真人的概率分数

在部署导出 `infer_type=score` 时，返回的第二个输出就是这个 `single_pred`，也就是最终活体分数。

### 2.6 闪光颜色一致性学习

MoEA 不只做真假分类，还额外学习闪光颜色顺序关系：

```python
self.r_net = nn.Sequential(...)
```

输出 3 个颜色通道值后，再构造成颜色比较结果：

```python
sc = torch.reshape(self.r_net(x), [-1, 3])
sc_p = torch.cat((sc[:, 0:1] - sc[:, 1:2], sc[:, 1:2] - sc[:, 2:3], sc[:, 2:3] - sc[:, 0:1]), 1)
```

这一步不是在做“图像分类”，而是在让模型学习：

```text
当前 normal cue 中，RGB 通道之间的相对大小关系是否符合对应闪光颜色。
```

它相当于一个颜色物理一致性约束。

## 3. 项目里 MoEA 是怎么实现的

从工程实现角度，归档中的 MoEA 主要有 4 个部分。

### 3.1 输入数据格式

训练数据由 [tg_dataset.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/dataset/tg_dataset.py) 读取。

每个样本至少包含：

- `xxx.jpg`：6 张 normal cue 竖向拼接的大图
- `xxx_d.jpg`：3 张深度图竖向拼接
- `xxx.txt`：3 个 packed RGB 颜色值

数据集逻辑会：

1. 从 `.txt` 中读出 3 个颜色值
2. 构造 `sc_flag`
3. 从 `.jpg` 中切出 6 张 normal cue
4. 从 `_d.jpg` 中切出对应深度图
5. 把 6 组 `(RGB + depth)` 重新拼成网络训练输入

这一点非常关键：

```text
MoEA 训练用的不是原始视频，
而是 ThunderGuard 预处理后的 normal cue 样本。
```

### 3.2 模型主干

主干在 [model.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/networks/MoEA/model.py) 中可以概括成：

```text
Input normal cue
  -> gate net
  -> attention branch
  -> shared stem + position embedding
  -> 3 experts (print/screen/3d)
  -> depth fusion
  -> depth attention fusion across frames
  -> classification head
  -> color regression head
```

其中 `PositionEmbedding` 也被显式加入到了视觉主干：

```python
x = self.head_stem(x) + self.pos_embedding
```

归档里作者自己留下的简述 [readme.txt](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/networks/MoEA/readme.txt) 也验证了这条思路：

```text
特征加入PositionEmbedding
使用MoE结构
用可训练的Gating Net融合Expert
最后分类
```

### 3.3 损失函数

损失定义在 [loss.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/networks/MoEA/loss.py)。

它不是单一 loss，而是 4 类损失的组合：

- `color_loss`
  颜色关系学习损失，仅对正样本回归
- `score_loss`
  真人/攻击二分类损失
- `map_loss`
  深度图相关损失
- `gate_loss`
  门控专家选择损失，仅在特定攻击训练分支中启用

总损失是：

```python
loss = loss_0 + loss_1 * 0.16 + loss_2 * 6 + gate_loss
```

这说明项目里的 MoEA 本质上是一个多任务联合优化模型，而不是只盯着最终 score。

### 3.4 训练方式

训练入口在 [train.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/train.py)。

训练时并不是把所有攻击混成一个 DataLoader，而是拆成三套训练流：

- 真人 + 印刷品
- 真人 + 屏幕攻击
- 真人 + 3D 攻击

代码中对应：

```python
train_print_data_loader
train_screen_data_loader
train_model_data_loader
```

每一步训练会同时从这三路数据取 batch，分别计算 loss，再把它们合并。

这和 MoEA 的三专家结构是一一对应的：

```text
训练数据按攻击类型拆分
门控和专家也按攻击类型建模
```

这是这个实现比较有特色的一点。

## 4. 部署态是怎么落地的

部署导出在 [convert_2_onnx.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/convert_2_onnx.py) 中进行。

当使用：

```text
--network MoEA --infer_type score
```

导出的 ONNX 输入输出是：

- 输入：`input`
- 输出：`color`, `score`

其中：

- `color` 是颜色一致性相关输出
- `score` 是活体分数

部署时通常只会重点使用 `score`，但 `color` 还能配合规则做颜色验证。

这也是为什么归档中可直接看到：

- [MoEA_score.onnx](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/resources/MoEA_score.onnx)

## 5. MoEA 的优势

### 5.1 比单分支模型更适合异质攻击

纸张、屏幕、头模在闪光活体里不是同一种攻击分布。

如果只用一个统一分支，模型需要自己在一套参数里同时兼顾所有攻击类型，容易互相干扰。  
MoEA 用三专家拆开建模，可以让不同攻击介质的表示更专业化。

### 5.2 兼顾物理信息和判别信息

MoEA 不只输出真假结果，还同时建模：

- 伪深度
- 颜色关系
- 多帧融合权重
- 专家门控权重

这使它比纯黑盒二分类更容易吸收“闪光活体”这种任务里的物理先验。

### 5.3 多帧注意力融合更适合闪光场景

闪光采集中，不同帧的重要性通常不一样：

- 有些帧正处于颜色切换
- 有些帧人脸反射更清晰
- 有些帧信息质量更高

MoEA 用 attention 去融合多帧深度图，比简单平均更灵活。

### 5.4 部署形态清晰

训练态虽然复杂，但部署态已经被压缩成：

```text
6 x 3 x 256 x 256 输入
-> color + score 输出
```

这使它在工程上仍然可用，不需要把训练时所有中间监督都保留下来。

## 6. MoEA 的局限

### 6.1 强依赖 ThunderGuard 预处理链路

MoEA 不吃原始摄像头视频，而是吃 ThunderGuard 处理后的 normal cue。

这意味着：

- 你必须先有正确的采样、对齐、法线/深度生成流程
- 预处理错了，模型输入分布就会偏

### 6.2 对采集协议依赖较强

它假设输入满足固定闪光流程和颜色关系。  
如果真实业务里的打光顺序、屏幕亮度、相机响应和训练数据差异太大，性能会明显波动。

### 6.3 结构复杂，训练成本高

相比普通二分类网络，MoEA 同时维护：

- 3 个专家
- 1 个 gate
- 1 个注意力分支
- 颜色头
- 深度头

训练复杂度、调参复杂度和排障成本都更高。

## 7. MoEA 更适合什么场景

MoEA 最适合下面这类任务：

### 7.1 有明确闪光协议的主动活体检测

例如：

- 手机或平板屏幕按固定颜色顺序闪烁
- 摄像头同步采集面部响应
- 后端利用颜色一致性和时序深度模式判断真假

这种场景下，MoEA 可以充分利用“采集协议”本身，而不仅仅看脸。

### 7.2 攻击类型多样，且差异明显

如果业务里既有：

- 打印攻击
- 屏幕翻拍
- 头模攻击

而且三者分布差别很大，那么 MoEA 这种多专家结构通常比单分支更合适。

### 7.3 可以接受复杂预处理链路的场景

MoEA 更适合：

- 追求效果
- 可控制采集方式
- 可以维护配套预处理

而不太适合：

- 任意来源图片直接推理
- 非标准采集环境
- 极简部署链路

## 8. 适用性总结

如果用一句话总结 MoEA 在这个项目中的定位：

```text
MoEA 是一个面向主动闪光活体场景、结合多专家门控、多帧注意力和颜色一致性学习的专用模型。
```

它的强项不是“通用图片活体”，而是：

```text
在固定闪光协议下，对多种攻击介质做更细粒度的差异化建模。
```

## 9. 阅读建议

如果你准备继续深入代码，建议按下面顺序阅读：

1. [tg_dataset.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/dataset/tg_dataset.py)
   先理解训练样本长什么样
2. [model.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/networks/MoEA/model.py)
   再理解 gate、attention、experts、分类头和颜色头
3. [loss.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/networks/MoEA/loss.py)
   最后看多任务损失如何绑定到训练

如果你的目标是部署推理，再看：

4. [convert_2_onnx.py](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/convert_2_onnx.py)
5. [MoEA_score.onnx](/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/resources/MoEA_score.onnx)
