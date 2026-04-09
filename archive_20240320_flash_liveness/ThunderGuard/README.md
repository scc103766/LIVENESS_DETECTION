# 闪光活体归档复现说明

本文档针对目录 `/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档` 的实际内容整理，目标是让归档可以被重新理解、复现和复查。

## 1. 归档内容总览

归档根目录下的有效内容可以分成 5 类：

```text
20240320闪光活体归档/
├── 读我.txt                         # 入口说明，指向 ThunderGuard/README.md
├── ThunderGuard/                    # 闪光活体主工程：预处理、训练、导出、推理
├── FaceAlign/                       # 人脸检测、对齐、深度图/法线图相关工具
├── dataset/                         # 已整理的数据与导出样本
├── environment/                     # 归档环境（conda 环境包与解压后的环境）
└── resources_smoke/                 # 一份额外模型资源
```

其中最关键的目录如下：

### `ThunderGuard/`

```text
ThunderGuard/
├── libtg/                           # C++ 图像预处理库，可编译 Linux/Android
├── pytg/                            # 训练、测试、导出 ONNX
├── tg_process/                      # 从视频取样、生成法线图
├── tg_infer/                        # ONNX / PyTorch 推理脚本
├── resources/                       # 现成权重与 ONNX
├── data/                            # 运行时数据目录（复现时通常需要自己准备）
└── nxreadme训练模型步骤.txt          # 较简略的旧训练笔记
```

### `FaceAlign/`

```text
FaceAlign/
├── face_detect/                     # 人脸和眼睛关键点检测
├── pyfa/                            # 对齐脚本
├── falib/                           # Android/Linux 对齐库
├── falib_demo/                      # demo 与旧 build 产物
└── data/                            # 示例数据
```

### `dataset/`

```text
dataset/
├── dataset/                         # 原始整理数据
├── tg_export/                       # 已导出的训练/测试样本，可直接用于复现训练
├── tg_export_smoke/                 # 烟雾/备用导出数据
└── 根据训练样本找原始视频.txt
```

### `environment/`

```text
environment/
├── exp_yyag_1.tgz
├── exp_yanyu_cv.tgz
└── exp_yanyu_cv/                    # 已解压环境，含 python3.7 / pytorch1.10 / cuda10.2
```

## 2. 建议的复现路径

这份归档里同时包含“完整数据处理链路”和“已导出的可直接训练样本”。  
如果你的目标是尽快复现结果，建议优先走下面这条最短路径：

1. 使用 `environment/exp_yanyu_cv` 作为参考环境。
2. 将 `dataset/tg_export/train` 和 `dataset/tg_export/test` 放到 `ThunderGuard/data/sample/` 下。
3. 在 `ThunderGuard/pytg` 中训练或直接测试已有模型。
4. 如需部署，使用 `convert_2_onnx.py` 导出 ONNX。

原因：

- `dataset/tg_export` 已经是训练脚本可直接消费的格式。
- `resources/` 中已有 `model_best.pth.tar` 和 `MoEA_score.onnx`，可以先验证推理链路。
- 原始 README 中提到的 `data_adapt/` 等目录在当前归档里并不存在，不适合作为主复现路径。

## 3. 环境说明

根据归档中的已解压环境 `environment/exp_yanyu_cv/conda-meta`，可以确认一套接近可运行的环境大致为：

- Linux
- Python 3.7
- PyTorch 1.10.1
- torchvision 0.11.2
- cudatoolkit 10.2
- numpy 1.21.5
- onnx / onnxruntime
- opencv-python
- tqdm

`FaceAlign` 里的旧说明仍提到 `tensorflow<1.9`，这主要对应其早期对齐链路；如果只复现 ThunderGuard 的训练/推理，不一定需要完整恢复这部分依赖。

## 4. 直接复现训练

### 4.1 准备数据目录

在 `ThunderGuard/` 下建立训练脚本默认使用的数据目录：

```shell
cd /supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard
mkdir -p data/sample
```

将现成导出样本复制为：

```text
ThunderGuard/data/sample/train
ThunderGuard/data/sample/test
```

来源目录为：

```text
../dataset/tg_export/train
../dataset/tg_export/test
```

训练脚本默认读取 `../data/sample`，这一点来自 `pytg/config.py`。

### 4.2 单卡训练

```shell
cd pytg
python train.py --network MoEA
```

说明：

- 原旧文档写成了 `python train --network MoEA`，实际文件名是 `train.py`。
- 默认模型输出目录是 `../resources/MoEA/`。

### 4.3 多卡训练

```shell
cd pytg
python -m torch.distributed.launch --nproc_per_node <卡数> train.py --network MoEA
```

## 5. 使用现成模型做推理

归档中已经有：

- `ThunderGuard/resources/MoEA/model_best.pth.tar`
- `ThunderGuard/resources/MoEA/checkpoint.pth.tar`
- `ThunderGuard/resources/MoEA_score.onnx`

### 5.1 测试导出的 ONNX

```shell
cd tg_infer
python infer_onnx.py --infer_type score --normal_cues_path ../data/sample/test --threshold 0.5
```

注意：

- `infer_onnx.py` 默认输入目录是 `../data/sample/test_depth`，如果你按本 README 准备的是 `test`，需要显式传 `--normal_cues_path`。
- 输入数据需要是训练脚本使用的那种样本格式，即每个样本包含 `.jpg`、`.txt`、`_d.jpg` 三类文件。

### 5.2 对当前目录格式进行推理

如果你手里是“每个样本 3 张 PNG”的目录结构，可以使用：

```shell
cd tg_infer
python infer_current_dir_onnx.py \
  --input-root <你的目录> \
  --onnx-path ../resources/MoEA_score.onnx \
  --threshold 0.5
```

该脚本会输出 `current_dir_infer_report.json`。

## 6. 导出 ONNX

```shell
cd /supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg
python convert_2_onnx.py --network MoEA --infer_type score
```

如果需要进一步简化 ONNX：

```shell
cd ../resources
python -m onnxsim TGAN_score.onnx TGModel.onnx --input-shape 6,3,256,256
```

注意：

- 这条命令保留自原始流程，但实际输入输出文件名是否一致，需要以 `convert_2_onnx.py` 的导出结果为准再核对一次。
- 当前归档中可直接看到的是 `MoEA_score.onnx`，不是 `TGAN_score.onnx`。

## 7. 从原始数据重新走完整流程

如果你不是只做训练复现，而是想从视频开始全流程重跑，可以按下面的链路：

1. `tg_process/tg_process.py --opt choose`
2. `FaceAlign/face_detect/face_detect_with_landmark.py`
3. `FaceAlign/pyfa/face_align.py`
4. `tg_process/tg_process.py --opt normal`
5. `pytg/gen_dataset.py`
6. `pytg/train.py`

### 7.1 从视频抽取取样帧

```shell
cd ../tg_process
python tg_process.py --opt choose --from_path <原始视频目录> --to_path <取样输出目录>
```

这里要求输入目录中的视频和同名 `.txt` 满足旧命名规则；`tg_process.py` 会校验文件名。

### 7.2 人脸检测与对齐

```shell
cd ../../FaceAlign/face_detect
python face_detect_with_landmark.py --data_path <取样输出目录>

cd ../pyfa
python face_align.py \
  --align_size 480 \
  --face_size 260 \
  --from_path <取样输出目录> \
  --to_path <对齐输出目录> \
  --normalize_percent -1.0
```

### 7.3 生成法线图

```shell
cd ../../ThunderGuard/tg_process
python tg_process.py --opt normal --from_path <对齐输出目录> --to_path <法线图输出目录>
```

### 7.4 切分训练/测试集

```shell
cd ../pytg
python gen_dataset.py --from_path <法线图目录上级> --to_path ../data/sample
```

`gen_dataset.py` 会读取：

- 子目录中的样本文件
- 可选的 `sample_meta.txt` 作为采样权重

并生成：

- `../data/sample/train`
- `../data/sample/test`

## 8. Android 相关

### 8.1 编译人脸对齐库

```shell
cd /supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/FaceAlign/falib
./build_android.sh
# 或
./build_android_64.sh
```

### 8.2 编译图像预处理库

```shell
cd /supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/libtg
./build_android.sh
# 或
./build_android_64.sh
```

脚本里依赖本机 NDK 路径，运行前需要先检查并修改脚本中的配置。

## 9. 当前归档里需要注意的过时点

下面这些内容在原始说明里存在，但和当前归档不完全一致：

- 原文中的 `Python: python3.1~3.6` 已明显过时，归档环境实际是 Python 3.7。
- 原文中的 `data_adapt/` 目录在当前归档中不存在，相关步骤不能直接照搬。
- 原文中的 `python train --network MoEA` 应为 `python train.py --network MoEA`。
- 原文中的 ONNX 文件名和当前目录中的实际文件名不完全一致。
- 原文把“数据处理全流程”和“现成样本直接复现”混在一起，不利于复现，本说明已拆开。

## 10. 推荐的最小验证顺序

如果你只是想快速确认归档能跑通，建议按下面顺序：

1. 先确认 `dataset/tg_export/train`、`dataset/tg_export/test` 已放到 `ThunderGuard/data/sample/`。
2. 运行 `python train.py --network MoEA`，确认训练能读到数据。
3. 直接用 `resources/MoEA_score.onnx` 运行一次 `infer_current_dir_onnx.py` 或 `infer_onnx.py`。
4. 最后再考虑是否需要从原始视频重跑 `tg_process + FaceAlign` 全链路。

这样最省时间，也最符合当前归档里的实际文件组织。
