# 炫彩活体数据下载与处理方案

更新时间: 2026-04-28

## 目标

构建一套尽量适合高级硅胶/高仿 3D 面具防御的 V3_1 数据管线：

- 真实炫彩采集数据保留逐帧颜色刺激 `.txt`，用于学习皮肤对颜色刺激的时序响应。
- 公开非炫彩 PAD/面具数据不伪造闪光时序，统一写入 `color=0` 的 neutral 协议。
- 硅胶、乳胶、纺织 3D、纸面 3D、真人佩戴纸面等 hard spoof 在训练集重复采样，避免被大量平面/屏幕攻击淹没。
- 训练/推理使用更保守的窗口融合，避免局部真实特征把伪造证据盖掉。

## 原始数据获取

高价值公开数据集大多不是开放直链，不能绕过授权下载：

- CASIA-SURF HiFiMask: 官方页面说明完整数据集已释放，但需要先签 license，再由数据方分享下载链接。适合高保真 mask 训练。
- CASIA-SURF SuHiFiMask: 官方页面说明需要签 license 后获取链接。适合更高级 mask 泛化。
- CASIA-SURF CeFA: 包含 2D/3D 攻击和 RGB/depth/IR，可做多模态预训练。
- WMCA: Zenodo 页面为 restricted files，包含 soft silicone flexible masks，是硅胶 mask 的关键补充。
- 3DMAD: Zenodo 页面为 restricted files，适合 3D mask 基线预训练。

已添加下载/授权记录脚本：

```bash
conda run -n anti-spoofing_scc_175 python scripts/download_flash_liveness_raw_data.py
```

输出：

```text
dataset/raw_downloads/
├── source_catalog.tsv
├── download_report.tsv
└── README.md
```

拿到授权后的直链，写一个 TSV：

```text
key	url	access	sha256	note
approved_hifimask	https://...	open_direct		approved institutional link
```

再下载：

```bash
conda run -n anti-spoofing_scc_175 python scripts/download_flash_liveness_raw_data.py \
  --source-list approved_sources.tsv
```

## 本地数据处理

当前项目已有本地原始 zip、已解压公开样本、20240320 炫彩归档和历史炫彩数据。先归档为按类型组织的资产库：

```bash
conda run -n anti-spoofing_scc_175 python scripts/organize_flash_liveness_assets_by_type.py
```

再构建 best protocol 数据集：

```bash
conda run -n anti-spoofing_scc_175 python scripts/prepare_flash_liveness_best_dataset.py --overwrite
```

输出：

```text
dataset/flash_liveness_best_protocol_v1/
├── train/live|spoof
├── val/live|spoof
├── test/live|spoof
├── manifest.tsv
├── summary.tsv
├── skipped.tsv
└── README.md
```

处理策略：

- 如果源视频旁边存在同名 `.txt`，认为是真实炫彩刺激标注，直接链接/复制。
- 如果没有 `.txt`，生成逐帧 `color=0`，表示原色视频/无额外闪光，而不是伪造 RGB 闪光。
- 只在 `train` split 对 hard spoof 类别重复采样，`val/test` 不重复，避免评估被复制样本污染。

## 推荐训练

```bash
bash scripts/run_flash_liveness_best_training.sh
```

关键参数：

```text
--missing-color-protocol neutral
--window-fusion quality_lower_trimmed_mean
--window-trim-ratio 0.2
--lambda-depth 0.05
--lambda-contrast 0.10
--lambda-fft 0.05
--aux-loss-max-ratio 0.35
--aux-loss-total-max-ratio 0.70
--yaw-augment-prob 0.35
--yaw-augment-max-ratio 0.10
```

`quality_lower_trimmed_mean` 会偏向保留低 live 概率窗口。对于高级硅胶面具，更应该让“可疑局部/可疑时间段”影响最终视频分数，而不是把高 live 局部窗口平均掉。

辅助 loss 会被动态限制：单个辅助项和辅助项总和都不能无限压过分类主损失，避免模型只关注 pseudo-depth、FFT 或某一类物理特征。训练增强使用整段一致的左右 yaw 透视，不做上下翻转，也不做简单左右镜像。

## 数据补采建议

现有公开数据仍不足以覆盖真正的“炫彩挑战 + 高级硅胶”。后续自采时建议：

- 每条视频保存同步的逐帧颜色 `.txt`，不要只保存最终视频。
- 颜色协议使用随机 RGB/CMY/白/暗、不同亮度和随机 hold/restoration 时间。
- 同一主体采集 live、硅胶、乳胶、树脂、纸面、屏幕 replay，保持同设备/同环境成组。
- 标注攻击材料、面具型号、是否露眼/露嘴、是否真人佩戴、环境光、设备型号。
- 采集 live 的 neutral 对照视频，避免模型把 neutral 协议本身学成 spoof。

## 官方来源

- CASIA-SURF HiFiMask: https://sites.google.com/view/face-anti-spoofing-challenge/dataset-download/casia-surf-hifimaskiccv2021
- CASIA-SURF SuHiFiMask: https://sites.google.com/view/face-anti-spoofing-challenge/dataset-download/casia-surf-suhifimaskcvpr2023
- CASIA-SURF CeFA: https://sites.google.com/view/face-anti-spoofing-challenge/dataset-download/casia-surf-cefacvpr2020
- WMCA: https://www.idiap.ch/en/scientific-research/data/wmca
- WMCA Zenodo: https://zenodo.org/records/4580313
- 3DMAD Zenodo: https://zenodo.org/records/4068477
