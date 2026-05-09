# raw/fake 旧版视频采集协议反推

数据来源：

`/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset/raw/fake`

## 全量读取结论

- 目录内共有 `8131` 个 `.txt`、`8157` 个 `.avi`。
- 去掉 `url.txt` 后，有效颜色 txt 为 `8130` 个。
- `8124` 个 txt 能找到同名 avi；有 `7` 个 txt 没有同名 avi，`33` 个 avi 没有同名 txt。
- txt 绝大多数为两列：`frame_idx,color_int`。
- 有 `8126 / 8130` 个有效 txt 是 5 段颜色协议。
- 5 段协议固定为：

```text
白屏 -> 彩色1 -> 彩色2 -> 彩色3 -> 黑屏
```

对应 packed RGB：

```text
16777215 -> c1 -> c2 -> c3 -> 0
```

视频容器头全量读取结果：

- `8101 / 8157` 个 avi 是 `MJPG` / `mjpeg`。
- 主体分辨率为 `1080x1920`。
- 主体帧率为 `30fps`。
- txt 行数和 avi 帧数基本一一对应：`8093 / 8124` 个可探测同名样本完全一致。
- 少量样本是损坏视频、缺失文件、提前中断或帧数不一致。

## 反推出的颜色池

中间 3 个彩色闪光来自 12 个高饱和 RGB 颜色：

| packed | RGB |
|---:|---|
| 16716820 | 255,20,20 |
| 16716928 | 255,20,128 |
| 16717055 | 255,20,255 |
| 16744468 | 255,128,20 |
| 16776980 | 255,255,20 |
| 8453908 | 128,255,20 |
| 1376020 | 20,255,20 |
| 1376128 | 20,255,128 |
| 1376255 | 20,255,255 |
| 1343743 | 20,128,255 |
| 1316095 | 20,20,255 |
| 8393983 | 128,20,255 |

这 12 色近似一个色相环，采集时从中取 3 个颜色作为一次炫光挑战。

## 反推出的时序

主流短协议大约是：

```text
18,18,18,18,16 frames @ 30fps
```

即总长约 `2.8s - 3.0s`。

另有一批长协议大约是：

```text
21,30,30,30,27 frames @ 30fps
```

即总长约 `4.3s - 4.6s`。

由于真实采集端应该是按系统时钟切换颜色、按摄像头回调写帧，实际每段帧数会有 1-2 帧波动。

## 反推采集脚本

已新增：

`scripts/legacy_facecollect_capture.py`

它复现旧版 FaceCollect 风格输出：

- 输出 `.avi`
- 编码默认 `MJPG`
- 同名 `.txt`
- txt 每行写 `frame_idx,color_int`
- 默认协议是 `白屏 + 3 个彩色闪光 + 黑屏`
- 支持从 `raw/fake` 历史 txt 中按经验分布抽取三色序列和段长

示例：

```bash
python scripts/legacy_facecollect_capture.py \
  --output-dir /tmp/legacy_facecollect \
  --device-name mate10 \
  --subject none \
  --attack-label 5 \
  --camera-id 0
```

只查看反推协议，不打开摄像头：

```bash
python scripts/legacy_facecollect_capture.py \
  --output-dir /tmp/legacy_facecollect \
  --dry-run
```

指定一组历史颜色：

```bash
python scripts/legacy_facecollect_capture.py \
  --output-dir /tmp/legacy_facecollect \
  --color-mode explicit \
  --colors 16717055,1376255,16716928 \
  --segment-frames 18,18,18,18,16
```

## 说明

这个脚本是根据 `raw/fake` 全量数据形态反推出来的可复现版本，不等价于找到了当年的原始采集端源码。当前仓库里能确认的是后处理链路会读取 `.avi + .txt`，而真正生成这些原始对的采集端源码没有在归档中完整出现。
