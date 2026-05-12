# Flash Collect Stimulus API Service

这个服务用于生成和采集 V3 `collect_flash` 风格的现场视频。

它做两件事：

1. 服务端根据训练协议生成浏览器闪光 timeline。
2. 浏览器/手机页面调用摄像头，全屏按 timeline 直接切色，同时录制现场画面并上传。
3. 服务端按录制视频的实际 `frame_count/fps` 重新生成同名 `recording_*.txt`。
4. 页面下载入口会把录制视频和对应 txt 打成一个 zip，避免只拿到视频而漏掉颜色标签。

默认颜色序号来自 `scripts/collect_flash_liveness_video.py`：

| 序号 | RGB | packed int |
| --- | --- | ---: |
| 1 | `(255, 20, 255)` | `16717055` |
| 2 | `(20, 255, 20)` | `1376020` |
| 3 | `(255, 20, 20)` | `16716820` |

默认时间线按当前 V3 best 训练集 `dataset/flash_liveness_asset_archive_fixed_collect_protocol` 生成：

```text
warmup 1.0s black
-> (255, 20, 255) hold 0.35s
-> (20, 255, 20) hold 0.35s
-> (255, 20, 20) hold 0.35s
-> repeat colors continuously until tail
-> tail 0.5s black
```

`restore_seconds` 默认是 `0.0`，对应训练集里的连续切色。`total_seconds` 是调用方指定的录制总时长，默认值 `3.0` 只是推荐值，不是协议限制；服务会按传入的视频总时长连续循环三色直到最后 `0.5s` 黑屏。

## V3 best 模型推荐采集参数

如果采集视频要送入当前 V3 best 模型：

```text
flash_liveness_runs/flash_liveness_v3_fixed_protocol_gpu3/best_flash_liveness_model.pth
```

为了最大限度贴近当前模型训练/测试协议，采集 session 必须贴近下面的数据集协议：

```text
dataset/flash_liveness_asset_archive_fixed_collect_protocol

warmup 1.0s black
-> (255, 20, 255) hold 0.35s
-> (20, 255, 20) hold 0.35s
-> (255, 20, 20) hold 0.35s
-> repeat selected colors continuously
-> tail 0.5s black
```

训练集中的同名 txt 是按原视频 `frame_count/fps` 生成逐帧颜色标签，所以协议没有固定 `cycles`，训练集视频时长也不是固定值。采集服务使用调用方传入的 `total_seconds` 控制录制总时长，并在 `warmup` 和 `tail` 之间持续循环三色。

对应 packed RGB：

```text
black: 0
purple: 16717055
green: 1376020
red: 16716820
```

推荐创建 session 的 API 请求。这里的 `total_seconds=3.0` 是默认推荐采集时长，可以按业务改成任意大于 `warmup_seconds + tail_seconds` 的时长。颜色顺序和时间参数被服务端固定校验为训练协议，不能改成其它颜色或 `restore_seconds=0.15`：

```bash
curl -X POST "http://127.0.0.1:18132/api/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "color_indices": [1, 2, 3],
    "total_seconds": 3.0,
    "warmup_seconds": 1.0,
    "hold_seconds": 0.35,
    "restore_seconds": 0.0,
    "tail_seconds": 0.5,
    "fps": 30,
    "width": 1080,
    "height": 1920
  }'
```

前端页面中协议参数应保持一致，`总时长秒` 可按实际采集需要调整：

```text
颜色序号: 1,2,3
总时长秒: 3.0  # 推荐值，可调整
warmup 秒: 1.0
hold 秒: 0.35
restore 秒: 0.0
tail 秒: 0.5
fps: 30
宽/高: 1080x1920
```

当 `total_seconds=3.0` 时，生成的刺激视频总时长为：

```text
3.0s = 1.0s black warmup + 1.5s continuous color loop + 0.5s black tail
```

### 送入 V3 推理服务

V3 推理服务路径：

```text
flash_liveness_v3_api_service/
```

推荐流程：

1. 用本服务按上面的 V3 推荐参数创建 session。
2. 浏览器/手机点击“开始录制”，录制现场摄像头视频。
3. 从 session 目录取 `recording_*.webm` 或 `recording_*.mp4`，以及服务端生成的同名 `recording_*.txt`。
4. 上传录制视频和同名 txt 到 V3 推理服务。

示例：

```bash
curl -X POST "http://127.0.0.1:18131/predict" \
  -F "file=@flash_collect_stimulus_api_service/outputs/<session_id>/recording_001.webm" \
  -F "txt_file=@flash_collect_stimulus_api_service/outputs/<session_id>/recording_001.txt"
```

如果不提供 `txt_file`，V3 推理服务会按 checkpoint 配置自动生成 `collect_flash` 颜色时间线。为了让自动补齐尽量正确，采集端必须使用上面的 V3 推荐协议，尤其是：

```text
restore_seconds = 0.0
total_seconds should match the recorded video duration
```

V3 推理服务自动补齐颜色标签时会读取上传视频的实际 `frame_count/fps`，因此视频可以是任意时长；关键是采集过程必须使用同一套 `fixed_collect_protocol` 闪光规则。

不要用其它刺激视频或外部颜色 txt 直接当作录制视频的 `txt_file` 上传。浏览器录制出来的摄像头视频可能因为 MediaRecorder、浏览器、手机设备而出现实际 fps 或帧数差异。

本服务现在会在 `/api/sessions/<session_id>/recording` 上传完成后，参考 `scripts/build_flash_liveness_fixed_protocol_dataset.py` 的处理方式，读取录制视频实际 `frame_count/fps` 并生成同名逐帧 txt：

```text
recording_001.webm
recording_001.txt
```

这个同名 txt 才是录制视频对应的颜色标签。

### 不推荐的调用方式

以下配置会偏离当前 V3 best 模型训练协议，可能降低准确率：

```text
restore_seconds = 0.15
使用固定 cycles 截断颜色序列，而不是按 total_seconds 连续循环到 tail
white warmup = 16777215
自定义三色，例如 8393983,1376255,1376020
```

这些配置更接近其它实验、ThunderGuard 采集方式或 V3.1 restore_original 数据集，不建议直接用于当前 V3 best 模型的线上准确率测试。

## 启动

```bash
CUDA_VISIBLE_DEVICES=1 /home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python \
  flash_collect_stimulus_api_service/app.py \
  --host 0.0.0.0 \
  --port 18132
```

手机访问时，摄像头权限通常要求 HTTPS 或 localhost。局域网手机访问 `http://server-ip:18132` 时，如果浏览器拒绝摄像头，需要用 HTTPS 反向代理或临时证书。

如果页面报错：

```text
TypeError: Cannot read properties of undefined (reading 'getUserMedia')
```

说明当前浏览器没有开放摄像头 API。最常见原因是手机用普通 HTTP 访问服务器 IP，例如：

```text
http://server-ip:18132/
```

现代 Chrome/Safari/Edge 只会在安全上下文开放摄像头：

- `https://...`
- `http://localhost`
- `http://127.0.0.1`

推荐访问方式：

1. 生产或多人测试：使用 HTTPS 域名或 HTTPS 反向代理访问采集服务。
2. Android USB 调试测试：在本机执行端口反向代理后，手机访问 `http://127.0.0.1:18132/`。

```bash
adb reverse tcp:18132 tcp:18132
```

3. 如果已有可信证书，也可以直接用采集服务 HTTPS 参数启动：

```bash
CUDA_VISIBLE_DEVICES=1 /home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python \
  flash_collect_stimulus_api_service/app.py \
  --host 0.0.0.0 \
  --port 18132 \
  --ssl-certfile /path/to/fullchain.pem \
  --ssl-keyfile /path/to/privkey.pem
```

仅生成 MP4、调用 `/api/sessions` 或上传文件不要求 HTTPS；只有浏览器/手机页面直接调用摄像头录制时才需要安全上下文。

## 页面

```text
http://127.0.0.1:18132/
```

页面流程：

1. 选择前置/后置或具体摄像头。
2. 输入录制总时长，颜色顺序固定为 `1,2,3`。
3. 点击 `开始录制`。
4. 浏览器请求摄像头，全屏按训练 timeline 切色，同时录制现场视频；录制过程中右下角会显示实时摄像头画面。
5. 录制结束后上传到服务端 session 目录。
6. 上传完成后，`下载录制视频+TXT` 下载本次录制视频和同名 `recording_*.txt`。

对应下载 API：

```text
/api/sessions/<session_id>/recordings/<recording_stem>.zip
```

## API

状态检查：

```bash
curl "http://127.0.0.1:18132/api/status"
```

创建采集 session：

```bash
curl -X POST "http://127.0.0.1:18132/api/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "color_indices": [1, 2, 3],
    "total_seconds": 3.0,
    "warmup_seconds": 1.0,
    "hold_seconds": 0.35,
    "restore_seconds": 0.0,
    "tail_seconds": 0.5,
    "fps": 30,
    "width": 1080,
    "height": 1920
  }'
```

返回字段：

- `recording_upload_url`: 浏览器录制视频上传入口
- `metadata_url`: session 元数据

上传录制视频：

```bash
curl -X POST "http://127.0.0.1:18132/api/sessions/<session_id>/recording" \
  -F "file=@recording.webm" \
  -F 'client_metadata={"source":"manual"}'
```

上传成功后返回 `recording_path`、`recording_txt_path` 和 `recording_protocol`。其中 `recording_protocol.frame_count/fps` 来自服务端读取录制视频得到的实际 metadata。

## 输出

默认输出目录：

```text
flash_collect_stimulus_api_service/outputs/
```

每个 session 会保存：

```text
metadata.json
recording_001.webm
recording_001.txt
```

其中 `recording_*.webm/mp4` 是浏览器/手机摄像头录到的现场画面，`recording_*.txt` 是按录制视频实际帧数/FPS 生成的训练协议颜色标签，可以继续送入 V3 推理服务。
