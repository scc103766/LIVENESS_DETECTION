# h5_raw/vreal 旧版 H5 采集代码反推

数据目录：

`/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset/h5_raw/vreal`

## 目录结构结论

`vreal` 是一个扁平目录，只有同名 `.mp4 + .txt`：

- `.mp4`: `625`
- `.txt`: `625`
- 未发现缺失配对样本。

文件名形态：

```text
1652405201303_h5_1_1_none_1.mp4
1652405201303_h5_1_1_none_1.txt
```

可反推出字段含义：

```text
timestamp_ms_h5_1_1_none_1
```

- `timestamp_ms`: 采集时毫秒时间戳。
- `h5`: 采集端类型，说明来自 H5/Web 或 H5+Native/WebView 采集链路。
- 最后一位 `1`: 真人标签，和 `vreal` 目录一致。

采集时间范围约为：

```text
2022-05-13 09:26:41 到 2022-05-18 16:58:40
```

## txt 格式

全部 `625` 个 txt 都是 5 行、每行 2 列：

```text
0,16777215
1,16717055
2,1343743
3,1376255
4,0
```

这不是逐帧颜色日志，而是 5 个采集阶段/关键帧的颜色表：

```text
0: 白屏
1: 彩色1
2: 彩色2
3: 彩色3
4: 黑屏
```

固定结构：

```text
16777215 -> c1 -> c2 -> c3 -> 0
```

## 视频格式

全量 `625` 个 mp4 的容器头结论：

- 编码：`h264 / avc1`
- 像素格式：`yuv420p`
- 分辨率：`640x480`
- rotate tag：`270`
- 时长中位数：约 `4.0s`
- 帧数中位数：约 `102`
- fps 中位数：约 `25.2fps`
- 码率中位数：约 `5.2Mbps`

这类 MP4 更像移动端系统编码器或 Android WebView/Native Bridge 输出，而不是普通桌面浏览器固定输出。现代浏览器的 `MediaRecorder` 不一定能强制生成 H.264 MP4，可能只能生成 WebM。

## 颜色池

中间 3 个彩色闪光仍来自旧版 12 色池：

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

`vreal` 中三色组合非常分散：625 个样本里有 352 种 5 色序列，说明采集端很可能每次从 12 色池中随机抽取 3 个彩色阶段。

## 和 ThunderGuard 后处理的关系

`ThunderGuard/tg_process/tg_process.py` 中有专门兼容 5 行 txt 的逻辑：

```python
def fill_frame_line_list(frame_line_list, target_len):
    if len(frame_line_list) == 5:
        ...
```

这说明旧流程允许原始 txt 只有 5 行。后处理会把 5 个阶段扩展到完整视频帧数，再进入 `choose` 阶段挑 5 张代表帧。

因此 `h5_raw/vreal` 的采集端不需要逐帧写 txt，它只需要写：

```text
0,白屏颜色
1,第一种彩色
2,第二种彩色
3,第三种彩色
4,黑屏颜色
```

## 反推采集流程

最接近原始数据的流程应为：

1. H5 页面或 H5+Native/WebView 打开前置摄像头。
2. 请求约 `640x480` 视频流。
3. 开始录制 H.264 MP4。
4. 页面全屏依次显示 5 个颜色阶段：
   - 白屏
   - 随机彩色 1
   - 随机彩色 2
   - 随机彩色 3
   - 黑屏
5. 每个阶段约 `0.8s`，总时长约 `4s`。
6. 停止录制。
7. 保存同名 `.mp4` 和 5 行 `.txt`。

## 已反推实现

新增目录：

`legacy_h5_vreal_capture/`

包含：

- `index.html`: H5 采集页面，负责摄像头、闪光协议、视频录制、txt 生成。
- `server.py`: 本地上传/保存服务。
- `README.md`: 运行说明。

运行：

```bash
cd /supercloud/llm-code/scc/scc/Liveness_Detection/legacy_h5_vreal_capture
python server.py
```

打开：

```text
http://<server-ip>:18082/
```

`server.py` 默认绑定 `0.0.0.0:18082`，因此同一可达网络里的手机或其他设备可以直接访问当前服务器页面；采集时使用访问设备自己的摄像头，生成的 `.mp4/.txt` 会通过 `/upload` 回传并保存到当前服务器：

```text
legacy_h5_vreal_capture/uploads/<sample_stem>/
```

如果是手机浏览器，通常需要 HTTPS 才允许访问摄像头。临时自签名证书启动方式：

```bash
cd /supercloud/llm-code/scc/scc/Liveness_Detection/legacy_h5_vreal_capture
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout certs/h5_vreal.key \
  -out certs/h5_vreal.crt \
  -days 7 \
  -subj "/CN=<server-ip>"

python server.py \
  --host 0.0.0.0 \
  --port 18443 \
  --cert-file certs/h5_vreal.crt \
  --key-file certs/h5_vreal.key
```

然后访问：

```text
https://<server-ip>:18443/
```

注意：这是按 `h5_raw/vreal` 内容和结构反推的可复现版本，不等于找到了当年的原始 H5 源码。若要完全贴近历史 MP4，需要在支持 H.264 MP4 的 Android H5/WebView 或 Native Bridge 环境运行。
