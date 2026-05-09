# Flash Liveness V3 API 调用手册

默认服务地址：

```text
http://127.0.0.1:18131
```

## 健康检查

```bash
curl "http://127.0.0.1:18131/health"
```

正常返回会包含：

```json
{
  "status": "ok",
  "service": "flash_liveness_v3_api_service",
  "version": "v3_fixed_protocol",
  "threshold": 0.9375
}
```

## 视频上传

```bash
curl -X POST "http://127.0.0.1:18131/predict" \
  -F "file=@/path/to/input.mp4"
```

没有同名 txt 时，服务会按 V3 固定 collect_flash 协议自动补齐颜色标签。

## 视频 + txt 上传

```bash
curl -X POST "http://127.0.0.1:18131/predict" \
  -F "file=@/path/to/input.mp4" \
  -F "txt_file=@/path/to/input.txt"
```

txt 格式：

```text
frame_index,color_int
```

例如：

```text
0,0
31,16717055
41,1376020
52,16716820
```

## zip 上传

zip 内建议包含同名视频和 txt：

```text
sample.mp4
sample.txt
```

调用：

```bash
curl -X POST "http://127.0.0.1:18131/predict" \
  -F "file=@/path/to/sample.zip"
```

如果 zip 内没有 txt，服务同样会使用 `collect_flash` 协议补齐。

## 本机路径调用

后端服务和视频文件在同一台机器时，可以调用：

```bash
curl -X POST "http://127.0.0.1:18131/predict_path" \
  -H "Content-Type: application/json" \
  -d '{"video_path":"/path/to/input.mp4","txt_path":"/path/to/input.txt"}'
```

`txt_path` 可省略。

## 返回示例

```json
{
  "request_id": "a12f3c4d5e6f",
  "result": "live",
  "prediction_id": 1,
  "probability_live": 0.9812,
  "threshold": 0.9375,
  "input_type": "video",
  "txt_source": "provided",
  "num_frames": 128,
  "num_frames_used": 128,
  "num_windows": 1,
  "checkpoint_path": ".../best_flash_liveness_model.pth",
  "metadata_json": ".../outputs/a12f3c4d5e6f/metadata.json",
  "api_version": "v3_fixed_protocol"
}
```

调用方通常只需要关注：

- `result`
- `probability_live`
- `threshold`
- `txt_source`
- `num_frames`
- `num_windows`
