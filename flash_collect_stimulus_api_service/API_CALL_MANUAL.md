# Flash Collect Stimulus API 调用手册

本文档说明当前闪光视频采集方案中，后端如何通过 API 创建采集任务、接收前端/手机录制的视频、生成同名颜色标签 txt，并将结果继续送入 V3 活体检测服务。

## 1. 服务信息

默认服务地址：

```text
https://192.168.17.175:18132
```

当前服务使用 HTTPS 启动。服务器本机调试也可以访问：

```text
https://127.0.0.1:18132
```

如果使用的是自签名证书，`curl` 示例需要加 `-k`；给外部人员长期使用时建议换成可信 CA 证书或 HTTPS 反向代理域名。

当前服务目录：

```text
/supercloud/llm-code/scc/scc/Liveness_Detection/flash_collect_stimulus_api_service
```

默认输出目录：

```text
/supercloud/llm-code/scc/scc/Liveness_Detection/flash_collect_stimulus_api_service/outputs
```

启动命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python \
  /supercloud/llm-code/scc/scc/Liveness_Detection/flash_collect_stimulus_api_service/app.py \
  --host 0.0.0.0 \
  --port 18132 \
  --ssl-certfile /path/to/fullchain.pem \
  --ssl-keyfile /path/to/privkey.pem
```

## 2. 采集流程

后端调用采集 API 的完整流程：

1. 后端调用 `/api/sessions` 创建一次采集 session。
2. 后端把 `session_id`、`recording_upload_url`、`stimulus.timeline` 返回给前端或手机端。
3. 前端/手机端按 `stimulus.timeline` 全屏闪光，同时调用摄像头录制现场视频。
4. 前端/手机端把录制视频上传到 `/api/sessions/{session_id}/recording`。
5. 采集服务读取上传视频的实际 `frame_count/fps`，生成同名 `recording_*.txt`。
6. 后端调用 `/api/sessions/{session_id}/metadata` 查询视频和 txt 路径。
7. 后端可下载视频+txt zip，或直接把视频和 txt 上传到 V3 推理服务。

注意：后端 API 负责创建采集任务和保存结果，实际摄像头采集必须发生在浏览器/手机端。服务端无法通过 API 直接打开用户手机摄像头。

## 3. 固定闪光协议

当前服务固定使用 V3 `fixed_collect_protocol` 协议：

```text
warmup_seconds = 1.0
hold_seconds = 0.35
restore_seconds = 0.0
tail_seconds = 0.5
color_indices = [1, 2, 3]
```

颜色顺序：

| 序号 | RGB | packed int |
| ---: | --- | ---: |
| 1 | `[255, 20, 255]` | `16717055` |
| 2 | `[20, 255, 20]` | `1376020` |
| 3 | `[255, 20, 20]` | `16716820` |

默认推荐录制总时长：

```text
total_seconds = 4.0
```

`total_seconds` 可以按业务调整，但必须大于：

```text
warmup_seconds + tail_seconds
```

服务会在 `warmup` 和 `tail` 之间持续循环三色，而不是按固定 cycles 截断。

## 4. 健康检查

### 请求

```bash
curl -k "https://127.0.0.1:18132/health"
```

也可以调用：

```bash
curl -k "https://127.0.0.1:18132/api/status"
```

### 返回示例

```json
{
  "status": "ok",
  "service": "flash_collect_stimulus_api_service",
  "output_dir": "/supercloud/llm-code/scc/scc/Liveness_Detection/flash_collect_stimulus_api_service/outputs",
  "recommended_total_seconds": 4.0,
  "duration_policy": "caller_defined_total_seconds",
  "flash_protocol": {
    "name": "fixed_collect_protocol",
    "warmup_seconds": 1.0,
    "hold_seconds": 0.35,
    "restore_seconds": 0.0,
    "tail_seconds": 0.5,
    "color_order_rgb": [[255, 20, 255], [20, 255, 20], [255, 20, 20]],
    "color_order_packed": [16717055, 1376020, 16716820]
  },
  "default_palette": {
    "1": [255, 20, 255],
    "2": [20, 255, 20],
    "3": [255, 20, 20]
  }
}
```

## 5. 创建采集 Session

### 接口

```text
POST /api/sessions
Content-Type: application/json
```

### 请求字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `color_indices` | `number[]` | 否 | `[1,2,3]` | 颜色序号，当前必须是 `[1,2,3]` |
| `colors_rgb` | `number[][]` | 否 | 空 | 可不传；如果传，必须等于 `[[255,20,255],[20,255,20],[255,20,20]]` |
| `total_seconds` | `number` | 否 | `4.0` | 录制总时长，可调整 |
| `warmup_seconds` | `number` | 否 | `1.0` | 当前必须是 `1.0` |
| `hold_seconds` | `number` | 否 | `0.35` | 当前必须是 `0.35` |
| `restore_seconds` | `number` | 否 | `0.0` | 当前必须是 `0.0` |
| `cycles` | `number` | 否 | `1` | 当前必须是 `1` |
| `tail_seconds` | `number` | 否 | `0.5` | 当前必须是 `0.5` |
| `fps` | `number` | 否 | `30` | 前端生成刺激 timeline 时使用，范围 `[1,120]` |
| `width` | `number` | 否 | `1080` | 采集页面参考宽度，范围 `[64,3840]` |
| `height` | `number` | 否 | `1920` | 采集页面参考高度，范围 `[64,3840]` |
| `codec` | `string` | 否 | `mp4v` | 4 字符 FourCC；浏览器实际编码由 MediaRecorder 决定 |

### 请求示例

```bash
curl -k -X POST "https://127.0.0.1:18132/api/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "color_indices": [1, 2, 3],
    "total_seconds": 4.0,
    "warmup_seconds": 1.0,
    "hold_seconds": 0.35,
    "restore_seconds": 0.0,
    "tail_seconds": 0.5,
    "fps": 30,
    "width": 1080,
    "height": 1920
  }'
```

### 返回示例

```json
{
  "session_id": "5eec32b6b4f1",
  "metadata_url": "/api/sessions/5eec32b6b4f1/metadata",
  "recording_upload_url": "/api/sessions/5eec32b6b4f1/recording",
  "stimulus": {
    "protocol_name": "fixed_collect_protocol",
    "requested_duration_seconds": 4.0,
    "warmup_seconds": 1.0,
    "hold_seconds": 0.35,
    "restore_seconds": 0.0,
    "tail_seconds": 0.5,
    "color_order_rgb": [[255, 20, 255], [20, 255, 20], [255, 20, 20]],
    "color_order_packed": [16717055, 1376020, 16716820],
    "timeline": []
  }
}
```

后端需要把返回中的 `session_id`、`recording_upload_url` 和 `stimulus` 传给前端。

## 6. 前端/手机端采集要求

前端拿到 `stimulus.timeline` 后，需要：

1. 请求摄像头权限。
2. 进入全屏或尽量全屏显示闪光颜色。
3. 按 timeline 切换颜色。
4. 同时使用浏览器 `MediaRecorder` 录制摄像头视频。
5. 录制结束后上传视频到 `recording_upload_url`。

浏览器摄像头权限通常要求安全上下文：

```text
https://...
http://localhost
http://127.0.0.1
```

如果通过 VSCode 端口转发到本机，推荐前端访问：

```text
https://127.0.0.1:18132/
```

## 7. 上传录制视频

### 接口

```text
POST /api/sessions/{session_id}/recording
Content-Type: multipart/form-data
```

### 表单字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | file | 是 | 浏览器或手机端录制出来的视频 |
| `client_metadata` | string | 否 | JSON 字符串，记录设备、浏览器、业务流水号等 |

支持视频后缀：

```text
.webm .mp4 .mov .mkv .avi .m4v
```

服务端按上传文件的原始后缀保存视频，不做 WebM 到 MP4 的二次转码；同时会基于该上传视频的实际 `frame_count/fps` 生成同名 txt。返回中的 `recording_path` 指向实际保存的视频文件。

### 请求示例

```bash
curl -k -X POST "https://127.0.0.1:18132/api/sessions/5eec32b6b4f1/recording" \
  -F "file=@recording.webm" \
  -F 'client_metadata={"source":"browser","device":"android","biz_id":"case_001"}'
```

### 返回示例

```json
{
  "session_id": "5eec32b6b4f1",
  "recording_path": "/supercloud/llm-code/scc/scc/Liveness_Detection/flash_collect_stimulus_api_service/outputs/5eec32b6b4f1/recording_001.webm",
  "recording_txt_path": "/supercloud/llm-code/scc/scc/Liveness_Detection/flash_collect_stimulus_api_service/outputs/5eec32b6b4f1/recording_001.txt",
  "recording_bundle_url": "/api/sessions/5eec32b6b4f1/recordings/recording_001.zip",
  "recording_bytes": 123456,
  "recording_txt_bytes": 1024,
  "recording_protocol": {
    "txt_path": "/supercloud/llm-code/scc/scc/Liveness_Detection/flash_collect_stimulus_api_service/outputs/5eec32b6b4f1/recording_001.txt",
    "frame_count": 120,
    "fps": 30.0,
    "duration_seconds": 4.0,
    "used_decode_frame_count": false,
    "protocol_mode": "fixed_collect_protocol_recording_duration",
    "warmup_seconds": 1.0,
    "hold_seconds": 0.35,
    "restore_seconds": 0.0,
    "tail_seconds": 0.5,
    "colors_rgb": [[255, 20, 255], [20, 255, 20], [255, 20, 20]],
    "color_ints": [16717055, 1376020, 16716820]
  },
  "metadata_url": "/api/sessions/5eec32b6b4f1/metadata"
}
```

上传成功后，服务端会在 session 目录下生成：

```text
recording_001.webm
recording_001.txt
metadata.json
```

`recording_001.txt` 是按上传视频实际 `frame_count/fps` 生成的逐帧颜色标签，格式：

```text
frame_index,packed_rgb_color
0,0
1,0
...
```

## 8. 查询 Session 元数据

### 接口

```text
GET /api/sessions/{session_id}/metadata
```

### 请求示例

```bash
curl -k "https://127.0.0.1:18132/api/sessions/5eec32b6b4f1/metadata"
```

### 返回重点字段

```json
{
  "session_id": "5eec32b6b4f1",
  "request": {},
  "stimulus": {},
  "recordings": [
    {
      "path": ".../recording_001.webm",
      "txt_path": ".../recording_001.txt",
      "bundle_url": "/api/sessions/5eec32b6b4f1/recordings/recording_001.zip",
      "client_metadata": {},
      "protocol": {}
    }
  ]
}
```

如果一个 session 上传多次视频，`recordings` 会追加：

```text
recording_001.webm
recording_001.mp4
recording_002.mp4
recording_003.mp4
```

## 9. 下载视频和 TXT 压缩包

### 接口

```text
GET /api/sessions/{session_id}/recordings/{recording_stem}.zip
```

`recording_stem` 示例：

```text
recording_001
```

### 请求示例

```bash
curl -k -L -o 5eec32b6b4f1_recording_001_video_txt.zip \
  "https://127.0.0.1:18132/api/sessions/5eec32b6b4f1/recordings/recording_001.zip"
```

zip 内包含：

```text
5eec32b6b4f1_recording_001.webm
5eec32b6b4f1_recording_001.txt
```

上传接口返回的 `recording_bundle_url` 是相对路径，外部后端可以用下面的规则拼出完整 zip 地址：

```text
zip_url = COLLECT_BASE + recording_bundle_url
```

例如：

```text
https://192.168.17.175:18132/api/sessions/5eec32b6b4f1/recordings/recording_001.zip
```

采集页面上传成功后会同时显示两个入口：

```text
下载录制视频+TXT
打开下载链接
```

部分手机浏览器或自签名 HTTPS 场景会忽略 HTML `download` 属性，导致第一个按钮没有保存 zip。此时点击 `打开下载链接`，浏览器会直接打开 `recording_bundle_url` 对应的 zip 地址进行下载。

## 10. 调用 V3 活体推理

V3 推理服务默认地址：

```text
http://127.0.0.1:18131
```

推荐用上传后返回的 `recording_path` 和同名 `recording_txt_path` 调用；视频后缀以实际上传保存结果为准，可能是 `.webm`，也可能是 `.mp4`：

```bash
curl -X POST "http://127.0.0.1:18131/predict" \
  -F "file=@/supercloud/llm-code/scc/scc/Liveness_Detection/flash_collect_stimulus_api_service/outputs/5eec32b6b4f1/recording_001.webm" \
  -F "txt_file=@/supercloud/llm-code/scc/scc/Liveness_Detection/flash_collect_stimulus_api_service/outputs/5eec32b6b4f1/recording_001.txt"
```

不要把外部刺激视频的 txt 直接绑定到浏览器录制视频上。浏览器录制视频的真实 FPS、帧数、起止时刻可能和刺激页面参数有偏差，必须使用本服务上传后生成的同名 txt。

## 11. Python 后端示例

```python
from pathlib import Path

import requests


COLLECT_BASE = "https://192.168.17.175:18132"
V3_BASE = "http://127.0.0.1:18131"
REQUESTS_VERIFY = False  # 自签名证书用 False；可信 CA 证书请改为 True


def create_collect_session(total_seconds: float = 4.0) -> dict:
    payload = {
        "color_indices": [1, 2, 3],
        "total_seconds": total_seconds,
        "warmup_seconds": 1.0,
        "hold_seconds": 0.35,
        "restore_seconds": 0.0,
        "tail_seconds": 0.5,
        "fps": 30,
        "width": 1080,
        "height": 1920,
    }
    response = requests.post(
        f"{COLLECT_BASE}/api/sessions",
        json=payload,
        timeout=10,
        verify=REQUESTS_VERIFY,
    )
    response.raise_for_status()
    return response.json()


def upload_recording(session_id: str, recording_path: str | Path, metadata: dict | None = None) -> dict:
    recording_path = Path(recording_path)
    client_metadata = metadata or {}
    with recording_path.open("rb") as file:
        response = requests.post(
            f"{COLLECT_BASE}/api/sessions/{session_id}/recording",
            files={"file": (recording_path.name, file)},
            data={"client_metadata": json_dumps(client_metadata)},
            timeout=120,
            verify=REQUESTS_VERIFY,
        )
    response.raise_for_status()
    return response.json()


def download_recording_zip(uploaded: dict, save_path: str | Path | None = None) -> Path:
    bundle_url = uploaded["recording_bundle_url"]
    zip_url = f"{COLLECT_BASE}{bundle_url}"
    target_path = Path(save_path or f"{uploaded['session_id']}_recording_001_video_txt.zip")
    response = requests.get(zip_url, timeout=120, verify=REQUESTS_VERIFY)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


def infer_v3(video_path: str | Path, txt_path: str | Path) -> dict:
    with Path(video_path).open("rb") as video_file, Path(txt_path).open("rb") as txt_file:
        response = requests.post(
            f"{V3_BASE}/predict",
            files={
                "file": (Path(video_path).name, video_file),
                "txt_file": (Path(txt_path).name, txt_file),
            },
            timeout=180,
        )
    response.raise_for_status()
    return response.json()


session = create_collect_session(total_seconds=4.0)
print("session_id:", session["session_id"])
print("upload_url:", f"{COLLECT_BASE}{session['recording_upload_url']}")

# 这里通常由前端录制并上传；如果后端已经拿到录制文件，可以调用：
# uploaded = upload_recording(session["session_id"], "/path/to/recording.webm")
# zip_path = download_recording_zip(uploaded)
# print("zip_path:", zip_path)
# result = infer_v3(uploaded["recording_path"], uploaded["recording_txt_path"])
# print(result)
```

## 12. 常见错误

| HTTP 状态 | detail | 原因 |
| ---: | --- | --- |
| 400 | `color_indices must be [1, 2, 3]...` | 颜色顺序不是 V3 固定协议 |
| 400 | `warmup_seconds must be 1.0...` | warmup 参数不符合当前 V3 协议 |
| 400 | `hold_seconds must be 0.35...` | 每个颜色保持时长不符合当前 V3 协议 |
| 400 | `restore_seconds must be 0.0...` | 当前 V3 协议要求连续切色，无恢复间隔 |
| 400 | `tail_seconds must be 0.5...` | 黑尾时长不符合当前 V3 协议 |
| 400 | `total_seconds must be greater than warmup_seconds + tail_seconds` | 总时长太短 |
| 400 | `unsupported_recording_type:.xxx` | 上传的视频后缀不支持 |
| 404 | `session_not_found` | 上传视频时 session_id 不存在 |
| 404 | `metadata_not_found` | 查询的 session 不存在 |
| 404 | `recording_bundle_not_found` | 下载 zip 时 recording_stem 不存在 |
| 500 | `recording_txt_generation_failed:...` | 服务端无法读取视频帧数/FPS 或生成 txt |

## 13. 对接建议

- 后端保存业务流水号时，放入上传接口的 `client_metadata`。
- 前端录制完成后，优先上传原始 `webm/mp4`，不要自行改帧率或转码。
- V3 推理时必须使用服务端生成的同名 `recording_*.txt`。
- 如果手机浏览器拿不到摄像头权限，使用 HTTPS，或通过端口转发后访问 `http://127.0.0.1:18132/`。
- 如果要做线上联调，建议在后端记录 `session_id`、`recording_path`、`recording_txt_path`、`recording_protocol.frame_count`、`recording_protocol.fps` 和 V3 推理结果。

## 14. 手机录制失败：安全上下文问题

如果页面提示：

```text
录制失败: Error: 当前页面不是安全上下文，浏览器不会开放摄像头 API。
手机访问服务器 IP 时不能使用普通 http://192.168.17.175:18132。
请改用 HTTPS，或在 Android 调试时用 adb reverse 后访问 http://127.0.0.1:18132。
```

这不是采集协议错误，也不是后端接口不可用，而是浏览器摄像头权限策略。现代 Chrome、Safari、Edge 只允许在安全上下文中调用摄像头 API：

```text
https://...
http://localhost
http://127.0.0.1
```

手机直接访问普通 HTTP 服务器 IP，例如：

```text
http://192.168.17.175:18132/
```

不属于安全上下文，所以 `navigator.mediaDevices.getUserMedia` 会被浏览器禁用。

### 推荐方案 A：HTTPS 域名或 HTTPS 反向代理

给他人使用或多人联调时，推荐把采集服务放到 HTTPS 后面：

```text
https://collect.example.com/  ->  http://127.0.0.1:18132/
```

Caddy 示例：

```caddyfile
collect.example.com {
  reverse_proxy 127.0.0.1:18132
}
```

Nginx 示例：

```nginx
server {
    listen 443 ssl;
    server_name collect.example.com;

    ssl_certificate     /path/to/fullchain.pem;
    ssl_certificate_key /path/to/privkey.pem;

    client_max_body_size 200M;

    location / {
        proxy_pass http://127.0.0.1:18132;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
    }
}
```

然后让手机访问：

```text
https://collect.example.com/
```

### 推荐方案 B：采集服务直接 HTTPS 启动

如果已有证书，也可以直接用服务自身 HTTPS 参数启动：

```bash
CUDA_VISIBLE_DEVICES=1 /home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python \
  /supercloud/llm-code/scc/scc/Liveness_Detection/flash_collect_stimulus_api_service/app.py \
  --host 0.0.0.0 \
  --port 18132 \
  --ssl-certfile /path/to/fullchain.pem \
  --ssl-keyfile /path/to/privkey.pem
```

访问：

```text
https://服务器域名或IP:18132/
```

注意：手机浏览器通常不信任自签名证书。给他人使用时，应使用可信 CA 证书或内网统一下发受信任证书。

### 推荐方案 C：Android 本机调试，使用 adb reverse

适合开发人员手机 USB 调试，不适合大规模给他人使用。

如果采集服务已经在当前电脑本机可访问：

```text
http://127.0.0.1:18132/
```

执行：

```bash
adb reverse tcp:18132 tcp:18132
```

然后 Android 手机 Chrome 访问：

```text
http://127.0.0.1:18132/
```

此时手机里的 `127.0.0.1:18132` 会被 adb 转发到电脑的 `127.0.0.1:18132`，浏览器会把它当作 localhost 安全上下文，摄像头 API 可以打开。

如果服务在远程服务器上，且通过 VSCode SSH 端口转发到本机，需要链路是：

```text
远程服务器 18132
  -> VSCode SSH 转发到本机 127.0.0.1:18132
  -> adb reverse 到手机 127.0.0.1:18132
```

手机最终仍访问：

```text
http://127.0.0.1:18132/
```

### 推荐方案 D：iPhone/Safari 调试

iPhone 不支持 `adb reverse`。给 iPhone 或外部用户使用时，优先使用 HTTPS 域名：

```text
https://collect.example.com/
```

如果只是本机开发调试，可以使用 Mac Safari Web Inspector、iOS 可信证书配置或局域网 HTTPS 代理，但最终原则仍是：页面必须是 HTTPS 或 localhost。

### 不推荐方案

不要让用户直接使用：

```text
http://服务器IP:18132/
http://192.168.x.x:18132/
http://10.x.x.x:18132/
```

这些地址能打开页面，但手机浏览器不会开放摄像头，所以会录制失败。
