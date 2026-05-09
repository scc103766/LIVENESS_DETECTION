# Fused Face Liveness API V2 操作手册

## 1. 功能概述

`fused_face_liveness_api_v2.py` 是 `V2` 版本的人脸活体服务。

这版同时提供两种使用方式：

- 网页上传界面
- JSON 推理接口

它适合测试、演示和局域网部署。网页前端使用相对路径提交请求，不依赖固定 IP，所以同一套页面可以通过：

- `127.0.0.1`
- 局域网 IP
- 域名
- 反向代理地址

直接访问，而不需要改前端代码。

与主接口版本的区别：

- `fused_face_liveness_api.py` 使用 ThunderGuard `MoEA_score.onnx` 闪光活体方案。
- `fused_face_liveness_api_v2.py` 继续使用之前的 `DeePixBiS` 静态图活体方案。
- `V2` 内部单独实现 `bank` 标签识别、底库构建、图片读取、请求解析和匹配逻辑，不再复用 `V1` 的 `bank` 处理代码。
- 两个脚本代表两套不同处理方法，启动参数、返回字段和业务规则不要混用。

脚本路径：

`/supercloud/llm-code/scc/scc/Liveness_Detection/fused_face_liveness_api_v2.py`

## 2. 判定逻辑

服务使用当前已经验证过的融合规则：

1. 先做人脸检测和对齐。
2. 与 `bank` 底库做人脸特征匹配。
3. 只有 `FR min_score > 0.49` 才算命中 `bank`。
4. `bank` 可以同时放真人图和头模/攻击图。
5. 命中 `bank` 中的真人图，输出真人。
6. 命中 `bank` 中的头模/攻击图，输出假人。
7. 如果没有命中任何 `bank` 样本，则进入 `DeePixBiS` 兜底。
8. 当 `DeePixBiS pixel_score > 0.68` 时，输出真人。
9. 否则输出假人。

外部活体调用标记：

- 命中 `bank` 时：`need_external_liveness = false`
- 未命中 `bank` 时：`need_external_liveness = true`
- 这个字段只表示“是否需要调用其他活体检测”，不直接等同于最终真人/假人结果。

默认 `bank` 目录：

`/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/bank`

## 3. 依赖要求

建议在这套环境运行：

`/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python`

请先确认环境里已经安装：

- `fastapi`
- `uvicorn`
- `torch`
- `torchvision`
- `opencv-python`
- `numpy`

如果需要 URL 拉图，还需要能访问外网或内网图片地址。

## 4. 启动方式

推荐启动命令：

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python \
/supercloud/llm-code/scc/scc/Liveness_Detection/fused_face_liveness_api_v2.py \
--host 0.0.0.0 \
--port 18120 \
--gallery-dir /supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/bank \
--fr-threshold 0.49 \
--spoof-threshold 0.68 \
--device cuda
```

说明：

- `--host 0.0.0.0` 表示监听所有网卡，便于别人访问。
- `--port 18120` 是 `V2` 默认端口。
- `--gallery-dir` 指向同时包含真人和头模/攻击样本的 `bank` 底库目录。
- `bank` 图片文件名需要能识别标签。系统会按文件名分词精确判断，当前目录中的 `*_true.png` 会归为真人，`*_toumo.jpg/png` 会归为假人。
- 兼容标签词包括：`true/genuine/real/local/live` 归为真人，`toumo/head/model/fake/spoof` 归为假人。
- `--device` 可以写 `cuda`、`cuda:0` 或 `cpu`。

启动后默认地址如下：

- 首页：`http://127.0.0.1:18120/`
- 网页页签：`http://127.0.0.1:18120/ui`
- 健康检查：`http://127.0.0.1:18120/health`
- JSON 接口：`http://127.0.0.1:18120/predict`
- 网页表单提交：`http://127.0.0.1:18120/predict-ui`

如果局域网地址是 `192.168.17.175`，则外部访问可写成：

- `http://192.168.17.175:18120/`

## 5. 网页上传使用方法

浏览器打开：

`http://192.168.17.175:18120/`

页面能力包括：

- 选择本地图片文件上传
- 显示最终真人/假人结果
- 显示 `FR min_score`
- 显示 `DeePixBiS pixel_score`
- 显示最佳 `bank` 匹配及其标签
- 显示结果依据 `basis`
- 可选显示绘框图和对齐人脸图

适用场景：

- 给测试同事直接人工验证
- 给业务同事演示
- 快速检查 `bank` 真人/头模强判断规则和阈值是否符合预期

## 6. JSON 接口说明

### 6.1 健康检查

```bash
curl http://192.168.17.175:18120/health
```

### 6.2 接口地址

```text
POST /predict
```

### 6.3 支持的输入方式

`/predict` 支持以下几种输入：

- `multipart/form-data` 上传文件
- `application/json` 传 `image_path`
- `application/json` 传 `image_base64`
- `application/json` 传 `image_url`
- 原始图片二进制请求体

重要说明：

- `image_path` 只适合服务机器本地调用
- 外部调用优先推荐：
  - 文件上传
  - `image_base64`
  - `image_url`

## 7. 调用示例

### 7.1 文件上传

```bash
curl -X POST "http://192.168.17.175:18120/predict" \
  -F "file=@/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test/fake_0001_00_00_01_0.jpg"
```

### 7.2 本地路径调用

这个方式只适合服务所在机器本地调用。

```bash
curl -X POST "http://192.168.17.175:18120/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test/genuine_0001_00_00_02_0.jpg"
  }'
```

### 7.3 Base64 调用

```bash
IMG_B64=$(base64 -w 0 /supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test/genuine_0001_00_00_02_0.jpg)

curl -X POST "http://192.168.17.175:18120/predict" \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\":\"$IMG_B64\"}"
```

### 7.4 URL 调用

```bash
curl -X POST "http://192.168.17.175:18120/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/test_face.jpg"
  }'
```

### 7.5 Python 调用

```python
import requests

url = "http://192.168.17.175:18120/predict"
image_path = "/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test/genuine_0001_00_00_02_0.jpg"

with open(image_path, "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.status_code)
print(response.json())
```

## 8. JSON 返回说明

建议优先看下面字段：

1. `binary_result`
   最终二分类结果，取值为 `real` 或 `fake`。
2. `fr.min_score`
   是否命中 `bank` 的关键分数；大于 `0.49` 才认为命中。
3. `bank_result.result`
   `bank` 命中状态，取值为 `bank_real_match`、`bank_fake_match`、`bank_real_exact`、`bank_fake_exact`、`not_in_bank`。
4. `need_external_liveness`
   命中 `bank` 时为 `false`，未命中 `bank` 时为 `true`。
5. `spoof.pixel_score`
   未命中 `bank` 时，是否通过 `DeePixBiS > 0.68` 的真人兜底规则。
6. `basis`
   最终结果的文字解释。

如果调用方需要根据是否命中 `bank` 决定是否继续调用其他活体检测，优先看下面两个字段：

- `bank_result.result`
  取值为 `bank_real_match`、`bank_fake_match`、`bank_real_exact`、`bank_fake_exact`、`not_in_bank`。
- `need_external_liveness`
  命中 `bank` 时为 `false`，未命中 `bank` 时为 `true`。未命中时调用方可以继续调用其他活体检测模型。

常见返回字段包括：

- `result`
- `binary_result`
- `basis`
- `bank_result`
- `need_external_liveness`
- `external_liveness_next_action`
- `fr`
- `spoof`
- `best_bank_match`
- `best_bank_label`
- `annotated_image_path`
- `aligned_face_path`
- `annotated_image_base64`
- `aligned_face_base64`

返回示例：

未命中 `bank`，需要继续调用其他活体检测时：

```json
{
  "result": "live_not_in_bank",
  "binary_result": "real",
  "need_external_liveness": true,
  "external_liveness_next_action": "call_external_liveness",
  "bank_result": {
    "result": "not_in_bank",
    "matched": false,
    "label": "unmatched",
    "binary_result": "unknown",
    "match_type": "none",
    "score": 0.443804,
    "threshold": 0.49,
    "matched_path": "/path/to/best_bank_candidate.jpg",
    "need_external_liveness": true,
    "next_action": "call_external_liveness"
  },
  "basis": [
    "No bank match above threshold=0.49",
    "Best bank score=0.443804",
    "DeePixBiS pixel_score=0.693180 > threshold=0.68",
    "Did not match any prepared bank identity, and DeePixBiS score supports real."
  ],
  "fr": {
    "label": "unmatched",
    "min_score": 0.443804,
    "max_score": 0.443804,
    "threshold": 0.49
  },
  "spoof": {
    "pixel_score": 0.693180,
    "binary_score": 0.955090,
    "combined_score": 0.824135,
    "label": "real"
  }
}
```

命中 `bank` 真人图，不需要继续调用其他活体检测时：

```json
{
  "result": "bank_real_match",
  "binary_result": "real",
  "need_external_liveness": false,
  "external_liveness_next_action": "use_bank_result",
  "bank_result": {
    "result": "bank_real_match",
    "matched": true,
    "label": "real",
    "binary_result": "real",
    "match_type": "fr_embedding",
    "score": 0.612345,
    "threshold": 0.49,
    "matched_path": "/path/to/184838b7-f5de-4290-8fbd-3b2c9e335127_0_true.png",
    "need_external_liveness": false,
    "next_action": "use_bank_result"
  }
}
```

命中 `bank` 头模/攻击图，不需要继续调用其他活体检测时：

```json
{
  "result": "bank_fake_match",
  "binary_result": "fake",
  "need_external_liveness": false,
  "external_liveness_next_action": "use_bank_result",
  "bank_result": {
    "result": "bank_fake_match",
    "matched": true,
    "label": "fake",
    "binary_result": "fake",
    "match_type": "fr_embedding",
    "score": 0.588888,
    "threshold": 0.49,
    "matched_path": "/path/to/20260407-103317_toumo.jpg",
    "need_external_liveness": false,
    "next_action": "use_bank_result"
  }
}
```

## 9. 部署建议

### 9.1 局域网部署

最简单的做法：

- 服务端用 `--host 0.0.0.0`
- 固定端口，比如 `18120`
- 通知别人用局域网地址访问，例如：
  - `http://192.168.17.175:18120/`

需要确保：

- 本机防火墙放行 `18120`
- 所在网络允许其他机器访问这台设备

### 9.2 反向代理部署

如果想让访问地址更稳定，建议在 `nginx` 或网关后面做代理：

- 外部统一访问域名
- 代理到 `127.0.0.1:18120`

这样即使机器 IP 变化，前端页面也不用改，因为页面提交的是相对路径。

### 9.3 公网部署注意事项

如果要对公网开放，建议额外处理：

- 访问鉴权
- 请求大小限制
- 并发限制
- 超时设置
- HTTPS
- 图片 URL 白名单或下载超时

### 9.4 模型与资源路径建议

启动时尽量显式指定：

- `--gallery-dir`
- `--yolo-path`
- `--arcface-path`
- `--deepixbis-weights`
- `--output-dir`

这样更适合迁移到新机器。

## 10. 常见问题

### 为什么网页能在不同地址下工作

因为 `V2` 页面提交的是：

- `/predict-ui`
- `/predict`

这些都是相对路径，不是写死某个 IP。

### 为什么外部调用不建议传 `image_path`

因为 `image_path` 是服务机器自己的本地路径，不是调用方电脑上的路径。对外调用请优先用文件上传、`base64` 或 `URL`。

### 如何判断最终是真人还是假人

先看：

- `binary_result`

再结合：

- `fr.min_score`
- `spoof.pixel_score`
- `basis`

这里的核心理解是：

- `bank` 现在同时负责“已知真人放行”和“已知头模/攻击拦截”
- 命中 `bank` 且 `FR min_score > 0.49` 时，优先采用 `bank` 标签作为最终判断
- 没命中 `bank` 时，当前 V2 仍会按 `DeePixBiS > 0.68` 给出 `binary_result`
- 如果你要额外接入其他活体检测，则以 `need_external_liveness` 为触发开关：命中 `bank` 为 `false`，未命中 `bank` 为 `true`
- 如果你要知道具体是否命中 `bank`，以 `bank_result.result` 为准

如果后面你继续调整阈值或 `bank` 规则，这份文档也应该同步更新。
