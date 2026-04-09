# Fused Face Liveness API 操作手册

## 1. 功能说明

`fused_face_liveness_api.py` 提供一个单张图片推理接口。

接口处理逻辑如下：

1. 输入图片后，先做人脸检测、对齐和特征提取。
2. 使用 `bank` 底库进行人脸特征匹配。
3. 只有 `FR min_score > 0.49` 才算命中 `bank`。
4. 如果命中 `bank` 中的真人图片，返回真人。
5. 如果命中 `bank` 中的头模图片，返回假人。
6. 如果没有命中 `bank`，则使用 `DeePixBiS` 做兜底判断。
7. 当 `DeePixBiS pixel_score > 0.68` 时，返回真人。
8. 否则返回假人。

当前默认 `bank` 目录：

`/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/bank`

API 脚本路径：

`/supercloud/llm-code/scc/scc/Liveness_Detection/fused_face_liveness_api.py`

## 2. 启动方式

请先确认运行环境中已经安装：

- `fastapi`
- `uvicorn`
- `torch`
- `opencv-python`

启动命令：

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python \
/supercloud/llm-code/scc/scc/Liveness_Detection/fused_face_liveness_api.py \
--host 0.0.0.0 \
--port 18119 \
--gallery-dir /supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/bank \
--fr-threshold 0.49 \
--spoof-threshold 0.68
```

说明：

- `--host 0.0.0.0` 表示服务监听本机所有网卡地址，适合外部机器访问。
- 如果当前服务器对外地址是 `192.168.17.175`，那么别人调用时应使用：
  `http://192.168.17.175:18119`

健康检查：

```bash
curl http://127.0.0.1:18119/health
```

外部调用健康检查示例：

```bash
curl http://192.168.17.175:18119/health
```

## 3. 输入方式

`/predict` 支持以下输入方式：

- `multipart/form-data` 上传图片文件
- `application/json` 传 `image_path`
- `application/json` 传 `image_base64`
- `application/json` 传 `image_url`
- 直接发送原始图片二进制

重要说明：

- `image_path` 是服务所在机器上的本地路径，不是调用方自己电脑上的路径。
- 如果是别人通过外网地址调用，推荐使用：
  - 文件上传
  - `image_base64`
  - `image_url`
- 外部调用通常不要使用 `image_path`，除非传入的是服务器上的绝对路径。

## 4. 调用示例

### 4.1 通过本地图片路径调用

这个方式只适合在服务所在机器本地调用。

```bash
curl -X POST "http://192.168.17.175:18119/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test/genuine_0001_00_00_02_0.jpg"
  }'
```

### 4.2 通过文件上传调用

```bash
curl -X POST "http://192.168.17.175:18119/predict" \
  -F "file=@/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test/fake_0001_00_00_01_0.jpg"
```

### 4.3 通过 Base64 调用

```bash
IMG_B64=$(base64 -w 0 /supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test/genuine_0001_00_00_02_0.jpg)

curl -X POST "http://192.168.17.175:18119/predict" \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\":\"$IMG_B64\"}"
```

### 4.4 通过 Python 调用

```python
import requests

url = "http://192.168.17.175:18119/predict"
image_path = "/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test/genuine_0001_00_00_02_0.jpg"

with open(image_path, "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.status_code)
print(response.json())
```

## 5. 返回结果说明

接口返回 JSON，核心字段如下：

- `result`
  最终详细结果类别。

- `binary_result`
  最终二分类结果，接口对外可以直接按这个字段判断：
  - `real`
  - `fake`

- `basis`
  最终结果的文字依据列表。这个字段会直接解释为什么返回真人或假人。

- `fr`
  最终结果中的人脸识别判断依据，重点看：
  - `fr.min_score`
  - `fr.threshold`
  - `fr.label`

- `spoof`
  最终结果中的活体检测判断依据，重点看：
  - `spoof.pixel_score`
  - `spoof.threshold`
  - `spoof.label`

作为最终结果时，建议优先看下面 4 个 JSON 字段，它们就是最终判断依据：

1. `binary_result`
   最终真人/假人结果。
2. `fr.min_score`
   是否通过 `FR > 0.49` 的关键分数。
3. `spoof.pixel_score`
   是否通过 `DeePixBiS > 0.68` 兜底判断的关键分数。
4. `basis`
   用自然语言解释最终结果是如何得到的。

- `result`
  当前详细分类结果，例如：
  - `bank_real_exact`
  - `bank_fake_exact`
  - `bank_real_match`
  - `bank_fake_match`
  - `live_not_in_bank`
  - `fake_not_in_bank`

- `binary_result`
  最终二分类结果：
  - `real`
  - `fake`

- `basis`
  文字版判断依据。

- `fr`
  人脸特征匹配结果，包含：
  - `min_score`
  - `threshold`
  - `label`

- `spoof`
  `DeePixBiS` 输出，包含：
  - `pixel_score`
  - `binary_score`
  - `combined_score`
  - `label`

- `best_bank_match`
  当前最相近的 `bank` 图片路径。

- `best_bank_label`
  最相近 `bank` 图片的标签。

- `annotated_image_path`
  推理结果绘框图路径。

- `aligned_face_path`
  对齐后人脸图路径。

## 6. 返回示例

```json
{
  "result": "live_not_in_bank",
  "binary_result": "real",
  "basis": [
    "No bank match above threshold=0.49",
    "Best bank score=0.443804",
    "DeePixBiS pixel_score=0.693180 > threshold=0.68",
    "Did not match the prepared person or head models in bank, and DeePixBiS > 0.68."
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

## 7. 当前规则总结

- `FR min_score > 0.49` 才算命中 `bank`
- 命中 `bank` 真人，输出真人
- 命中 `bank` 头模，输出假人
- 未命中 `bank` 且 `DeePixBiS pixel_score > 0.68`，输出真人
- 未命中 `bank` 且 `DeePixBiS pixel_score <= 0.68`，输出假人
