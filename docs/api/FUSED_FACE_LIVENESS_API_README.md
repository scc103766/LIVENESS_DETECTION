# Fused Face Liveness API 操作手册

## 1. 功能说明

`fused_face_liveness_api.py` 提供两类接口：

- 单张图片推理：`/predict`
- 本地文件夹批量推理：`/predict-batch`

当前版本已经不再使用 DeePixBiS 作为主活体模型，而是复用 ThunderGuard 的旧模型：

`/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/resources/MoEA_score.onnx`

接口处理逻辑如下：

1. 输入图片后，先做人脸检测、对齐和特征提取。
2. 使用 `bank` 底库进行人脸特征匹配。
3. 只有 `FR min_score > 0.49` 才算命中 `bank`。
4. 当前 `bank` 只用于拦截头模攻击图。
5. 如果命中 `bank` 中的头模图片，返回假人。
6. 如果没有命中 `bank` 头模，则将这张对齐后的人脸构造成 ThunderGuard 所需测试格式。
7. 接口会自动模拟三色打光，生成：
   - ThunderGuard 风格 `xxx.jpg`
   - 对应颜色 `xxx.txt`
   - 占位深度图 `xxx_d.jpg`
8. 再调用 `MoEA_score.onnx` 得到 `flash_liveness.score`。
9. 当 `flash_liveness.score > 0.93` 时，返回真人。
10. 否则返回假人。

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
- `onnxruntime`

启动命令：

```bash
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python \
/supercloud/llm-code/scc/scc/Liveness_Detection/fused_face_liveness_api.py \
--host 0.0.0.0 \
--port 18119 \
--gallery-dir /supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/bank \
--fr-threshold 0.49 \
--flash-threshold 0.93 \
--flash-onnx-path /supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/resources/MoEA_score.onnx
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

### 3.1 单张接口 `/predict`

`/predict` 支持以下输入方式：

- `multipart/form-data` 上传图片文件
- `application/json` 传 `image_path`
- `application/json` 传 `image_base64`
- `application/json` 传 `image_url`
- 直接发送原始图片二进制

重要说明：

- `image_path` 是服务所在机器上的本地路径，不是调用方自己电脑上的路径。
- 如果是别人通过外部地址调用，推荐使用：
  - 文件上传
  - `image_base64`
  - `image_url`

### 3.2 批量接口 `/predict-batch`

`/predict-batch` 用于批量处理服务机器本地目录中的图片。

当前推荐输入参数：

- `folder_path`
- `recursive`
- `limit`
- `return_images`

重要说明：

- `folder_path` 必须是服务器本地目录
- 该接口不会上传整个文件夹，而是读取服务器上已经存在的目录
- 批量接口会逐张调用现有单图逻辑，失败样本会单独返回错误，不影响整批结果

## 4. 调用示例

### 4.1 单张通过本地图片路径调用

这个方式只适合在服务所在机器本地调用。

```bash
curl -X POST "http://192.168.17.175:18119/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test/genuine_0001_00_00_02_0.jpg"
  }'
```

### 4.2 单张通过文件上传调用

```bash
curl -X POST "http://192.168.17.175:18119/predict" \
  -F "file=@/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test/fake_0001_00_00_01_0.jpg"
```

### 4.3 单张通过 Base64 调用

```bash
IMG_B64=$(base64 -w 0 /supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test/genuine_0001_00_00_02_0.jpg)

curl -X POST "http://192.168.17.175:18119/predict" \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\":\"$IMG_B64\"}"
```

### 4.4 批量通过本地目录调用

```bash
curl -X POST "http://192.168.17.175:18119/predict-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test",
    "recursive": true
  }'
```

### 4.5 批量限制只处理前 10 张

```bash
curl -X POST "http://192.168.17.175:18119/predict-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test",
    "recursive": true,
    "limit": 10
  }'
```

### 4.6 Python 单张调用

```python
import requests

url = "http://192.168.17.175:18119/predict"
image_path = "/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test/genuine_0001_00_00_02_0.jpg"

with open(image_path, "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.status_code)
print(response.json())
```

### 4.7 Python 批量调用

```python
import requests

url = "http://192.168.17.175:18119/predict-batch"
payload = {
    "folder_path": "/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test",
    "recursive": True,
}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
```

## 5. 单张返回结果说明

单张接口 `/predict` 返回 JSON，建议优先看下面 4 个字段，它们就是最终判断依据：

1. `binary_result`
   最终真人/假人结果。
2. `fr.min_score`
   是否通过 `FR > 0.49` 的关键分数。
3. `flash_liveness.score`
   ThunderGuard 闪光活体模型输出分数。
4. `basis`
   用自然语言解释最终结果是如何得到的。

常见字段包括：

- `result`
- `binary_result`
- `basis`
- `fr`
- `flash_liveness`
- `best_bank_match`
- `best_bank_label`
- `annotated_image_path`
- `aligned_face_path`
- `thunderguard_sample_jpg`
- `thunderguard_sample_txt`
- `thunderguard_sample_depth_path`

`flash_liveness` 中重点字段：

- `score`
- `threshold`
- `color_validation_pass`
- `color_triplets_rgb`
- `color_triplets_packed`

## 6. 批量返回结果说明

批量接口 `/predict-batch` 会返回：

- `folder_path`
- `recursive`
- `requested_count`
- `success_count`
- `failed_count`
- `real_count`
- `fake_count`
- `results`

其中：

- `results` 是逐张结果列表
- 每个成功样本会返回和 `/predict` 基本一致的字段
- 每个失败样本会返回：
  - `status = error`
  - `source_value`
  - `error`

## 7. 单张返回示例

```json
{
  "result": "live_not_in_bank",
  "binary_result": "real",
  "basis": [
    "No bank match above threshold=0.49",
    "Best bank score=0.443804",
    "ThunderGuard flash score=0.945100 > threshold=0.93",
    "ThunderGuard color_validation_pass=true",
    "Did not match any head model in bank; synthetic ThunderGuard flash-liveness score is above threshold, so real probability is high."
  ],
  "fr": {
    "label": "unmatched",
    "min_score": 0.443804,
    "max_score": 0.443804,
    "threshold": 0.49
  },
  "flash_liveness": {
    "label": "real",
    "score": 0.9451,
    "threshold": 0.93,
    "color_validation_pass": true
  }
}
```

## 8. 批量返回示例

```json
{
  "folder_path": "/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/test",
  "recursive": true,
  "requested_count": 20,
  "success_count": 18,
  "failed_count": 2,
  "real_count": 9,
  "fake_count": 9,
  "results": []
}
```

## 9. 当前规则总结

- `FR min_score > 0.49` 才算命中 `bank`
- `bank` 当前只用于拦截头模攻击
- 命中 `bank` 头模，输出假人
- 未命中 `bank` 时，构造 ThunderGuard 测试样本并调用 `MoEA_score.onnx`
- `flash_liveness.score > 0.93`，输出真人
- `flash_liveness.score <= 0.93`，输出假人
