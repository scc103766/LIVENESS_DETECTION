# LIVENESS_DETECTION

Face liveness detection workspace for bank-based identity matching, anti-spoofing inference, flash-liveness training research, and supporting face-detection utilities.

This GitHub release is organized in a code-first form. Large datasets, trained weights, archived environments, and generated outputs are intentionally excluded from version control. See [DATA_ASSETS.md](/supercloud/llm-code/scc/scc/Liveness_Detection/github_release/DATA_ASSETS.md) for details.

## Main Capabilities

- YOLOv7-face based face detection and alignment
- ArcFace style face embedding extraction and bank matching
- DeePixBiS based anti-spoofing inference
- Fused liveness decision pipeline with bank rules and fallback thresholds
- Flash-liveness training scripts for video-based experiments
- FastAPI inference services, including a V2 service with a simple web upload UI
- A curated code-only mirror of the `20240320闪光活体归档` archive

## Key Entry Points

### Bank And Liveness APIs

- `fused_face_liveness_api.py`
  - production-style FastAPI JSON API
  - applies bank matching plus DeePixBiS fallback logic

- `fused_face_liveness_api_v2.py`
  - V2 API with JSON interface and a simple browser upload page
  - front-end uses relative paths so it works behind localhost, LAN IPs, or reverse proxies without hardcoded host changes

- `fused_face_liveness_eval.py`
  - offline evaluation script for bank/test-set verification and result logging

- `FUSED_FACE_LIVENESS_API_README.md`
  - operation manual for launching and calling the API

### Face Detection And Recognition

- `Face_detection_yolo_align.py`
  - wrapper around YOLOv7-face detection, keypoints, and face alignment

- `face_interactive_liveness.py`
  - bank similarity based face comparison workflow

- `Face_compare.py`
  - pairwise face-comparison utility for offline analysis

### Flash Liveness Training

- `flash_liveness_project.py`
  - flash-liveness training and inference script

- `flash_liveness_project_v2.py`
  - V2 version with full-frame sequential reading and per-frame color annotations

- `scripts/prepare_flash_liveness_video_dataset.py`
- `scripts/prepare_flash_liveness_video_dataset_v2.py`
  - dataset preparation helpers

### Archived Module

- `archive_20240320_flash_liveness/`
  - code-and-doc only mirror of the local `20240320闪光活体归档`
  - keeps `ThunderGuard` and `FaceAlign` source trees
  - excludes datasets, conda environments, model weights, ONNX exports, and build outputs

## Repository Layout

```text
LIVENESS_DETECTION/
├── archive_20240320_flash_liveness/
├── backbones/
├── scripts/
├── yolov7_face/
├── Face-Anti-Spoofing-using-DeePixBiS/
├── fused_face_liveness_api.py
├── fused_face_liveness_api_v2.py
├── fused_face_liveness_eval.py
├── flash_liveness_project.py
├── flash_liveness_project_v2.py
├── face_interactive_liveness.py
├── Face_detection_yolo_align.py
├── README.md
├── DATA_ASSETS.md
├── FUSED_FACE_LIVENESS_API_README.md
└── GITHUB_RELEASE_GUIDE.md
```

## Omitted Assets

The local working directory originally contained many non-code assets, including:

- raw datasets and videos
- prepared bank images
- experiment archives
- environment snapshots
- large checkpoints and exported ONNX files
- generated evaluation images and logs

These are excluded from Git tracking. See [DATA_ASSETS.md](/supercloud/llm-code/scc/scc/Liveness_Detection/github_release/DATA_ASSETS.md).

## Environment Notes

The local team often runs code with:

- `/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python`

Typical dependencies include:

- `torch`
- `torchvision`
- `opencv-python`
- `numpy`
- `fastapi`
- `uvicorn`
- `faiss`

Check `requirements.txt` and subproject requirements before reproducing a full environment.

## API Rule Summary

The current bank-based API rule is:

1. Detect and align the face.
2. Compare against a labeled bank.
3. A bank hit is valid only when `FR min_score > 0.49`.
4. If the matched bank image is labeled real, return real.
5. If the matched bank image is labeled fake or head-model, return fake.
6. If bank matching fails, use DeePixBiS fallback.
7. If `DeePixBiS pixel_score > 0.68`, return real.
8. Otherwise, return fake.

## Notes For GitHub Upload

- `Face-Anti-Spoofing-using-DeePixBiS/` was originally present as an embedded Git repository in the local workspace.
- The raw `20240320闪光活体归档/` contains large data and environment snapshots, so only a curated code-only mirror is committed here.
- See [GITHUB_RELEASE_GUIDE.md](/supercloud/llm-code/scc/scc/Liveness_Detection/github_release/GITHUB_RELEASE_GUIDE.md) for release intent.
