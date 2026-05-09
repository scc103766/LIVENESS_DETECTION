# Data And Asset Notes

## Scope

This repository is prepared for GitHub publication in a code-first form.
Raw datasets, videos, environment exports, large evaluation outputs, and local secrets are intentionally excluded from version control. The V3 fixed-protocol best checkpoint is the only weight included in this release, stored as split parts under `weights/flash_liveness_v3_fixed_protocol/`.

## Excluded Asset Groups

### Datasets

- `dataset/`
  - Local experiment outputs and prepared training splits.
- `Face-Anti-Spoofing-using-DeePixBiS/data/`
  - DeePixBiS sample images, bank images, and local test sets.
- `20240320闪光活体归档/`
  - Archived flash-liveness project, datasets, and environment snapshot.
- `炫彩闪烁活体/`
  - Historical flash-liveness data and archives.

### Model Weights

The following kinds of files are excluded by default:

- `*.pt`
- `*.pth`
- `*.pth.tar`
- `*.onnx`
- `*.faiss`
- `*.npy`

Examples in the local working directory include ArcFace, YOLOv7-face, ResNet18, and DeePixBiS checkpoints.

Exception:

- `weights/flash_liveness_v3_fixed_protocol/best_flash_liveness_model.pth.part-*`
  - split Git-safe parts of the V3 fixed-protocol best checkpoint
  - restore with `bash weights/flash_liveness_v3_fixed_protocol/restore_best_weight.sh`

### Generated Outputs

- `flash_liveness_runs/`
- `Face-Anti-Spoofing-using-DeePixBiS/fusion_api_outputs/`
- `Face-Anti-Spoofing-using-DeePixBiS/fusion_eval_outputs_*/`
- `Face-Anti-Spoofing-using-DeePixBiS/inference_outputs_*/`
- `yolov7_face/runs/`

## Data Reconstruction Notes

### Flash Liveness Datasets

The repository contains scripts that describe how sanitized datasets were prepared:

- `scripts/prepare_flash_liveness_video_dataset.py`
- `scripts/prepare_flash_liveness_video_dataset_v2.py`

Related design and preparation notes are documented in:

- `README.md`
- `docs/thunderguard/THUNDERGUARD_REPRO.md`

### Bank-Based API Testing

The API and evaluation scripts expect a labeled bank directory with filenames that imply the class:

- names containing `true`, `genuine`, `real`, `local`, `live` -> real
- names containing `toumo`, `head`, `model`, `fake`, `spoof` -> fake

Typical local layout used during development:

```text
Face-Anti-Spoofing-using-DeePixBiS/data/test_1_compare/
├── bank/
└── test/
```

That data is not published here. Recreate it locally before running the API or evaluation flows.

## Sensitive Files

- `auth.json` is excluded because it may contain local credentials or environment-specific configuration.

## Recommendation

Keep all large data and weights in external object storage, an internal NAS path, or a separate private release package, and keep this Git repository focused on:

- source code
- scripts
- documentation
- lightweight configuration
