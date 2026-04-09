# Data And Asset Notes

## Scope

This repository is prepared for GitHub publication in a code-first form.
Raw datasets, videos, environment exports, trained weights, large evaluation outputs, and local secrets are intentionally excluded from version control.

## Excluded Asset Groups

### Datasets

- `dataset/`
  - Local experiment outputs and prepared training splits.
- `Face-Anti-Spoofing-using-DeePixBiS/data/`
  - DeePixBiS sample images, bank images, and local test sets.
- `20240320闪光活体归档/`
  - Original raw archive with datasets, environments, resources, and outputs.
  - A curated code-only mirror is published instead at `archive_20240320_flash_liveness/`.
- `炫彩闪烁活体/`
  - Historical flash-liveness data and archives.

### Model Weights

The following kinds of files are excluded:

- `*.pt`
- `*.pth`
- `*.pth.tar`
- `*.onnx`
- `*.faiss`
- `*.npy`

Examples in the local working directory include ArcFace, YOLOv7-face, ResNet18, and DeePixBiS checkpoints.

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
- `THUNDERGUARD_REPRO.md`
- `archive_20240320_flash_liveness/README.md`

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
- curated archive mirrors without heavy assets
