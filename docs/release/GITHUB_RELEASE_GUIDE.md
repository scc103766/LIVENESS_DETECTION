# GitHub Release Guide

## Goal

This repository is organized for a public or shared GitHub upload that preserves:

- code
- scripts
- API logic
- project documentation

and excludes:

- raw data
- videos and images
- model weights
- training outputs
- local environments

## Included Main Areas

- root Python scripts for face comparison, flash liveness, and bank-based API inference
- Flash Liveness V3/V3.1 code:
  - `flash_liveness_project_v3.py`
  - `flash_liveness_project_v3_1.py`
  - `flash_physical_features.py`
  - `flash_liveness_v3_api_service/`
- `backbones/`
- `scripts/`
- `docs/`
- `commands/`
- `assets/`
- `yolov7_face/` source code
- `Face-Anti-Spoofing-using-DeePixBiS/` source code
- repository documentation

## Excluded Main Areas

- `dataset/`
- `20240320闪光活体归档/`
- `炫彩闪烁活体/`
- `Face-Anti-Spoofing-using-DeePixBiS/data/`
- model weight files
- generated results and logs
- `flash_liveness_v3_api_service/outputs/`
- local upload files from API services

## Important Note About The DeePixBiS Subproject

The local workspace originally contained `Face-Anti-Spoofing-using-DeePixBiS/` as an embedded Git repository.
For a single-repository GitHub upload, it should be flattened into the parent repository so that its source files are committed directly instead of as a broken gitlink entry.

## Suggested Push Flow

1. Clean tracked large assets from the Git index.
2. Add `.gitignore` for datasets, weights, and outputs.
3. Add documentation files:
   - `README.md`
   - `PROJECT_STRUCTURE.md`
   - `docs/data/DATA_ASSETS.md`
   - `docs/api/FUSED_FACE_LIVENESS_API_README.md`
   - `docs/flash_liveness/FLASH_LIVENESS_PROJECT_V3_README.md`
   - `docs/flash_liveness/FLASH_LIVENESS_PROJECT_V3_TECHNICAL_ANALYSIS.md`
   - `docs/release/GITHUB_RELEASE_GUIDE.md`
4. Point `origin` to the target repository.
5. Commit and push.

## Target Repository

`https://github.com/scc103766/LIVENESS_DETECTION.git`
