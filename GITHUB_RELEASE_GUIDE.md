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
- `backbones/`
- `scripts/`
- `yolov7_face/` source code
- `Face-Anti-Spoofing-using-DeePixBiS/` source code
- `archive_20240320_flash_liveness/` curated code-only archive module
- repository documentation

## Excluded Main Areas

- `dataset/`
- `20240320闪光活体归档/`
- `environment/` snapshots from archived projects
- `炫彩闪烁活体/`
- `Face-Anti-Spoofing-using-DeePixBiS/data/`
- model weight files
- generated results and logs

## Archive Module Policy

The raw `20240320闪光活体归档/` directory is not committed directly because it mixes:

- source code
- local datasets
- conda environments
- exported ONNX files
- generated inference outputs
- compiled CMake artifacts

Instead, a curated mirror is published at `archive_20240320_flash_liveness/` that keeps:

- `ThunderGuard` source and notes
- `FaceAlign` source and notes
- lightweight XML and text resources that explain the code

while omitting large or environment-specific assets.

## Important Note About The DeePixBiS Subproject

The local workspace originally contained `Face-Anti-Spoofing-using-DeePixBiS/` as an embedded Git repository.
For a single-repository GitHub upload, it should be flattened into the parent repository so that its source files are committed directly instead of as a broken gitlink entry.

## Suggested Push Flow

1. Clean tracked large assets from the Git index.
2. Add `.gitignore` for datasets, weights, and outputs.
3. Add documentation files:
   - `README.md`
   - `DATA_ASSETS.md`
   - `FUSED_FACE_LIVENESS_API_README.md`
   - `GITHUB_RELEASE_GUIDE.md`
4. Point `origin` to the target repository.
5. Commit and push.

## Target Repository

`https://github.com/scc103766/LIVENESS_DETECTION.git`
