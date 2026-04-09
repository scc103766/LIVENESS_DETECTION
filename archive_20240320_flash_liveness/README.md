# 20240320 Flash Liveness Archive Module

This directory is a curated, code-only mirror of the local archive:

`/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档`

It is included in the GitHub release as an archive module so the engineering logic is retained without publishing large datasets, conda environments, model weights, ONNX exports, or generated outputs.

## Included Content

### ThunderGuard

Kept source and documentation for:

- `libtg/`
  - C++ preprocessing and liveness support library
- `pytg/`
  - training, testing, dataset generation, and ONNX export scripts
- `tg_infer/`
  - PyTorch and ONNX inference scripts
- `tg_process/`
  - preprocessing scripts for normal-cue generation
- `README.md`
- `nxreadme训练模型步骤.txt`

### FaceAlign

Kept source and documentation for:

- `face_detect/`
  - face and landmark detection scripts
- `falib/`
  - C++ face-alignment library source, headers, and CMake files
- `falib_demo/`
  - demo source files
- `pyfa/`
  - Python alignment pipeline and wrapper code
- `README.md`

## Intentionally Omitted

The original archive also contains many non-code assets that are not suitable for a public code repository.

Omitted groups include:

- `dataset/`
- `environment/`
- `resources_smoke/`
- `ThunderGuard/resources/`
- `ThunderGuard/data/`
- old inference output folders inside `tg_infer/`
- CMake build directories and compiled libraries
- model files such as `*.pth`, `*.pth.tar`, `*.dat`, `*.onnx`
- sample images and videos used only as local demos
- Python cache directories

## Lightweight Resources Retained

Some lightweight text and XML resources are kept because they help explain the code structure, such as:

- `FaceAlign/pyfa/resources/cv-data/*.xml`
- training notes and readme files
- CMake build scripts and headers

## Recommended Reading Order

1. `ThunderGuard/README.md`
2. `FaceAlign/README.md`
3. `读我.txt`
4. `ThunderGuard/pytg/train.py`
5. `ThunderGuard/tg_infer/auto_test_onnx_models.py`

## Reconstruction Note

If you need the full original archive for local reproduction, use the private local path above and place the missing datasets, resources, and environments back next to this code. This Git module is intentionally documentation-first and source-first.
