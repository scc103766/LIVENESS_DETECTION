# 项目 GPU 使用约定

更新日期：2026-05-08

本项目后续运行只使用物理 GPU：

```text
0,1,2
```

以下物理 GPU 不作为本项目运行卡：

```text
3,4,5,6
```

单卡推理服务默认优先使用物理 `1` 号卡，例如：

```bash
CUDA_VISIBLE_DEVICES=1 ...
```

多卡训练默认使用物理 `0,1,2`：

```bash
GPU_IDS=0,1,2 bash scripts/run_flash_liveness_v3_gpu012_training.sh
```

注意：`CUDA_VISIBLE_DEVICES=0,1,2` 会把物理 GPU `0/1/2` 映射为进程内 `cuda:0/cuda:1/cuda:2`。代码和命令中的 `--device cuda:0` 表示“当前进程可见的第 0 张卡”，不是一定等于物理 0 号卡。
