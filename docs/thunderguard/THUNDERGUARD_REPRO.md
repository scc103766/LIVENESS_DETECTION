# ThunderGuard Reproduction Notes

This note captures the shortest local reproduction path for
`/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档`.

## Current Status

- Code root:
  `/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard`
- Ready-made exported dataset:
  `/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/tg_export`
- Training entry:
  `/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg/train.py`
- Default documented network:
  `MoEA`

## Fastest Reproduction Path

The archive already contains exported training samples, so we do not need to
rerun:

- video frame choosing
- face alignment
- normal/depth map generation

We only need:

1. Make `ThunderGuard/data/sample/train` and `test` point to `dataset/tg_export`.
2. Restore or prepare a Python environment with:
   `torch`, `torchvision`, `opencv-python`, `onnxruntime`, `tqdm`, `numpy`.
3. Run training from `ThunderGuard/pytg`.

## Dataset Layout Used by Training

`pytg/train.py` reads:

- `../data/sample/train`
- `../data/sample/test`

Each sample is a triplet with the same basename:

- `xxx.txt`
- `xxx.jpg`
- `xxx_d.jpg`

The final numeric suffix in the basename is the attack type label:

- `1`: real
- `2/3`: print
- `4/5`: screen replay
- `6`: mask / 3D model

## Training Command

From:

`/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/pytg`

Single GPU / CPU:

```bash
../../scripts/run_thunderguard_train.sh --network MoEA
```

Smoke test:

```bash
bash /supercloud/llm-code/scc/scc/Liveness_Detection/scripts/run_thunderguard_train.sh \
  --smoke \
  --network MoEA
```

Multi-GPU:

```bash
LD_LIBRARY_PATH='../environment/exp_yanyu_cv/lib' \
../environment/exp_yanyu_cv/bin/python \
-m torch.distributed.launch --nproc_per_node 2 train.py --network MoEA
```

## Important Notes

- `nxreadme训练模型步骤.txt` mentions `MTGAN`, but the codebase contains
  `MoEA`, `MoE`, `TG`, `TGA`, `AG`, `SGTD`, `DTGAN`, `MultiEA`.
  The main README uses `MoEA`, so that is the safer default.
- The archived environment bundles are under:
  `20240320闪光活体归档/environment/`
- At the time of inspection, the current default Python did not have the
  required deep learning packages installed.
- A restored local training environment now exists at:
  `/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/environment/exp_yanyu_cv`
- For quick validation, a smoke subset can be generated under:
  `/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/tg_export_smoke`
