# Flash Liveness V3 Fixed-Protocol Best Weight

本目录保存 V3 fixed-protocol best checkpoint 的 Git 分片版本。

原始权重文件：

```text
best_flash_liveness_model.pth
```

原始大小：

```text
214879434 bytes
```

原始 SHA256：

```text
03145cccb43c958bb947a4f9c1d26a2e588fcc870c4974183614f590940e987b
```

由于原始 `.pth` 超过 GitHub 普通 git 单文件 100MB 限制，本 release 使用 45MiB 左右的分片保存：

```text
best_flash_liveness_model.pth.part-00
best_flash_liveness_model.pth.part-01
best_flash_liveness_model.pth.part-02
best_flash_liveness_model.pth.part-03
best_flash_liveness_model.pth.part-04
```

恢复权重：

```bash
bash weights/flash_liveness_v3_fixed_protocol/restore_best_weight.sh
```

恢复后会生成：

```text
weights/flash_liveness_v3_fixed_protocol/best_flash_liveness_model.pth
```

脚本会自动校验文件大小和 SHA256。恢复出的 `.pth` 被 `.gitignore` 忽略，不会再次作为普通大文件提交。

V3 API 服务默认 checkpoint 路径已经指向恢复后的文件。
