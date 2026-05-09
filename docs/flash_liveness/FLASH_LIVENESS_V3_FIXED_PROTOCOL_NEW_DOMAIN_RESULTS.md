# Flash Liveness V3 Fixed-Protocol 新域协议测试结果

## 评测对象

- 模型：
  `/supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/flash_liveness_v3_fixed_protocol_gpu3/best_flash_liveness_model.pth`
- 数据集：
  `/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_new_domain_video_protocol_v1v2`
- 推理脚本：
  `/supercloud/llm-code/scc/scc/Liveness_Detection/scripts/infer_flash_liveness_v3_video.py`
- 运行设备：
  历史评测记录保留旧运行环境说明；按 2026-05-08 更新后的项目约定，后续运行只使用物理 `0/1/2`，不再使用 `3/4/5/6`

## 推理配置

- `window_size=256`
- `window_stride=128`
- `window_fusion=quality_trimmed_mean`
- `window_trim_ratio=0.2`
- `window_min_quality=0.05`
- `threshold=0.9375`

## 输出文件

- 逐视频预测：
  `/supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/flash_liveness_v3_fixed_protocol_new_domain_eval_gpu3/predictions.csv`
- 机器可读汇总：
  `/supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/flash_liveness_v3_fixed_protocol_new_domain_eval_gpu3/summary.json`
- 评测摘要：
  `/supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/flash_liveness_v3_fixed_protocol_new_domain_eval_gpu3/summary.md`

## 总体结果

- 总样本数：`280`
- 成功样本数：`280`
- 失败样本数：`0`
- 运行时长：`727.824s`

整体指标：

- `accuracy = 0.950000`
- `auc = 0.955895`
- `eer = 0.064037`
- `apcer = 0.007634`
- `bpcer = 0.666667`
- `acer = 0.337150`
- `tp / tn / fp / fn = 6 / 260 / 2 / 12`

标签分布：

- `live = 18`
- `spoof = 262`

## 按 split 指标

| split | total | live | spoof | accuracy | auc | eer | apcer | bpcer | acer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 176 | 12 | 164 | 0.931818 | 0.959350 | 0.084350 | 0.012195 | 0.833333 | 0.422764 |
| val | 54 | 4 | 50 | 0.981481 | 0.905000 | 0.245000 | 0.000000 | 0.250000 | 0.125000 |
| test | 50 | 2 | 48 | 0.980000 | 0.989583 | 0.010417 | 0.000000 | 0.500000 | 0.250000 |

## 关键结论

1. 模型对 spoof 样本识别很强。
   在 `262` 个 spoof 视频里只错了 `2` 个，spoof 侧准确率约为 `99.24%`。

2. 模型对新域 live 样本召回明显不足。
   `18` 个 live 视频里只判对了 `6` 个，`bpcer=0.666667`，说明当前保存阈值 `0.9375` 对这批新域 live 来说过高。

3. 整体 `accuracy=0.95` 不能单独代表泛化效果很好。
   这是一个高度不均衡数据集，spoof 占绝大多数，因此更应同时关注 `bpcer`、`acer` 和 live 召回。

4. 排序能力其实不差。
   `auc=0.955895`、`eer=0.064037` 说明模型分数排序仍然有区分度，但当前固定阈值没有对齐新域数据分布。

5. 新域更像是“阈值迁移问题 + live 域偏移问题”，而不只是模型完全失效。
   其中一个 live 错分样本分数正好落在 `eer_threshold=0.45011532` 附近，说明若重新做新域阈值校准，指标预计会明显改善。

## 错分样本

总错分：`14`

其中：

- live 被错判为 spoof：`12`
- spoof 被错判为 live：`2`

### live -> spoof

- `train/live_extra_angles__41d464355e__Core_-_selfie_mode.mp4` -> `0.80225837`
- `train/live_extra_angles__4aef28d597__1._Back_close.mp4` -> `0.72736168`
- `train/live_outdoor__21cd9b4128__18.mp4` -> `0.45011532`
- `train/live_outdoor__6bddd8095a__52.mp4` -> `0.73156857`
- `train/live_outdoor__be322ec4ad__65.mp4` -> `0.61759883`
- `train/live_public_real__18a17d5e2e__4.mp4` -> `0.77487385`
- `train/live_public_real__1ac6522a41__30.mp4` -> `0.60340297`
- `train/live_public_real__8ff0672325__30.mp4` -> `0.53203768`
- `train/live_public_real__a4ab30e491__16.mp4` -> `0.75758815`
- `train/live_public_real__fa68c2d3dd__22.mp4` -> `0.88498712`
- `val/live_outdoor__a4dca0ed07__33.mp4` -> `0.00130778`
- `test/live_public_real__3b30a6866a__6.mp4` -> `0.55153930`

### spoof -> live

- `train/spoof_wrapped_3d_paper__01fcd72a48__101.mp4` -> `0.93963927`
- `train/spoof_wrapped_3d_paper__67a9c120b1__135.mp4` -> `0.98199892`

## 结果解读

从这次结果看，当前固定协议 V3 模型更像是一个“高安全阈值、强 spoof 拦截”的模型：

- 优点是对大部分攻击视频压得很稳，误放行很少。
- 问题是对新域真人视频偏保守，导致较高的拒真率。

如果业务目标是安全优先，这个结果说明模型具备不错的攻击拦截能力。
如果业务目标要求真人通过率，这个阈值不能直接上线，建议至少补做：

1. 新域验证集阈值重标定。
2. 对 `live_outdoor`、`live_public_real`、`live_extra_angles` 这几类真人视频做专门增广或小规模微调。
3. 对 `spoof_wrapped_3d_paper` 的两个误放行样本单独复查，确认是否属于包覆式 3D 纸面攻击的薄弱点。

## 建议的下一步

1. 用当前 `predictions.csv` 扫一遍阈值，找新域上的最优 operating point。
2. 输出一版“安全优先阈值”和一版“通过率优先阈值”。
3. 如果后续要继续提升泛化，优先补真人新域样本，而不是继续单纯堆 spoof。
