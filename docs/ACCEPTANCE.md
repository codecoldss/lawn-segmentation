# 三类基线验收：2026-09-30

使用已完成80轮训练的第34轮最佳权重，不重新训练。云机为westd:46220，目录 `/root/lawn_segmentation`；实测RTX4090、驱动报告49140MiB显存，PyTorch2.3.0+cu121、CUDA可用。代码、权重与验收结果保存到GitHub的 `feature/segformer-three-class-baseline` 分支。

| 项目 | 实际结果 | 证据 |
|---|---|---|
| 原始4145对完整数据 | 未通过：249个零字节掩码 | 原始source_audit.json |
| 有效3896对 | 通过：2652训练/1244验证，尺寸标签合法、无重名、无跨集合完全相同图片、采集组分离 | data_audit.json |
| 最佳权重复评 | 通过：1244张，混淆矩阵逐项等于第34轮 | metrics.json |
| 30张视觉检查 | AI已逐张记录；发现误分，人类签字未完成 | VISUAL_REVIEW.md、五张review_sheet |
| 单图/文件夹推理 | 通过：1张+2张，掩码/二值图/全部像素坐标一致 | inference_report.json |
| ONNX结构及真实图片对比 | 通过：10张，掩码一致率均100%，最大logits绝对误差1.1920929e-5 | onnx_verification.json |
| 原有指标单元检查 | 2项通过，远端直接调用测试函数；并非pytest完整运行 | tests.log |
| RKNN与RK3588 | 仅提供转换入口，转换及板端均未实测 | DEPLOYMENT.md |

除原始审计外，表中报告均位于 `results/segformer_b0_3class_20260928/acceptance_20260930`。SHA256SUMS覆盖203个原始云端验收产出，下载后全部核对通过；后加的审查记录和环境补录另行保存。输出是2D分割，不能证明导航或障碍物安全性能。

## 可复制命令

在工程根目录运行，下例假定已有与本次相同的有效数据清单和本地缓存的SegFormer配置。当前数据含空标签，准备时须显式使用 `--skip-empty-labels`；绝不自行补造标签。

```bash
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python scripts/prepare_data.py --skip-empty-labels
# 重现训练；默认配置的最大轮数为80、随机种子见已归档config.json。
python scripts/train.py --data data/processed --output artifacts/reproduce_train
# 验收已保存的权重（以下不重训）：
python scripts/acceptance.py --checkpoint results/segformer_b0_3class_20260928/best.pt --history results/segformer_b0_3class_20260928/metrics.json --data data/processed --output artifacts/acceptance_reproduce --device cuda
python scripts/export_onnx.py --checkpoint results/segformer_b0_3class_20260928/best.pt --output artifacts/acceptance_reproduce/model.onnx --device cuda
python scripts/verify_onnx.py --checkpoint results/segformer_b0_3class_20260928/best.pt --onnx artifacts/acceptance_reproduce/model.onnx --samples-json artifacts/acceptance_reproduce/selected_samples.json --output artifacts/acceptance_reproduce/onnx_verification.json
python scripts/verify_inference.py --checkpoint results/segformer_b0_3class_20260928/best.pt --samples-json artifacts/acceptance_reproduce/selected_samples.json --output artifacts/interface_checks
# 接口直接使用：--input可为一张PNG或含PNG的文件夹，替换为实际路径。
python scripts/infer.py --checkpoint results/segformer_b0_3class_20260928/best.pt --input path/to/image.png --output artifacts/single --device cuda
python scripts/infer.py --checkpoint results/segformer_b0_3class_20260928/best.pt --input path/to/folder --output artifacts/folder --device cuda
```

验收输出应含数据审计、1244行每图指标、30张样例的原图/真值/预测/叠加/掩码/可通行图、五张对照表及训练曲线。ONNX比较预先设定logits容差atol=5e-4、rtol=1e-3，掩码一致率至少99.99%，不是看结果后调整门槛。接口测试核对每张640×480图全部307200个坐标。

## 问题与下一步

样例16把灌木预测成草地，会污染可通行候选；样例28存在强光下草地漏分。类别2标签还覆盖部分硬质道路，需老师统一标注语义。先补齐249个缺失标签并复核灌木边界、红草和过曝草，再开展公平的新实验；当前30张中发红候选主要是枯黄/露土，未能确认典型红草覆盖。四类规则见FOUR_CLASS_ANNOTATION.md。上述待补标和待板卡项不需要保持空闲云机开机。
