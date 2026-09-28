# 最终训练结果核对与归档

目标机器与目录：`connect.westd.seetacloud.com:46220`，`/root/lawn_segmentation/artifacts/train_20260928T143323Z`。

核对：训练进程 PID 4472 已退出；`metrics.json` 有 80 轮结果。最佳为第 34 轮，mIoU 0.8949948628、像素准确率 0.9881775753；第 80 轮 mIoU 0.8768962594。关机监控日志记录 PID 4472 在 `2026-09-28T15:25:36Z` 结束后调用了关机命令。

从云机复制最终 `best.pt`、`metrics.json`、`train.log` 和 `config.json` 至 `results/segformer_b0_3class_20260928/`。SHA256：best.pt `11d7046746a14c052d990e04386c7e0694589fd157874ee55b608f16ef578107`；metrics.json `3860015cb50546e579440113abee07566f48de49a754125de3727e76ce360b4e`；train.log `563b9df874e6a808245316c43bd24713a8bcdf1c3f8befa0031e400b47fca0be`。

发现：自动关机只等待训练 PID 退出，早于最终归档和验收，导致云机曾不可连接；该条件不满足“收尾完成再关”。本地已补齐训练完成状态、最终指标和 Skill 关机/公平比较规则。下一步：校验改动并推送到 GitHub；所有交付处理完前保持云机开启。
