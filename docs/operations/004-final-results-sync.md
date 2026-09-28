# 最终训练结果核对与归档

目标机器与目录：`connect.westd.seetacloud.com:46220`，`/root/lawn_segmentation/artifacts/train_20260928T143323Z`。

核对：训练进程 PID 4472 已退出；`metrics.json` 有 80 轮结果。最佳为第 34 轮，mIoU 0.8949948628、像素准确率 0.9881775753；第 80 轮 mIoU 0.8768962594。关机监控日志记录 PID 4472 在 `2026-09-28T15:25:36Z` 结束后调用了关机命令。监控只依赖训练 PID 退出，早于最终归档和验收，关机时机设置错误。

从云机复制最终 `best.pt`、`metrics.json`、`train.log` 和 `config.json` 至 `results/segformer_b0_3class_20260928/`。SHA256：best.pt `11d7046746a14c052d990e04386c7e0694589fd157874ee55b608f16ef578107`；metrics.json `3860015cb50546e579440113abee07566f48de49a754125de3727e76ce360b4e`；train.log `563b9df874e6a808245316c43bd24713a8bcdf1c3f8befa0031e400b47fca0be`。

已将最终权重、80轮指标和日志推送至 GitHub 分支 `feature/segformer-three-class-baseline`，提交 `ef97e47`。同时更新训练公平比较规则和云机关机条件。

后续验收：云机重连后 `.venv/bin/python` 检测到 `torch.cuda.is_available() == False`。完整评测/可视化命令因模型按 CUDA 加载而失败；CPU ONNX 导出运行6分钟无产物后已停止。没有再次执行云机关机。待 CUDA 恢复后再生成30张可视化样例、完成抽检和 ONNX 导出；RK3588/RKNN 板端验证需有板卡后进行。
