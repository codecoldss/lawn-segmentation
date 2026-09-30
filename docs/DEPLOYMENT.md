# ONNX 与 RK3588 输入输出约定

本次 ONNX 已通过结构检查及10张真实验证图片的数值比较。RKNN 转换入口仅保存供有匹配工具链时执行；本次没有安装 RKNN Toolkit2，也没有板卡，不能声称 RKNN 转换、初始化、运行或性能通过。

输入 `images`：固定 `float32[1,3,480,640]`，NCHW、RGB。将原图调整为640×480，除以255，再逐通道执行 `(value - mean) / std`；mean为 `[0.485,0.456,0.406]`，std为 `[0.229,0.224,0.225]`。RKNN入口使用恒等预处理，避免重复标准化。板端要确认运行时实际输入布局，显式完成布局转换。

输出 `logits`：固定 `float32[1,3,480,640]`，三个通道依次对应背景、草地、泥土草；沿通道取argmax得到 `uint8[480,640]` 类别图。无需softmax。可通行候选仅 `mask == 1`，输出0/255二值图；这不是机器人安全通行保证。像素坐标采用左上角原点、`[y,x]`、0开始计数。

```bash
python scripts/export_onnx.py --checkpoint results/segformer_b0_3class_20260928/best.pt --output artifacts/segformer.onnx --device cuda
# 在单独的匹配RKNN Toolkit2环境执行，本次未实测：
python scripts/export_rknn.py --onnx artifacts/segformer.onnx --output artifacts/segformer_rk3588.rknn
```

转换入口依据 [Rockchip官方ONNX转换示例](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8/python/convert.py) 的config/load_onnx/build/export_rknn顺序实现，默认关闭量化。SegFormer算子是否兼容、FP精度变化及输出布局都需要实际转换确认；若失败，保留工具日志并按具体算子处理。

板端验收须记录板卡、SDK/驱动、模型哈希、输入布局、`rknn_init`/`rknn_run`返回值、同一验证样本精度、预热后延迟/FPS、内存和功耗。INT8另需固定校准集及独立精度比较。PDF中的历史指标不能作为本项目实测值。
