# SegFormer-B0 三类草地分割结果

状态：**训练中途快照**，截至 2026-09-28T14:40:00.810105+00:00 完成 9/80 轮。运行机器：`connect.westd.seetacloud.com:46220`。

当前最佳：第 7 轮，mIoU=0.892511，像素准确率=0.986923。背景/草地/泥土草 IoU 分别为 0.975228 / 0.980743 / 0.721561。完整每轮指标和混淆矩阵见 [metrics.json](metrics.json)。

数据：原始 4,145 张图像中，249 张灰度标签是零字节；本次有效训练集 2,652 张，验证集 1,244 张，验证序列固定为 2/3/4。详情见 [source_audit.json](source_audit.json)。

文件：
- [best.pt](best.pt)：当前最佳权重，SHA256 `795347c354c32427a6dffb13adc6e0480586c00aabe04cde0e4359998891c97e`。
- [config.json](config.json)、[environment.txt](environment.txt)、[input_sha256.txt](input_sha256.txt)：配置、软件版本和输入摘要。
- [train.log](train.log)、[snapshot.json](snapshot.json)：截至快照时间的训练日志及状态。

80 轮最终结果、人工可视化复核、ONNX 与板端验证尚未完成；本目录目前不代表这些环节通过。
