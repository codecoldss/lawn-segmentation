# 训练前检查
时间：2026-09-28T14:00:10.673292+00:00
机器：connect.westd.seetacloud.com:46220
目录：/root/lawn_segmentation
结果：RTX 4090；torch 2.3.0+cu121，CUDA 可用；缺 transformers。
现有配置：SegFormer-B0，80轮，batch=4，seed=20260918。
发现：RandomScale 后未恢复固定尺寸，可能导致 DataLoader 拼接失败。
下一步：独立环境补依赖，恢复480×640尺寸，审计有效数据，启动训练。
