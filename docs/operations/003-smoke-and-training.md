# 前向验证与训练启动
机器：connect.westd.seetacloud.com:46220
目录：/root/lawn_segmentation
依赖：torch 2.3.0+cu121、transformers 4.44.2、albumentations 1.4.3；CUDA 可用。
验证：两批真实样本完成前向/反向，形状为[B,3,480,640]/[B,480,640]，loss与显存见 smoke.log。
配置：SegFormer-B0，3类，80轮，batch=4，seed=20260918。
启动：PID=4472；输出=artifacts/train_20260928T143323Z/train.log；权重与指标输出=artifacts/train_20260928T143323Z。
数据：训练2652，验证1244；249空标签已排除。
训练日志每1步及每50步打印loss。

首条运行确认：PID 4472仍运行；epoch 1 step 300/663，train_loss=0.08979；GPU利用率98%，已用显存2875 MiB。
