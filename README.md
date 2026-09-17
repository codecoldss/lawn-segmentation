# 割草机器人草地分割（三类 SegFormer 基线）

本工程实现割草机器人 **2D 草地语义分割**：`0=背景`、`1=草地`、`2=泥土草`。它不包含双目点云、SLAM、规控、电机控制或避障检测。模型输出的像素坐标、掩码和可通行区域供后续模块使用，不能直接视为实机导航成功。

原始数据由 8 组采集序列和 1 组补充序列组成，名义上有 4,145 对 `640×480` 图像/灰度掩码。审计发现其中 249 张灰度标签为零字节文件，因此严格模式会停止；当前可训练的临时集为 3,896 对。验证集固定为序列 `2/3/4`（1,244 张），临时训练集为其余有效序列（2,652 张），避免同一连续序列跨训练/验证集。

## 环境

Python 3.10+；训练机需安装 CUDA 兼容的 PyTorch。先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装与 CUDA 匹配的 `torch`，再执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
```

## 可执行 SOP

1. 只做全量源数据审计（不解压、不写入工程数据）：

```powershell
python scripts/prepare_data.py --audit-only
```

严格模式的通过条件是 `sample_count=4145`、`train_count=2901`、`val_count=1244`，图像和掩码均为 `640×480`，标签值仅为 `0/1/2`。当前灰度标签实际含 249 个零字节占位文件，严格审计会失败并列出问题；彩色标签只覆盖其余已存在的灰度标签，不能补回这 249 张。拿到修复标签前，可显式加 `--skip-empty-labels` 生成临时数据集，其通过条件为 `sample_count=3896`、`train_count=2652`、`val_count=1244`；原始 ZIP 不会被修改。

2. 从不可变 ZIP 生成训练数据和清单。首次执行：

```powershell
python scripts/prepare_data.py --skip-empty-labels
```

如需覆盖仅由本脚本生成的 `data/processed`，显式执行 `python scripts/prepare_data.py --overwrite`；原始 ZIP 永不修改。

3. 运行三类 SegFormer-B0 训练：

```powershell
python scripts/train.py --data data/processed --output artifacts/train
```

通过条件：生成 `artifacts/train/best.pt`、`config.json`、`metrics.json`；`metrics.json` 含总体像素准确率、每类 IoU、mIoU 与混淆矩阵。实际数值只以本次训练日志为准。

4. 评测并产生 30 张人工审查样本：

```powershell
python scripts/evaluate.py --checkpoint artifacts/train/best.pt --data data/processed --output artifacts/eval --visualize 30
```

人工检查 `artifacts/eval/visual_review` 中的原图、预测叠加、真值彩色图和可通行掩码，重点记录阴影、红草、泥土草和灌木边缘的漏分/误分。`traversable.png` 只将类别 `1=草地` 标为可通行；泥土草保守地不纳入。

5. 单图或文件夹 2D 推理：

```powershell
python scripts/infer.py --checkpoint artifacts/train/best.pt --input path\to\image-or-folder --output artifacts/infer
```

输出 `mask.png`（类别 ID）、`color.png`、`overlay.png`、`traversable.png` 及 `pixels.json`（每类像素坐标，`y,x` 顺序）。

6. 导出固定输入尺寸 ONNX：

```powershell
python scripts/export_onnx.py --checkpoint artifacts/train/best.pt --output artifacts/segformer_b0_3class_640x480.onnx
```

ONNX 输入为 `images: float32[1,3,480,640]`（RGB、ImageNet 标准化），输出为 `logits: float32[1,3,480,640]`。RKNN 转换、`rknn_init`、`rknn_run`、FPS、内存、量化精度和实机结果必须在指定 RK3588 板卡上单独测量；ONNX 文件不是板端成功证据。

## 测试

```powershell
$env:PYTHONPATH = "$PWD\src"
pytest -q
python -m py_compile scripts\*.py src\lawn_segmentation\*.py
```

## 第二阶段：过曝草

当前灰度掩码只有 `0/1/2`，因此不能伪造第四类。新增 `3=过曝草` 前，先冻结标注规范、对新增掩码抽样复核，然后将配置、调色板、评测与模型输出统一改为四类，并同三类基线进行同一验证集对比。
