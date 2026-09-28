# SegFormer-B0 训练交接方案（供 Luna 执行）

## 1. 当前状态
- 唯一目标：`ssh -p 46220 root@connect.westd.seetacloud.com`。
- 项目：`/root/lawn_segmentation`，已确认文件存在；无需重复传数据。
- GPU：系统报告 RTX 4090、49140 MiB；CUDA 可用。GPU型号与显存按驱动报告记录，不推断硬件改装情况。
- 基础 Python：`/root/miniconda3/bin/python`，torch 2.3.0+cu121。
- 独立环境：`.venv`（system-site-packages），补充依赖正在安装；以实际进程和导入检查为准。
- 已执行数据审计/解压：3896对有效样本，训练2652、验证1244（组2/3/4）。
- 原4145个灰度标签中249个零字节，已显式排除，不称为完整4145对验收通过。
- 已修复 data.py：RandomScale 后以最近邻掩码插值的 A.Resize 恢复480×640，避免batch尺寸不一致。尚待真实batch验证。
- 已初始化本机 Git 分支 feature/segformer-baseline；没有提交，没有配置远端，不影响运行现有代码。
- **训练已于2026-09-28 14:33 UTC启动；PID=4472，运行目录 artifacts/train_20260928T143323Z。详见 docs/operations/003-smoke-and-training.md。**

## 2. 操作记录 skill
先读 `.agents/skills/training-operation-log/SKILL.md`。
每个关键步骤写 `docs/operations/NNN-主题.md`，仅包含：时间、机器、关键命令、结果/证据、下一步。
失败及修复也要记；每个shell查询不用单独写长文。
所有状态以实时检查为准。用户偏好总结由 Astra 写；执行者不能伪称作者模型。

## 3. 环境与数据门槛
登录后：
```bash
cd /root/lawn_segmentation
export PATH="/root/lawn_segmentation/.venv/bin:$PATH"
export PYTHONPATH="/root/lawn_segmentation/src"
export NO_ALBUMENTATIONS_UPDATE=1
nvidia-smi
pgrep -af '[p]ip install'
tail -20 docs/operations/install-retry.log
```
若原安装仍运行，先等待或排查，不并行启动另一个pip。原HTTP阿里源返回无可用版本，官方HTTPS PyPI已开始下载。若进程失败：
```bash
python -m pip install --index-url https://pypi.org/simple numpy==1.26.4 transformers==4.44.2 albumentations==1.4.3 opencv-python-headless==4.10.0.84 'Pillow>=10' pytest
python -c 'import torch, transformers, albumentations, cv2, numpy; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, albumentations.__version__, numpy.__version__)'
python -m pytest -q
wc -l data/processed/train.jsonl data/processed/val.jsonl
cat data/processed/audit_report.json
```
确认2652/1244、无序列交叉；随机批次图像必须[B,3,480,640]，掩码[B,480,640]且只有0/1/2。
无需重复解压。审计原始输出为 docs/operations/prepare.log。

## 4. 预训练权重与短验证
模型 nvidia/mit-b0；原站huggingface.co连接测试超时，hf-mirror.com配置地址可达，但大权重下载尚未验证。
先试官方站；若失败，可用镜像并记录模型来源、解析后的revision和本地文件SHA256：
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```
保持该环境变量用于后续所有加载命令，禁止遇到下载失败后改成随机初始化。

执行2个真实训练batch的前向/反向短验证，验证上述形状、有限loss和GPU显存：
```bash
python - <<'PY'
import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from lawn_segmentation.data import build_dataset, read_manifest
loader = DataLoader(build_dataset(read_manifest("data/processed/train.jsonl"), True),
                    batch_size=4, shuffle=True, num_workers=0)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0", num_labels=3, ignore_mismatched_sizes=True).cuda()
model.train()
for i, batch in enumerate(loader):
    assert batch["pixel_values"].shape == (4,3,480,640)
    assert batch["labels"].shape == (4,480,640)
    assert set(batch["labels"].unique().tolist()) <= {0,1,2}
    loss = model(pixel_values=batch["pixel_values"].cuda(),
                 labels=batch["labels"].cuda()).loss
    assert torch.isfinite(loss).item()
    loss.backward()
    model.zero_grad(set_to_none=True)
    print(i, loss.item(), torch.cuda.max_memory_allocated(), flush=True)
    if i == 1: break
PY
```
短验证不会保存权重。若OOM，将正式配置复制后仅降低batch到2并记录；禁止CPU回退。
classifier未在预训练权重中初始化的提示属于三类任务新头；其他大量encoder缺失需要排查。

## 5. 正式训练
固定配置：640×480、三类、80轮、batch4、AdamW、lr6e-5、weight_decay0.01、seed20260918、workers4。
先检查没有现有train.py进程。输出目录使用新的运行名，避免覆盖。
```bash
RUN="artifacts/train_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN"
python -m pip freeze > "$RUN/environment.txt"
cp configs/segformer_b0_3class.json "$RUN/requested_config.json"
sha256sum scripts/train.py src/lawn_segmentation/data.py data/processed/train.jsonl data/processed/val.jsonl > "$RUN/input_sha256.txt"
nohup python -u scripts/train.py --data data/processed --config configs/segformer_b0_3class.json --output "$RUN" > "$RUN/train.log" 2>&1 < /dev/null &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$RUN/train.pid"
printf '%s\n' "$RUN" > docs/operations/active-run.txt
```
将RUN、PID和命令写入步骤记录。验证进程仍活着、GPU出现Python占用、日志无Traceback。
现有代码每轮结束才输出指标，首轮无日志不等于卡死；先核实GPU利用率与进程。
首轮必须产生metrics.json和best.pt，再报告“训练运行正常”。
完整完成标准：进程正常结束，metrics.json含80轮且指标有限，最佳权重可重新加载。
时间预估用首轮真实用时×剩余轮数，不能凭空给ETA。

当前脚本没有resume/last checkpoint；中断后不能声称从断点恢复。保留旧目录，在新目录重启，或先明确实现并验证resume再使用。
若需提高运行可见性，可增加每50batch的loss/epoch/step输出，保留80轮和其他超参数。
Git记录仅纳入代码/配置/skill/MD；禁止git add .收进现有原始图片目录或凭据。

## 6. 评测、可视化、推理、导出
```bash
RUN="$(cat docs/operations/active-run.txt)"
python scripts/evaluate.py --checkpoint "$RUN/best.pt" --data data/processed --output "$RUN/eval" --visualize 30
```
检查像素准确率、每类IoU、mIoU和混淆矩阵。不要预设一个虚构的达标精度。
**现有evaluate只取前30张**：改为在2/3/4组中固定seed抽样各10张；另按图像逐张统计草/泥土草互相误分像素数，保存最高的案例。按真实图片补足阴影、发红草、灌木边缘；未人工查看不能写“人工复核通过”。
当前train尚未逐轮生成可视化。如要满足原始完整验收，在正式训练前补充固定少量验证图每轮原图/真值/预测/叠加输出，复用现有infer辅助函数；避免每轮输出全量像素坐标文件。
单图和文件夹测试选真实有效验证图路径：
```bash
python scripts/infer.py --checkpoint "$RUN/best.pt" --input <真实单图路径> --output "$RUN/infer_one"
python scripts/infer.py --checkpoint "$RUN/best.pt" --input <小型验证图文件夹> --output "$RUN/infer_folder"
python -m pip install --index-url https://pypi.org/simple onnx onnxruntime
python scripts/export_onnx.py --checkpoint "$RUN/best.pt" --output "$RUN/segformer_b0_3class_640x480.onnx"
```
ONNX需运行checker，再用同一标准化输入对比PyTorch/ONNX Runtime输出形状[1,3,480,640]和数值误差，记录误差与argmax一致率；只导出文件不能算验证完成。
二值可通行图仅类别1；泥土草类别2不纳入。RKNN/板端验证留待板卡实测。

## 7. 最终交接
报告仅写真实轮数、最佳轮次/指标、权重/日志/报告路径和未完成项。
保留固定验证集、环境版本、配置、数据审计、源码摘要及操作记录。
下一位执行者先检查实时安装状态，再短验证，然后启动训练；无需再次询问启动授权。
