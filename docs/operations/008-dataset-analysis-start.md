# 数据集MD5及分布分析：2026-10-03

本地工程：C:/my_project/割草机器人/草地分割；不连接或重启付费云机。按training-operation-log Skill记录关键操作。

新增scripts/analyze_dataset.py直接只读扫描9个原图ZIP及灰度标签ZIP：文件MD5、解码像素MD5、有效图像标签对重复、跨集合泄漏、同图不同标签、类像素占比与逐组统计。零字节标签另列，不把空文件重复当作重复图片。额外用相邻文件名图片的64位dHash提供相似候选，仅作为待人工核对项，不自动删除。

本机Linux Python缺Pillow/numpy，创建临时venv又因ensurepip缺失失败；改用Codex自带Windows Python，Pillow/numpy已可用，未修改系统环境。输出 results/dataset_analysis_20261003，原始数据与训练划分不变。

下一步：运行全量扫描、核对旧数据审计计数，查看相似候选及光照类别样例，生成分析报告并推送GitHub。
