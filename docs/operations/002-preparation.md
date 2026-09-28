数据审计通过有效集3896对，训练2652，验证1244；排除249空标签。证据：prepare.log。
RandomScale后添加Resize(480,640)，真实batch验证待Luna执行。
首次pip阿里HTTP源失败，已改官方HTTPS源，当前安装进行中。证据：install.log、install-retry.log。
用户要求先交接方案，正式训练尚未启动。下一步按../TRAINING_PLAN.md执行。

学术加速检查：/etc/network_turbo存在。原SSH环境未设置proxy；本次仅在检查命令内source。PyPI HTTP 200/1.085s，Hugging Face HTTP 200/2.989s。pip安装保持直连，下载预训练权重时按需启用。

代理实测：PyPI文件256KiB直连11.91s，学术代理4.62s；因此将transformers/albumentations下载切换为source /etc/network_turbo。原慢下载已停止。
