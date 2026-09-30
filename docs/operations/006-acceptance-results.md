# 验收结果与归档：2026-09-30

机器：`connect.westd.seetacloud.com:46220`，`/root/lawn_segmentation`。验收进程PID1564，输出 `artifacts/acceptance_20260930`，日志run.log/evaluation.log/export.log；未重训。

操作：运行acceptance.py审计和全量复评，CUDA导出ONNX，执行verify_onnx.py与verify_inference.py；直接调用原有两个指标测试函数，编译scripts/src。验收与导出退出码均0，集成检查完成。GPU和权重哈希见005。

结果：有效3896对数据通过，1244张混淆矩阵与第34轮完全一致，mIoU=0.8949948627784652。ONNX在10张真实验证图上像素分类一致率均1.0，最大logits误差1.1920929e-5；SHA256为 `f7129d3760ed02b845eb063729da664c00732b6c7c1a01d2e2c111fe51597297`。单图1张/文件夹2张接口的全部像素坐标及掩码关系通过。生成30组样例、五张对照表、训练曲线；AI已逐张视觉审查，16灌木误分、28强光漏分记录在VISUAL_REVIEW.md，人类签字未完成。

归档：云端tar下载到本地结果目录，SHA256SUMS的203个文件逐个校验全部通过。最初归档调用因工作目录尚不存在被本地拒绝，未执行远端操作；改用现有项目根目录后成功。新增RKNN转换入口及部署约定，未安装工具链/未执行转换。原始249张空标签、典型红草覆盖不足与RK3588验证仍需后续完成。

下一步：审核变更、提交推送并核对远端提交；确认所有云端所需材料归档后按用户既有要求关机。完整命令和验收范围见docs/ACCEPTANCE.md。
