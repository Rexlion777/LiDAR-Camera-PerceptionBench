# Third-party components

本仓库不再分发下列第三方资产：

- **OpenPCDet / PointPillars**：运行时外部依赖，遵循其上游许可证。
- **KITTI 3D Object Detection Dataset**：需由使用者从 KITTI 官方渠道自行申请和下载，并遵守数据集条款。
- **模型权重、ONNX 文件与 TensorRT engine**：不包含在仓库中；需由使用者依据相应上游许可自行准备或生成。
- **CUDA、cuDNN、TensorRT**：NVIDIA 软件，遵循 NVIDIA 的适用条款。

仓库中的评测脚本、适配层、退化实验、运行质量诊断和可视化代码为本项目的原创工程内容。任何第三方名称仅用于说明兼容性，不表示背书或隶属关系。
