# 代码地图

## 推荐阅读顺序

1. [`kitti_io.py`](../runtime/lidar_system_algorithm/kitti_io.py)：KITTI 点云、图像、标注与标定文件入口。
2. [`calibration.py`](../runtime/lidar_system_algorithm/calibration.py) + [`transforms.py`](../runtime/lidar_system_algorithm/transforms.py)：Camera-LiDAR 坐标链、3D box 和投影实现。
3. [`openpcdet_adapter.py`](../runtime/lidar_system_algorithm/openpcdet_adapter.py)：OpenPCDet 模型接入、预测转换与导出。
4. [`deployment_acceptance.py`](../runtime/lidar_system_algorithm/deployment_acceptance.py)：退化设置、批处理和指标汇总。
5. [`pointpillars_wrapper_runtime.py`](../runtime/lidar_system_algorithm/pointpillars_wrapper_runtime.py)：PointPillars 子模块包装与推理数据合同。
6. [`profiling.py`](../runtime/lidar_system_algorithm/profiling.py) + [`online_latency.py`](../runtime/lidar_system_algorithm/online_latency.py)：分阶段 latency、warm-up、均值与尾延迟。
7. [`failure_matcher.py`](../runtime/lidar_system_algorithm/failure_matcher.py)：类别、距离与匹配维度的结果分析。
8. [`tracking.py`](../runtime/lidar_system_algorithm/tracking.py)：基础关联与向量化优化实现。

## 目录职责

| 路径 | 内容 | 规模 |
|---|---|---:|
| `runtime/lidar_system_algorithm/` | 可复用的几何、评测、tracking、部署与诊断模块 | 19 modules |
| `scripts/lidar_system_algorithm/` | 训练、评测、压力测试、TensorRT 与报告生成入口 | 49 scripts |
| `scripts/portfolio/` | 公开图表与系统总览生成器 | 1 generator |
| `tests/` | 单元测试、合同测试与实验结果回归测试 | 63 files |
| `configs/` | PointPillars 训练与微调配置 | YAML |
| `evidence/raw/` | AP、FP/FN、距离分段、标定、逐帧延迟与运行质量记录 | 18 CSVs / 16k+ rows |
| `assets/portfolio/` | 从结果表重新生成的公开图表 | 16 figures |

## 主要实验入口

### 感知与官方评测

- `run_kitti_pointcloud_pipeline.py`
- `run_pointpillars_inference_profile.py`
- `run_kitti_official_eval.py`
- `run_kitti_failure_matcher.py`

### 标定、同步与退化

- `run_calibration_sync_robustness.py`
- `run_lidar_deployment_acceptance_benchmark.py`
- `run_lidar_deployment_acceptance_full_batch.py`
- `generate_deployment_acceptance_dense_diagnostics.py`

### TensorRT 与性能定位

- `audit_tensorrt_binding_contract.py`
- `debug_wrapper_pytorch_core_parity.py`
- `debug_tensorrt_submodule_bisection.py`
- `debug_tensorrt_backbone_head_only_diff.py`
- `run_tensorrt_backbone_head_only_eval.py`
- `profile_tensorrt_backbone_head_online_pipeline.py`

### 训练与微调

- `build_kitti_stratified_splits.py`
- `run_pointpillars_subset_training.py`
- `run_pointpillars_expanded_finetune.py`
- `diagnose_finetune_drift.py`

## 测试体系

测试按四类组织：

- **Geometry contracts**：标定矩阵、坐标变换、投影和 BEV box；
- **Runtime contracts**：wrapper batch dict、TensorRT binding、dynamic shape 和 decode；
- **Algorithm regression**：tracking、DBSCAN baseline、failure matcher 和 profiling；
- **Artifact regression**：AP 表、诊断表、dashboard、报告字段和训练结果结构。
