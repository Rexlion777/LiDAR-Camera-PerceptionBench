# 代码地图

## 面试优先阅读路径

1. `runtime/lidar_system_algorithm/kitti_io.py`：KITTI 点云、标注与标定文件入口。
2. `calibration.py` + `transforms.py`：Camera-LiDAR 坐标链和投影合同。
3. `openpcdet_adapter.py`：OpenPCDet 接入与预测导出边界。
4. `pointpillars_wrapper_runtime.py`：PyTorch / TensorRT 子图包装与数据合同。
5. `deployment_acceptance.py`：退化矩阵、执行状态和验收结果组织。
6. `profiling.py` + `online_latency.py`：分阶段延迟、warm-up、分位数与在线总延迟。
7. `failure_matcher.py`：按类别、距离和失效类型进行归因。
8. `tracking.py`：legacy 与向量化关联实现。

## 目录职责

| 路径 | 内容 | 面试价值 |
|---|---|---|
| `runtime/lidar_system_algorithm/` | 可复用运行库 | 展示坐标合同、评测、部署、诊断的核心实现 |
| `scripts/lidar_system_algorithm/` | 49 个实验与审计入口 | 展示从训练、评测到 TensorRT bisection 的工程闭环 |
| `tests/` | 63 个测试文件 | 展示以合同和回归测试约束复杂实验链路 |
| `configs/` | PointPillars 子集微调配置 | 展示可复现实验边界 |
| `evidence/raw/` | 机器可读结果表 | 让 README 指标可追溯，而非仅展示图片 |
| `assets/figures/` | 投影、BEV、延迟和诊断图 | 支撑视觉检查与面试讲解 |

## TensorRT 调试链

`audit_tensorrt_binding_contract.py` → `debug_wrapper_pytorch_core_parity.py` → `debug_tensorrt_submodule_bisection.py` → `debug_tensorrt_backbone_head_only_diff.py` → `run_tensorrt_backbone_head_only_eval.py` → `profile_tensorrt_backbone_head_online_pipeline.py`

这条链路的重点不是“跑通 engine”，而是逐级验证 shape、binding、decode、padding、精度和在线端到端收益，最终收敛到可被证据支持的部署边界。
