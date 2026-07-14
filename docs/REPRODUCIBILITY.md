# 复现说明

## 轻量核心测试

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,visualization]"
pytest -q \
  tests/test_lidar_calibration.py \
  tests/test_lidar_transforms.py \
  tests/test_lidar_tracking.py \
  tests/test_lidar_tracking_optimized.py \
  tests/test_lidar_dbscan_baseline.py \
  tests/test_lidar_failure_matcher.py
```

## 完整 PointPillars / TensorRT 实验

完整实验需要自行准备 KITTI、OpenPCDet、兼容版本的 PyTorch/CUDA/TensorRT、PointPillars 权重，并在脚本参数中指定路径。仓库不会提交数据集、权重、ONNX 或 engine。

建议顺序：

1. 运行 `run_environment_audit.py` 检查环境。
2. 运行 `run_kitti_pointcloud_pipeline.py` 验证数据与标定。
3. 运行 `run_kitti_official_eval.py` 建立 PyTorch 基线。
4. 运行 wrapper parity 和 TensorRT bisection 脚本。
5. 仅在子图误差、AP parity 和在线延迟均通过后接受部署结果。

机器相关路径均通过命令行参数或环境变量传入，不写入仓库。
