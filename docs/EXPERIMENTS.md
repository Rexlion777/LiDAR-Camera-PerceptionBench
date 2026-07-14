# 实验矩阵

## E0：检测基线

KITTI → OpenPCDet PointPillars → 官方 AP → BEV 与 Camera 投影抽检。

## E1：输入退化

- 点云随机丢弃与稀疏化
- 距离裁剪与最大探测范围变化
- 坐标噪声与置信度阈值变化
- yaw 标定误差
- 相邻帧时间偏移代理

## E2：部署一致性

- wrapper PyTorch 对原始 OpenPCDet
- TensorRT backbone/head 对 wrapper PyTorch
- binding、shape、padding、decode 和 NMS 分段审计
- AP parity 与 latency 同时通过才接受部署边界

## E3：运行质量

- 输入点数、pillar 数与距离分布
- 预测数量、置信度、异常框与时间一致性
- 预处理、网络、后处理、tracking latency 的均值和尾延迟
- 与 AP drop 的相关性分析

## E4：关联优化

legacy 逐目标距离计算 → 向量化中心距离矩阵 → 距离门控 → 局部 assignment fallback。

实验脚本全部保留在 `scripts/lidar_system_algorithm/`，失败或仅部分完成的设置通过状态字段保留，避免只展示成功样本。
