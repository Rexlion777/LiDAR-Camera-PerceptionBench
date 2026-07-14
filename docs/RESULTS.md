# 结果与证据边界

## 可直接引用的结论

| 结果 | 数值 | 样本/边界 |
|---|---:|---|
| KITTI Car / Pedestrian / Cyclist moderate 3D AP | 50.22 / 23.90 / 45.77 | 3,769 帧，官方 evaluator |
| PointPillars BEV backbone/head | 6.745 → 3.635 ms | PyTorch → TensorRT，1.86× |
| 在线检测链路 | 33.076 → 28.480 ms | 保留 PyTorch VFE/scatter 与原生后处理 |
| Tracking association | 47.267 → 0.836 ms | 50 帧，56.6× |
| 退化验收矩阵 | 107 settings / 11,200 frame-runs | 71 executed，29 partial，7 skipped |

简历中的 `6.85 → 3.68 ms` 是早期四舍五入版本；仓库使用最终保留证据中的 `6.745 → 3.635 ms`。二者描述同一子图边界，但以仓库机器可读结果为准。

## 重要负结果

- **完整 TensorRT detector**：尚未建立可复核的全链精度与动态 pillar bucket 证据，所以不做此声明。
- **时间同步**：使用相邻帧偏移作为退化代理；没有声称完成真实 IMU/ego-motion 补偿的硬件同步系统。
- **无标签健康指标**：可以提示分布漂移和异常，但不能替代官方 AP。
- **微调**：200 个训练样本、50 个验证样本、3 epochs，只说明小样本管线有效，不代表完整 KITTI 收敛或 SOTA。

## 退化发现

- 80% 点丢弃是当前矩阵中最敏感的点云稀疏设置。
- Cyclist 是类别维度最敏感对象。
- 40–60 m 是距离维度最敏感区间。
- ±2° yaw 在 20 帧代理集上产生约 30.71 px 平均重投影位移。
- ±2 帧偏移在同一代理集上产生约 7.64–7.83 m BEV 位移。

原始表格位于 `evidence/raw/`，汇总入口为 `evidence/summary.json`。
