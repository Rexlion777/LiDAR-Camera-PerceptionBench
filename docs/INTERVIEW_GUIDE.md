# 面试讲解提纲

## 30 秒版本

我搭建了一个从 KITTI/OpenPCDet PointPillars 基线，到 Camera-LiDAR 几何验证、退化压力测试、运行质量诊断和 TensorRT 子图部署验收的完整系统。重点不是单次 AP，而是明确数据合同、定位失效来源，并用官方 AP 与在线延迟共同约束部署。最终在不越过已验证边界的前提下，将 backbone/head 从 6.745 ms 降到 3.635 ms，将 tracking association 从 47.267 ms 降到 0.836 ms。

## 三个最值得展开的问题

### 为什么不声称全 TensorRT？

因为 VFE/scatter 和 native post-processing 仍未进入已验证的 TensorRT 精度边界。面试时主动说明边界，体现对部署正确性的重视。

### 如何证明加速没有破坏精度？

先做 wrapper 与原模型一致性，再按 submodule bisection 定位误差，最后在固定 200 帧上比较官方 AP_R40，并测在线总延迟而不只测 engine latency。

### 时间同步做到了什么？

当前是相邻帧偏移代理实验，用于量化错帧对 BEV 几何的影响；没有使用真实 IMU/ego-motion 做在线补偿，因此不能包装成量产同步算法。

## 主动暴露的限制

- PointPillars 是成熟基线，项目价值来自系统评测、退化归因和部署闭环。
- 小样本微调只是管线验证。
- failure matcher 用于归因，不等同于 KITTI 官方 evaluator。
- label-free health metric 是报警信号，不是准确率真值。
