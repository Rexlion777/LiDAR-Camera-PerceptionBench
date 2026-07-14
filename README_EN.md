# LiDAR-Camera Perception, Calibration/Sync and TensorRT Deployment Benchmark

[中文首页](README.md) · [Architecture](docs/ARCHITECTURE.md) · [Code Map](docs/CODE_MAP.md) · [Experiments](docs/EXPERIMENTS.md) · [Reproduction](docs/REPRODUCIBILITY.md)

![System overview](assets/portfolio/01_system_overview.png)

An engineering-oriented 3D perception system spanning **KITTI/OpenPCDet data pipelines, PointPillars training and evaluation, camera-LiDAR geometry, controlled sensor perturbations, TensorRT optimization, online quality analytics, and multi-object association**.

## Highlights

| Perception | Experiments | Deployment | Association |
|---|---|---|---|
| **3,769** KITTI validation frames | **107** stress-test settings | **6.745 → 3.635 ms** | **47.267 → 0.836 ms** |
| Official Car / Ped / Cyc 3D AP | **11,200 frame-runs** | **1.86×** TensorRT speedup | **56.6×** speedup |

![Accuracy and deployment](assets/portfolio/02_accuracy_and_deployment.png)

## Engineering innovations and evidence

### 1. Accuracy-preserving deployment acceptance

The deployment study combines official AP parity, matched-frame latency, P50/P95 tails, per-frame speedup, and stage-level profiling. It therefore validates both task behavior and runtime behavior instead of presenting a single best-case timing number.

<p align="center">
  <img src="assets/portfolio/05_deployment_parity.png" width="49%" alt="TensorRT deployment parity">
  <img src="assets/portfolio/06_latency_distribution.png" width="49%" alt="Matched-frame latency distribution">
</p>
<p align="center">
  <img src="assets/portfolio/07_per_frame_runtime.png" width="49%" alt="Per-frame runtime behavior">
  <img src="assets/portfolio/08_stage_profile.png" width="49%" alt="Online stage profiler">
</p>

### 2. Failure localization beyond aggregate AP

Controlled degradation is decomposed by class, distance, prediction population, confidence, and distribution drift. The resulting evidence can support sensor-health thresholds, data collection priorities, and failure attribution.

<p align="center">
  <img src="assets/portfolio/10_class_robustness.png" width="49%" alt="Class-level robustness">
  <img src="assets/portfolio/11_range_robustness.png" width="49%" alt="Range-aware failure heatmap">
</p>
<p align="center"><img src="assets/portfolio/12_prediction_health.png" width="72%" alt="Prediction health observability"></p>

### 3. Vectorized association with state auditing

Legacy and vectorized association are replayed on identical detections. Alongside the 56.6× aggregate speedup, the benchmark records matrix workload, gated pairs, visible tracks, and lifecycle events.

<p align="center">
  <img src="assets/portfolio/13_tracking_latency.png" width="49%" alt="Tracking association latency">
  <img src="assets/portfolio/14_tracking_lifecycle.png" width="49%" alt="Track lifecycle audit">
</p>

### 4. Resource-aware and traceable experimentation

Voxel-size ablations connect sparse-tensor density to memory and preprocessing tails; decoded-output studies retain class, confidence, and range structure. All public charts are regenerated from **18 CSV tables containing 16,000+ records**.

<p align="center">
  <img src="assets/portfolio/09_voxelization_ablation.png" width="49%" alt="Voxelization resource ablation">
  <img src="assets/portfolio/15_inference_population.png" width="49%" alt="Inference output population">
</p>
<p align="center"><img src="assets/portfolio/16_evidence_coverage.png" width="72%" alt="Machine-readable evidence coverage"></p>

## System capabilities

- **3D perception pipeline:** KITTI I/O, voxelization, PointPillars training/fine-tuning, inference, decode/NMS, official AP, BEV and camera projection.
- **Geometry and calibration:** LiDAR-camera coordinate transforms, 3D box projection, yaw/translation sensitivity, and adjacent-frame synchronization analysis.
- **Robustness evaluation:** point dropout, range crop, noise, score threshold, calibration perturbation, temporal offset, and per-class/per-range diagnostics.
- **TensorRT deployment:** wrapper parity, binding and dynamic-shape audits, submodule bisection, AP alignment, stage-level profiling, and online latency measurement.
- **Runtime analytics:** prediction drift, confidence/range distribution, anomalous boxes, temporal consistency, latency spikes, and tracking association.

![Robustness landscape](assets/portfolio/03_robustness_landscape.png)

![Calibration sensitivity](assets/portfolio/04_calibration_sensitivity.png)

## Engineering footprint

- **19** reusable runtime modules;
- **50** experiment, audit, deployment, and reporting scripts;
- **63** contract and regression test files;
- **23.5k+** lines of core Python and tests;
- **18** machine-readable result tables with **16,000+** records;
- **16** reproducible portfolio figures;
- reproducible portfolio figures generated from repository evidence by [`generate_portfolio_figures.py`](scripts/portfolio/generate_portfolio_figures.py).

## Repository map

- `runtime/lidar_system_algorithm/`: calibration, transforms, evaluation, tracking, profiling, deployment, and diagnostics;
- `scripts/lidar_system_algorithm/`: training, official evaluation, robustness matrices, TensorRT debugging, and report generation;
- `scripts/portfolio/`: reproducible public visualization generator;
- `tests/`: geometry, runtime, deployment, and report regression tests;
- `evidence/raw/`: AP, failure analysis, calibration, latency, and runtime-quality tables;
- `assets/portfolio/`: curated public figures.

See the [Code Map](docs/CODE_MAP.md), [Experiment Matrix](docs/EXPERIMENTS.md), and [Reproduction Guide](docs/REPRODUCIBILITY.md) for details.

## License

Original code, documentation, and figures are available under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). Noncommercial study, research, modification, and redistribution are permitted; commercial use requires separate permission.
