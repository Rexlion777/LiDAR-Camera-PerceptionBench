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
- **23.1k+** lines of core Python and tests;
- **10+** machine-readable result tables;
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
