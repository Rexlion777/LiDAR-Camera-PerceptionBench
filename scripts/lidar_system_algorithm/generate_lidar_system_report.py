from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import read_json_or_default, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README, report, and resume bullets.")
    parser.add_argument("--input-dir", default="reports/lidar_system_algorithm", help="Input report directory.")
    parser.add_argument("--output", default="reports/lidar_system_algorithm/lidar_system_algorithm_report.md", help="Markdown report output path.")
    return parser.parse_args()


def _stage(summary: list[dict], name: str):
    for row in summary:
        if row.get("stage") == name:
            return row.get("mean_ms")
    return None


def main() -> None:
    args = parse_args()
    input_dir = (PROJECT_ROOT / args.input_dir).resolve() if not Path(args.input_dir).is_absolute() else Path(args.input_dir)
    output_path = (PROJECT_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    readme_path = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "README.md"
    resume_path = input_dir / "resume_bullets.md"

    env = read_json_or_default(input_dir / "environment_audit.json", {})
    online = read_json_or_default(input_dir / "online_latency_profile.json", {})
    voxel = read_json_or_default(input_dir / "voxelization_ablation.json", {})
    tracking = read_json_or_default(input_dir / "tracking_optimized_summary.json", {})
    trt_wrapper = read_json_or_default(input_dir / "tensorrt_real_sample_wrapper.json", {})
    trt_bucketed = read_json_or_default(input_dir / "tensorrt_bucketed_core_report.json", {})
    trt_bucketed_eval = read_json_or_default(input_dir / "tensorrt_bucketed_kitti_eval.json", {})
    robustness = read_json_or_default(input_dir / "calibration_sync_robustness.json", {})
    dashboard = read_json_or_default(input_dir / "eval_dashboard.json", {})
    official_eval = read_json_or_default(input_dir / "kitti_official_eval.json", {})
    failure = read_json_or_default(input_dir / "failure_matcher_summary.json", {})

    eval_dict = official_eval.get("official_result_dict", {}) if isinstance(official_eval, dict) else {}
    online_summary = online.get("online_summary", []) if isinstance(online, dict) else []
    debug_summary = online.get("debug_summary", []) if isinstance(online, dict) else []
    voxel_rows = voxel.get("summary_rows", []) if isinstance(voxel, dict) else []
    yaw_summary = robustness.get("yaw_summary", []) if isinstance(robustness, dict) else []
    time_summary = robustness.get("time_offset_summary", []) if isinstance(robustness, dict) else []
    failure_totals = failure.get("totals", {}) if isinstance(failure, dict) else {}

    report = f"""# LiDAR System Algorithm Report

## Pipeline Overview

This project is a compact LiDAR 3D perception system runtime rather than a demo-only PointPillars environment. The current scope includes KITTI ingestion, Camera-LiDAR calibration projection, DBSCAN baseline, PointPillars/OpenPCDet inference, KITTI native CUDA official eval, online latency schema, optimized tracking, TensorRT core wrapper work, calibration/time-offset robustness analysis, and GT-based failure analysis.

## Online Latency

- online_total_ms: `{_stage(online_summary, "online_total_ms")}`
- model_forward_ms: `{_stage(online_summary, "model_forward_ms")}`
- nms_ms: `{_stage(online_summary, "nms_ms")}`
- tracking_ms: `{_stage(online_summary, "tracking_ms")}`
- visualization_ms debug-only: `{_stage(debug_summary, "visualization_ms")}`

The previous mixed total was dominated by visualization. The current `online_total_ms` excludes visualization, image saving, and report writing.

## KITTI Official Eval

- Car moderate 3D AP_R40: `{eval_dict.get("Car_3d/moderate_R40", "N/A")}`
- Pedestrian moderate 3D AP_R40: `{eval_dict.get("Pedestrian_3d/moderate_R40", "N/A")}`
- Cyclist moderate 3D AP_R40: `{eval_dict.get("Cyclist_3d/moderate_R40", "N/A")}`

These AP values come from local evaluation of the provided checkpoint and remain the source of record for detection quality.

## Tracking Optimization

- association_method: `{tracking.get("association_method", "unknown")}`
- scipy_available: `{tracking.get("scipy_available", "unknown")}`
- legacy association latency ms: `{tracking.get("average_legacy_association_latency_ms", "unknown")}`
- optimized association latency ms: `{tracking.get("average_association_latency_ms", "unknown")}`
- average track count: `{tracking.get("average_track_count", "unknown")}`

This is a lightweight detection-to-tracking pipeline. It does not claim MOTA or IDF1.

## TensorRT Deployment Scope

### Real-Sample Core Wrapper

- status: `{trt_wrapper.get("status", "unknown")}`
- mean_trt_core_ms: `{trt_wrapper.get("mean_trt_core_ms", "unknown")}`
- mean_pytorch_core_ms: `{trt_wrapper.get("mean_pytorch_core_ms", "unknown")}`
- mean_online_total_ms: `{trt_wrapper.get("mean_online_total_ms", "unknown")}`
- mean_pytorch_online_total_ms: `{trt_wrapper.get("mean_pytorch_online_total_ms", "unknown")}`
- truncated_frame_count: `{trt_wrapper.get("truncated_frame_count", "unknown")}`

### Bucketed Capacity Analysis

- status: `{trt_bucketed.get("status", "unknown")}`
- successful_bucket_sizes: `{trt_bucketed.get("successful_bucket_sizes", "unknown")}`
- overall_truncation_rate: `{trt_bucketed.get("overall_truncation_rate", "unknown")}`
- bucket_hit_distribution: `{trt_bucketed.get("bucket_hit_distribution", "unknown")}`
- mean_trt_core_ms: `{trt_bucketed.get("mean_trt_core_ms", "unknown")}`
- mean_pytorch_core_ms: `{trt_bucketed.get("mean_pytorch_core_ms", "unknown")}`
- mean_online_total_ms: `{trt_bucketed.get("mean_online_total_ms", "unknown")}`
- mean_pytorch_online_total_ms: `{trt_bucketed.get("mean_pytorch_online_total_ms", "unknown")}`

TensorRT still covers the PointPillars core only. Voxelization and NMS remain outside the engine, and no FP16 acceleration is claimed.

## TensorRT Bucketed Official Eval

- status: `{trt_bucketed_eval.get("status", "unknown")}`
- AP delta: `{trt_bucketed_eval.get("ap_delta", "unavailable")}`

If this section is skipped, the report preserves the blocker instead of inventing AP numbers.

## Calibration / Time Offset Robustness

- yaw summary: `{yaw_summary}`
- time offset summary: `{time_summary}`

The time-offset experiment is still an adjacent-frame proxy and not a real IMU/ego-motion fusion result.

## Failure Analysis

- matcher status: `{failure.get("status", "unknown")}`
- TP / FP / FN: `{failure_totals}`
- by_class: `{failure.get("by_class", "unknown")}`
- by_range: `{failure.get("by_range", "unknown")}`

This matcher is an analysis tool for FP/FN attribution by class/range/difficulty. It does not replace KITTI official evaluation.

## Voxelization Ablation

{chr(10).join(f"- pillar_size={row.get('pillar_size')}: pillar_count_mean={row.get('pillar_count_mean')}, preprocess_mean_ms={row.get('preprocess_mean_ms')}" for row in voxel_rows) if voxel_rows else "- unavailable"}

## Dashboard

- eval dashboard: `reports/lidar_system_algorithm/eval_dashboard.md`
- TensorRT bucket latency figure: `projects/lidar_system_algorithm/figures/tensorrt_bucket_latency_vs_capacity.png`
- failure analysis gallery: `projects/lidar_system_algorithm/figures/failure_case_gallery.png`

## Mapping to Hesai System Algorithm Engineer

1. Sensor AI algorithm import: OpenPCDet adapter, checkpoint runtime, KITTI-format output.
2. Data pipeline and evaluation: KITTI ingestion, native CUDA official eval, GT failure analysis.
3. Calibration quality: Camera-LiDAR projection plus perturbation sensitivity.
4. Time sync awareness: adjacent-frame offset proxy and downstream impact analysis.
5. System bottleneck analysis: online latency schema, tracking optimization, bucketed TensorRT capacity-vs-truncation tradeoff.
6. Edge deployment preparation: realistic fixed-shape TensorRT core wrapper with honest deployment scope.

## Conclusion

The project now reads as a small LiDAR perception system algorithm runtime with bottleneck analysis, error analysis, and deployment pre-research. It stays honest about what is and is not inside TensorRT, and it does not overstate proxy experiments as production capabilities.
"""
    write_markdown(output_path, report)

    readme = f"""# LiDAR System Algorithm

## Goal

Build a reproducible offline LiDAR 3D perception runtime around KITTI, Camera-LiDAR calibration, DBSCAN baseline, PointPillars/OpenPCDet inference, official KITTI eval, online latency profiling, optimized tracking, TensorRT core deployment wrapper, calibration/time-offset robustness, and GT-based failure analysis.

## Honest Runtime Scope

- `online_total_ms` excludes visualization, report writing, and image saving.
- TensorRT covers PointPillars core only; voxelization and NMS remain outside the engine.
- The original 64-pillar TensorRT wrapper was a deployment precheck. The bucketed wrapper is a more realistic capacity study, not a full dynamic TensorRT detector.
- No FP16 acceleration, full TensorRT detector, training improvement, or IMU fusion is claimed.

## Suggested Commands

```bash
PYTHONPATH=. python scripts/lidar_system_algorithm/build_tensorrt_bucketed_engines.py --bucket-sizes 4096,8192,12000,16000,20000 --output-dir reports/lidar_system_algorithm
PYTHONPATH=. python scripts/lidar_system_algorithm/run_tensorrt_bucketed_core_wrapper.py --frames 50 --bucket-sizes 4096,8192,12000,16000,20000 --output-dir reports/lidar_system_algorithm
PYTHONPATH=. python scripts/lidar_system_algorithm/run_kitti_failure_matcher.py --pred-dir projects/lidar_system_algorithm/results/kitti_eval_txt --output-dir reports/lidar_system_algorithm
PYTHONPATH=. python scripts/lidar_system_algorithm/generate_eval_dashboard.py --input-dir reports/lidar_system_algorithm --version v2 --output-dir reports/lidar_system_algorithm
PYTHONPATH=. python scripts/lidar_system_algorithm/generate_lidar_system_report.py --input-dir reports/lidar_system_algorithm --output reports/lidar_system_algorithm/lidar_system_algorithm_report.md
```

## Key Outputs

- Online latency: `reports/lidar_system_algorithm/online_latency_profile.json`
- Tracking optimization: `reports/lidar_system_algorithm/tracking_optimized_summary.json`
- TensorRT bucketed capacity report: `reports/lidar_system_algorithm/tensorrt_bucketed_core_report.md`
- TensorRT bucketed eval: `reports/lidar_system_algorithm/tensorrt_bucketed_kitti_eval.md`
- Failure matcher: `reports/lidar_system_algorithm/failure_analysis_report.md`
- Eval dashboard v2: `reports/lidar_system_algorithm/eval_dashboard.md`
- Resume bullets: `reports/lidar_system_algorithm/resume_bullets.md`

## Current Highlights

- Online perception latency mean: `{_stage(online_summary, "online_total_ms")}` ms
- Tracking association mean: `{tracking.get("average_association_latency_ms", "unknown")}` ms
- KITTI Car/Ped/Cyclist moderate 3D AP_R40: `{eval_dict.get("Car_3d/moderate_R40", "N/A")}`, `{eval_dict.get("Pedestrian_3d/moderate_R40", "N/A")}`, `{eval_dict.get("Cyclist_3d/moderate_R40", "N/A")}`
- GT matcher TP / FP / FN: `{failure_totals}`
- Bucketed TensorRT truncation rate: `{trt_bucketed.get("overall_truncation_rate", "unknown")}`

## Limitations

- Bucketed TensorRT still does not equal a full dynamic detector.
- TensorRT official eval may be skipped if bucketed export/build/runtime blocks occur.
- GT matcher is for error attribution, not a leaderboard-equivalent evaluator.
- Time-offset robustness is still a proxy analysis.
"""
    write_markdown(readme_path, readme)

    resume = """# Resume Bullets

## A. 9.0 稳健版

基于 KITTI / OpenPCDet / PointPillars 搭建 LiDAR 3D 感知系统 runtime，完成点云读取、坐标变换、Camera-LiDAR 标定投影、BEV 可视化、DBSCAN baseline、PointPillars 推理接入与 KITTI native CUDA 官方评估，形成从原始点云到 3D box、相机投影验证和评估报告的可复现数据管线。

围绕系统算法工程导入，补充 online latency schema、tracking association 优化、pillar 参数实验、TensorRT real-sample core wrapper、标定扰动与时间同步 proxy 鲁棒性分析，并区分在线感知时延与离线可视化开销；TensorRT 仅覆盖 core，不夸大为完整 detector 或 FP16 加速。

## B. 9.3/9.5 强化版

基于 KITTI / OpenPCDet / PointPillars 构建 LiDAR 3D 感知系统 runtime，并在官方 KITTI eval 之外补充 GT-based failure matcher，用于按类别、距离分桶和难度分桶分析 TP / FP / FN；同时完成 realistic / bucketed TensorRT core wrapper，量化 pillar capacity、truncation rate、bucket hit distribution 与 core/wrapper latency tradeoff。

项目重点不是刷榜，而是围绕系统算法链路完成模型导入、后处理、在线时延建模、tracking 工程化优化、部署预研、标定/时间同步质量分析和错误归因。TensorRT 仅覆盖 PointPillars core，voxelization 与 NMS 仍在 engine 外，相关限制在报告中明确保留。

## C. 面试口述版

这个项目我没有把重点放在“重新训练刷 AP”，而是把它做成了一个可讲系统算法链路的 LiDAR 3D 感知 runtime。前面做了 KITTI 点云读取、Camera-LiDAR 标定解析、投影验证、DBSCAN baseline 和 PointPillars/OpenPCDet 推理，后面补了 KITTI official eval、online latency 拆分、tracking association 优化、标定扰动和 time offset proxy 分析。部署侧我没有把 TensorRT 说成 full detector，而是诚实地做了 core-only wrapper 和 bucketed capacity/truncation analysis，说明 voxelization 和 NMS 还在 engine 外。除此之外我还加了 GT matcher，把 FP/FN 按类别和距离分桶做错误归因，这样项目更像系统算法工程里的 runtime、瓶颈分析和质量评估工作，而不是单纯跑 demo。
"""
    write_markdown(resume_path, resume)
    print(f"Saved consolidated report: {output_path}")
    print(f"Saved README: {readme_path}")
    print(f"Saved resume bullets: {resume_path}")


if __name__ == "__main__":
    main()
