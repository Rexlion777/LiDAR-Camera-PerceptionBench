from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import read_json_or_default, write_csv, write_json, write_markdown
from runtime.lidar_system_algorithm.tensorrt_debug_utils import numeric_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run larger-slice backbone/head-only TensorRT eval and consolidate baseline/wrapper/TRT results.")
    parser.add_argument("--input-split", default="external/OpenPCDet/data/kitti/ImageSets/val.txt")
    parser.add_argument("--openpcdet-root", default="external/OpenPCDet")
    parser.add_argument("--kitti-root", default="data/kitti_object_raw/extracted")
    parser.add_argument("--wrapper-pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt_wrapper_pytorch_core_1000")
    parser.add_argument("--trt-pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt_trt_backbone_head_only_1000")
    parser.add_argument("--baseline-pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm")
    parser.add_argument("--target-frames", type=int, default=1000)
    parser.add_argument("--attempt-full-val", action="store_true")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _read_split_ids(path: Path, max_frames: int) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids[:max_frames] if max_frames > 0 else ids


def _run_subprocess(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"command failed: {' '.join(command)}")


def _prediction_dir_subset_stats(pred_dir: Path, sample_ids: list[str]) -> dict[str, object]:
    files = [pred_dir / f"{sample_id}.txt" for sample_id in sample_ids]
    empty_preview: list[str] = []
    total_box_count = 0
    invalid_geometry_count = 0
    invalid_line_count = 0
    per_class_box_count: dict[str, int] = {}
    scores: list[float] = []

    for path in files:
        if not path.exists():
            empty_preview.append(path.name)
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            empty_preview.append(path.name)
            continue
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 16:
                invalid_line_count += 1
                continue
            total_box_count += 1
            per_class_box_count[parts[0]] = per_class_box_count.get(parts[0], 0) + 1
            try:
                score = float(parts[15])
                height = float(parts[8])
                width = float(parts[9])
                length = float(parts[10])
                location = [float(parts[11]), float(parts[12]), float(parts[13])]
                rotation_y = float(parts[14])
            except Exception:
                invalid_geometry_count += 1
                invalid_line_count += 1
                continue
            scores.append(score)
            values = [height, width, length, rotation_y, *location]
            if height <= 0 or width <= 0 or length <= 0 or not all(np.isfinite(values)):
                invalid_geometry_count += 1

    return {
        "prediction_file_count": len(files),
        "empty_prediction_file_count": len(empty_preview),
        "empty_prediction_files_preview": empty_preview[:10],
        "total_box_count": total_box_count,
        "per_class_box_count": per_class_box_count,
        "score_summary": numeric_summary(scores),
        "invalid_geometry_count": invalid_geometry_count,
        "invalid_line_count": invalid_line_count,
    }


def _latency_row(route_name: str, route_payload: dict) -> dict[str, object]:
    latency = route_payload.get("latency_summary", {}) if isinstance(route_payload, dict) else {}
    trt_core = latency.get("trt_core_ms", {}) if isinstance(latency.get("trt_core_ms"), dict) else {}
    pytorch_core = latency.get("pytorch_core_ms", {}) if isinstance(latency.get("pytorch_core_ms"), dict) else {}
    preprocess = latency.get("preprocess_ms", {}) if isinstance(latency.get("preprocess_ms"), dict) else {}
    postprocess = latency.get("nms_postprocess_ms", {}) if isinstance(latency.get("nms_postprocess_ms"), dict) else {}
    return {
        "route": route_name,
        "trt_core_ms_mean": trt_core.get("mean"),
        "trt_core_ms_p50": trt_core.get("p50"),
        "trt_core_ms_p95": trt_core.get("p95"),
        "pytorch_core_ms_mean": pytorch_core.get("mean"),
        "pytorch_core_ms_p50": pytorch_core.get("p50"),
        "pytorch_core_ms_p95": pytorch_core.get("p95"),
        "preprocess_ms_mean": preprocess.get("mean"),
        "preprocess_ms_p50": preprocess.get("p50"),
        "preprocess_ms_p95": preprocess.get("p95"),
        "nms_postprocess_ms_mean": postprocess.get("mean"),
        "nms_postprocess_ms_p50": postprocess.get("p50"),
        "nms_postprocess_ms_p95": postprocess.get("p95"),
    }


def main() -> None:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "tensorrt_backbone_head_only_fullval_eval.json"
    report_md = output_dir / "tensorrt_backbone_head_only_fullval_eval.md"
    latency_csv = output_dir / "tensorrt_backbone_head_only_fullval_latency.csv"
    audit_json = output_dir / "tensorrt_backbone_head_only_fullval_prediction_audit.json"

    split_path = _resolve(args.input_split)
    all_ids = _read_split_ids(split_path, 0)
    requested_frame_count = len(all_ids) if args.attempt_full_val else min(args.target_frames, len(all_ids))
    result_scope = "full-val" if args.attempt_full_val else f"slice-{requested_frame_count}"
    skipped_reason = None if args.attempt_full_val else "Full-val rerun skipped this turn to limit wrapper/TRT runtime cost; executed a 1000-frame slice instead."

    temp_split = output_dir / f"trt_backbone_head_only_eval_scope_{requested_frame_count}.txt"
    temp_split.write_text("\n".join(all_ids[:requested_frame_count]) + "\n", encoding="utf-8")

    runner = PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "wsl_kitti_eval_runner.py"
    baseline_json = output_dir / "tensorrt_backbone_head_only_baseline_eval_1000.json"
    _run_subprocess(
        [
            sys.executable,
            str(runner),
            "--openpcdet-root",
            str(_resolve(args.openpcdet_root)),
            "--label-dir",
            str((_resolve(args.kitti_root) / "training" / "label_2").resolve()),
            "--pred-dir",
            str(_resolve(args.baseline_pred_dir)),
            "--split-file",
            str(temp_split),
            "--output-json",
            str(baseline_json),
        ]
    )

    _run_subprocess(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "run_wrapper_pytorch_core_eval.py"),
            "--kitti-root",
            args.kitti_root,
            "--eval-max-frames",
            str(requested_frame_count),
            "--padding-strategy",
            "unique_dummy_coord_padding",
            "--zero-padded-pillars-after-vfe",
            "--pred-dir",
            args.wrapper_pred_dir,
            "--report-suffix",
            "_1000",
            "--output-dir",
            args.output_dir,
        ]
    )

    _run_subprocess(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "run_tensorrt_backbone_head_only_eval.py"),
            "--kitti-root",
            args.kitti_root,
            "--eval-max-frames",
            str(requested_frame_count),
            "--pred-dir",
            args.trt_pred_dir,
            "--report-suffix",
            "_1000",
            "--output-dir",
            args.output_dir,
        ]
    )

    baseline = read_json_or_default(baseline_json, {})
    wrapper = read_json_or_default(output_dir / "wrapper_pytorch_core_eval_1000.json", {})
    trt = read_json_or_default(output_dir / "tensorrt_backbone_head_only_eval_1000.json", {})
    selected_ids = all_ids[:requested_frame_count]
    baseline_stats = _prediction_dir_subset_stats(_resolve(args.baseline_pred_dir), selected_ids)
    wrapper_stats = _prediction_dir_subset_stats(_resolve(args.wrapper_pred_dir), selected_ids)
    trt_stats = _prediction_dir_subset_stats(_resolve(args.trt_pred_dir), selected_ids)

    baseline_dict = baseline.get("official_result_dict", {}) if isinstance(baseline, dict) else {}
    wrapper_dict = wrapper.get("official_result_dict", {}) if isinstance(wrapper, dict) else {}
    trt_dict = trt.get("official_result_dict", {}) if isinstance(trt, dict) else {}

    payload = {
        "status": "completed",
        "result_scope": result_scope,
        "frame_count": requested_frame_count,
        "skipped_reason": skipped_reason,
        "baseline": {
            "prediction_dir": str(_resolve(args.baseline_pred_dir)),
            **baseline_stats,
            "official_result_dict": baseline_dict,
        },
        "wrapper_pytorch_core": {**wrapper, "prediction_audit": wrapper_stats},
        "trt_backbone_head_only": {**trt, "prediction_audit": trt_stats},
        "ap_delta_vs_baseline": {
            "wrapper": {
                key: (wrapper_dict.get(key) - baseline_dict.get(key)) if (wrapper_dict.get(key) is not None and baseline_dict.get(key) is not None) else None
                for key in ["Car_3d/moderate_R40", "Pedestrian_3d/moderate_R40", "Cyclist_3d/moderate_R40"]
            },
            "trt": {
                key: (trt_dict.get(key) - baseline_dict.get(key)) if (trt_dict.get(key) is not None and baseline_dict.get(key) is not None) else None
                for key in ["Car_3d/moderate_R40", "Pedestrian_3d/moderate_R40", "Cyclist_3d/moderate_R40"]
            },
        },
    }

    latency_rows = [_latency_row("trt_backbone_head_only", trt)]
    write_json(report_json, payload)
    write_json(audit_json, {"baseline": baseline, "wrapper_pytorch_core": wrapper, "trt_backbone_head_only": trt})
    write_csv(latency_csv, latency_rows, fieldnames=list(latency_rows[0].keys()) if latency_rows else ["route", "frame_id"])
    write_markdown(
        report_md,
        "# TensorRT Backbone/Head-only Larger Eval\n\n"
        f"- Scope: `{result_scope}`\n"
        f"- Frame count: `{requested_frame_count}`\n"
        f"- Skipped reason: `{skipped_reason}`\n"
        f"- Baseline Car/Ped/Cyc moderate AP_R40: `{baseline_dict.get('Car_3d/moderate_R40')}` / `{baseline_dict.get('Pedestrian_3d/moderate_R40')}` / `{baseline_dict.get('Cyclist_3d/moderate_R40')}`\n"
        f"- Wrapper Car/Ped/Cyc moderate AP_R40: `{wrapper_dict.get('Car_3d/moderate_R40')}` / `{wrapper_dict.get('Pedestrian_3d/moderate_R40')}` / `{wrapper_dict.get('Cyclist_3d/moderate_R40')}`\n"
        f"- TRT Car/Ped/Cyc moderate AP_R40: `{trt_dict.get('Car_3d/moderate_R40')}` / `{trt_dict.get('Pedestrian_3d/moderate_R40')}` / `{trt_dict.get('Cyclist_3d/moderate_R40')}`\n"
        f"- Wrapper empty prediction files / invalid geometry: `{wrapper_stats.get('empty_prediction_file_count')}` / `{wrapper_stats.get('invalid_geometry_count')}`\n"
        f"- TRT empty prediction files / invalid geometry: `{trt_stats.get('empty_prediction_file_count')}` / `{trt_stats.get('invalid_geometry_count')}`\n"
        f"- TRT core mean/p50/p95 ms: `{trt.get('latency_summary', {}).get('trt_core_ms', {}).get('mean')}` / `{trt.get('latency_summary', {}).get('trt_core_ms', {}).get('p50')}` / `{trt.get('latency_summary', {}).get('trt_core_ms', {}).get('p95')}`\n"
        f"- TRT total boxes: `{trt_stats.get('total_box_count')}`\n",
    )
    print(json.dumps({"status": "completed", "report": str(report_json)}, indent=2))


if __name__ == "__main__":
    main()
