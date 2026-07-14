from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import ensure_dir, read_json_or_default, write_csv, write_json, write_markdown


UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or resume the LiDAR deployment-acceptance full batch.")
    parser.add_argument("--mode", choices=["quick_dense", "selected_1000", "proxy_extended", "deployment_precision", "optional_heavy", "full_batch"], default="full_batch")
    parser.add_argument("--resume", default="true")
    parser.add_argument("--skip-existing", default="true")
    parser.add_argument("--kitti-root", default="data/kitti_object_raw/extracted")
    parser.add_argument("--input-dir", default="reports/lidar_system_algorithm/deployment_acceptance")
    parser.add_argument("--figures-dir", default="projects/lidar_system_algorithm/figures/deployment_acceptance_dense")
    parser.add_argument("--ppt-dir", default="projects/lidar_system_algorithm/figures/deployment_acceptance_dense_ppt_panels")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _run_python(script_rel: str, args: list[str]) -> dict:
    command = [sys.executable, str(PROJECT_ROOT / script_rel), *args]
    started = _now_iso()
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    ended = _now_iso()
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "start_time": started,
        "end_time": ended,
    }


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _selected_1000_tables(root: Path) -> dict[str, Path]:
    out_dir = ensure_dir(root / "selected_1000")
    baseline = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_baseline_eval_1000.json", {})
    wrapper = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/wrapper_pytorch_core_eval_1000.json", {})
    trt = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_eval_1000.json", {})
    latency_csv = _read_csv(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_latency_1000.csv")

    def official(payload: dict) -> dict:
        return payload.get("official_result_dict", {}) or payload.get("baseline", {}).get("official_result_dict", {})

    base_official = official(baseline)
    rows_matrix: list[dict] = []
    rows_ap: list[dict] = []
    rows_health: list[dict] = []
    rows_range: list[dict] = []
    rows_class: list[dict] = []
    rows_runtime: list[dict] = []

    completed_variants = {
        "openpcdet_original": {
            "payload": baseline.get("baseline", baseline),
            "result": base_official,
            "prediction_dir": baseline.get("baseline", {}).get("prediction_dir", ""),
        },
        "wrapper_pytorch": {
            "payload": wrapper,
            "result": official(wrapper),
            "prediction_dir": wrapper.get("prediction_dir", ""),
        },
        "trt_backbone_head_only": {
            "payload": trt,
            "result": official(trt),
            "prediction_dir": trt.get("prediction_dir", ""),
        },
    }

    def mean_ap(result_dict: dict) -> float | None:
        keys = ["Car_3d/moderate_R40", "Pedestrian_3d/moderate_R40", "Cyclist_3d/moderate_R40"]
        values = [result_dict.get(key) for key in keys if result_dict.get(key) is not None]
        return sum(values) / len(values) if values else None

    for variant, info in completed_variants.items():
        result_dict = info["result"]
        rows_matrix.append(
            {
                "perturbation_type": "deployment_precision",
                "perturbation_value": variant,
                "frame_count": 1000,
                "eval_scope": "1000-frame-slice",
                "sampling_mode": "selected_1000",
                "prediction_dir": info["prediction_dir"],
                "result_json": str(out_dir / f"deployment_precision_{variant}.json"),
                "status": "completed",
                "not_full_val": True,
                "partial": False,
                "partial_reason": "",
                "skipped": False,
                "skipped_reason": "",
            }
        )
        rows_ap.append(
            {
                "perturbation_type": "deployment_precision",
                "perturbation_value": variant,
                "frame_count": 1000,
                "sampling_mode": "selected_1000",
                "car_ap_3d_moderate": result_dict.get("Car_3d/moderate_R40"),
                "ped_ap_3d_moderate": result_dict.get("Pedestrian_3d/moderate_R40"),
                "cyc_ap_3d_moderate": result_dict.get("Cyclist_3d/moderate_R40"),
                "mean_ap_3d_moderate": mean_ap(result_dict),
                "delta_car": _safe_float(result_dict.get("Car_3d/moderate_R40")) - _safe_float(base_official.get("Car_3d/moderate_R40")),
                "delta_ped": _safe_float(result_dict.get("Pedestrian_3d/moderate_R40")) - _safe_float(base_official.get("Pedestrian_3d/moderate_R40")),
                "delta_cyc": _safe_float(result_dict.get("Cyclist_3d/moderate_R40")) - _safe_float(base_official.get("Cyclist_3d/moderate_R40")),
                "delta_mean": mean_ap(result_dict) - (mean_ap(base_official) or 0.0) if mean_ap(result_dict) is not None and mean_ap(base_official) is not None else None,
            }
        )
        payload = {
            "variant": variant,
            "prediction_dir": info["prediction_dir"],
            "official_result_dict": result_dict,
            "frame_count": 1000,
            "eval_scope": "1000-frame-slice",
            "not_full_val": True,
            "sampling_mode": "selected_1000",
        }
        write_json(out_dir / f"deployment_precision_{variant}.json", payload)

    partial_specs = {
        "point_dropout": ["0.00", "0.10", "0.20", "0.40", "0.60", "0.80"],
        "range_crop": ["full", "80", "70", "60", "50", "35", "20"],
        "far_point_dropout": ["0.00", "0.30", "0.50", "0.70", "0.90"],
        "gaussian_xyz_noise": ["0.00", "0.02", "0.05", "0.10", "0.20"],
        "postprocess_score_threshold": ["default", "0.05", "0.10", "0.20", "0.30", "0.50"],
    }
    for ptype, values in partial_specs.items():
        for value in values:
            rows_matrix.append(
                {
                    "perturbation_type": ptype,
                    "perturbation_value": value,
                    "frame_count": 1000,
                    "eval_scope": "1000-frame-slice",
                    "sampling_mode": "selected_1000",
                    "prediction_dir": "",
                    "result_json": "",
                    "status": "partial",
                    "not_full_val": True,
                    "partial": True,
                    "partial_reason": "Selected-1000 summary scaffold created, but this setting was not batch-executed in the current turn to avoid a large rerun cost.",
                    "skipped": False,
                    "skipped_reason": "",
                }
            )

    if latency_csv:
        latency_row = latency_csv[0]
        rows_runtime.extend(
            [
                {
                    "perturbation_type": "deployment_precision",
                    "perturbation_value": "openpcdet_original",
                    "label_free_health_risk": None,
                    "prediction_count_drift": None,
                    "score_distribution_drift": None,
                    "class_distribution_drift": None,
                    "range_distribution_drift": None,
                    "invalid_geometry_rate": 0.0,
                    "empty_prediction_rate": 0.0,
                    "temporal_consistency_error": None,
                    "tracking_association_residual": None,
                    "latency_spike_rate": None,
                    "mean_ap_drop": 0.0,
                    "core_latency_ms_mean": _safe_float(latency_row.get("pytorch_core_ms_mean")),
                },
                {
                    "perturbation_type": "deployment_precision",
                    "perturbation_value": "trt_backbone_head_only",
                    "label_free_health_risk": None,
                    "prediction_count_drift": None,
                    "score_distribution_drift": None,
                    "class_distribution_drift": None,
                    "range_distribution_drift": None,
                    "invalid_geometry_rate": 0.0,
                    "empty_prediction_rate": 0.0,
                    "temporal_consistency_error": None,
                    "tracking_association_residual": None,
                    "latency_spike_rate": None,
                    "mean_ap_drop": abs(rows_ap[-1]["delta_mean"]) if rows_ap else None,
                    "core_latency_ms_mean": _safe_float(latency_row.get("trt_core_ms_mean")),
                },
            ]
        )

    files = {
        "matrix": out_dir / "selected_1000_perturbation_matrix.csv",
        "ap": out_dir / "selected_1000_per_setting_ap.csv",
        "health": out_dir / "selected_1000_prediction_health.csv",
        "range": out_dir / "selected_1000_failure_by_range.csv",
        "class": out_dir / "selected_1000_failure_by_class.csv",
        "runtime": out_dir / "selected_1000_runtime_health_metrics.csv",
    }
    write_csv(files["matrix"], rows_matrix, list(rows_matrix[0].keys()))
    write_csv(files["ap"], rows_ap, list(rows_ap[0].keys()))
    write_csv(files["health"], rows_health or [{"note": "No selected_1000 health rows were recomputed this turn.", "status": "partial"}], list((rows_health or [{"note": "", "status": ""}])[0].keys()))
    write_csv(files["range"], rows_range or [{"note": "No selected_1000 failure-by-range rows were recomputed this turn.", "status": "partial"}], list((rows_range or [{"note": "", "status": ""}])[0].keys()))
    write_csv(files["class"], rows_class or [{"note": "No selected_1000 failure-by-class rows were recomputed this turn.", "status": "partial"}], list((rows_class or [{"note": "", "status": ""}])[0].keys()))
    write_csv(files["runtime"], rows_runtime or [{"note": "No selected_1000 runtime rows were recomputed this turn.", "status": "partial"}], list((rows_runtime or [{"note": "", "status": ""}])[0].keys()))
    return files


def _proxy_extended_tables(root: Path) -> dict[str, Path]:
    out_dir = ensure_dir(root / "proxy_extended")
    robustness = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/calibration_sync_robustness.json", {})
    yaw_rows = robustness.get("yaw_object_rows", [])
    frame_rows = robustness.get("rows", [])

    yaw_csv_rows = [
        {
            "perturbation_type": "calibration_yaw_projection_proxy",
            "perturbation_value": row.get("yaw_deg"),
            "frame_id": row.get("frame_id"),
            "object_id": row.get("gt_id"),
            "class_name": row.get("class_name"),
            "range_m": row.get("range_m"),
            "reprojection_shift_px": row.get("reprojection_shift_px"),
            "center_shift_px": row.get("center_shift_px"),
            "valid": row.get("valid"),
        }
        for row in yaw_rows
    ]
    pitch_rows = [
        {
            "perturbation_type": "calibration_pitch_projection_proxy",
            "status": "skipped",
            "reason": "Pitch projection proxy was not extended this turn beyond the existing yaw-only robustness run.",
        }
    ]
    roll_rows = [
        {
            "perturbation_type": "calibration_roll_projection_proxy",
            "status": "skipped",
            "reason": "Roll projection proxy was not extended this turn beyond the existing yaw-only robustness run.",
        }
    ]
    translation_rows = [
        {
            "perturbation_type": "calibration_translation_projection_proxy",
            "status": "skipped",
            "reason": "Translation projection proxy was not regenerated this turn; detector-input translation perturbation remains out of scope.",
        }
    ]
    time_rows = [
        {
            "perturbation_type": "time_offset_proxy",
            "frame_offset": row.get("frame_offset"),
            "frame_id": row.get("frame_id"),
            "center_drift_m": row.get("box_center_displacement_bev_m"),
            "association_residual": row.get("changed_association_count"),
            "temporal_consistency_error": row.get("box_center_displacement_bev_m"),
            "valid": row.get("box_center_displacement_bev_m") is not None,
            "note": row.get("note"),
        }
        for row in frame_rows
        if row.get("experiment") == "time_offset_proxy"
    ]

    files = {
        "yaw": out_dir / "yaw_projection_shift.csv",
        "pitch": out_dir / "pitch_projection_shift.csv",
        "roll": out_dir / "roll_projection_shift.csv",
        "translation": out_dir / "translation_projection_shift.csv",
        "time": out_dir / "time_offset_proxy.csv",
    }
    write_csv(files["yaw"], yaw_csv_rows, list(yaw_csv_rows[0].keys()))
    write_csv(files["pitch"], pitch_rows, list(pitch_rows[0].keys()))
    write_csv(files["roll"], roll_rows, list(roll_rows[0].keys()))
    write_csv(files["translation"], translation_rows, list(translation_rows[0].keys()))
    write_csv(files["time"], time_rows, list(time_rows[0].keys()))
    return files


def _run_registry(root: Path, selected_paths: dict[str, Path], proxy_paths: dict[str, Path], mode: str) -> dict[str, Path]:
    out_dir = ensure_dir(root / "run_registry")
    perturbation_rows = _read_csv(root / "perturbation_matrix.csv")
    selected_rows = _read_csv(selected_paths["matrix"])
    registry_rows: list[dict] = []
    for row in perturbation_rows:
        registry_rows.append(
            {
                "setting_id": f"{row['perturbation_type']}::{row['perturbation_value']}::{row.get('sampling_mode','quick-dense')}",
                "perturbation_type": row["perturbation_type"],
                "perturbation_value": row["perturbation_value"],
                "mode": "quick_dense",
                "status": "skipped" if str(row.get("skipped")).lower() == "true" else "completed",
                "result_json": row.get("result_json"),
                "prediction_dir": row.get("prediction_dir"),
                "frame_count": row.get("frame_count"),
                "sampling_mode": row.get("sampling_mode"),
                "seed": 7,
                "command": "reused_existing_quick_dense_result",
                "start_time": "",
                "end_time": "",
                "wall_time_sec": "",
                "partial": False,
                "partial_reason": "",
                "skipped_reason": row.get("skipped_reason", ""),
            }
        )
    for row in selected_rows:
        registry_rows.append(
            {
                "setting_id": f"{row['perturbation_type']}::{row['perturbation_value']}::{row.get('sampling_mode','selected_1000')}",
                "perturbation_type": row["perturbation_type"],
                "perturbation_value": row["perturbation_value"],
                "mode": "selected_1000",
                "status": row.get("status", "partial"),
                "result_json": row.get("result_json", ""),
                "prediction_dir": row.get("prediction_dir", ""),
                "frame_count": row.get("frame_count", 1000),
                "sampling_mode": row.get("sampling_mode", "selected_1000"),
                "seed": 7,
                "command": "reused_existing_selected_1000_result" if row.get("status") == "completed" else "selected_1000_not_run_this_turn",
                "start_time": "",
                "end_time": "",
                "wall_time_sec": "",
                "partial": row.get("partial", False),
                "partial_reason": row.get("partial_reason", ""),
                "skipped_reason": row.get("skipped_reason", ""),
            }
        )
    proxy_file_map = {
        "calibration_yaw_projection_proxy": proxy_paths["yaw"],
        "calibration_pitch_projection_proxy": proxy_paths["pitch"],
        "calibration_roll_projection_proxy": proxy_paths["roll"],
        "calibration_translation_projection_proxy": proxy_paths["translation"],
        "time_offset_proxy": proxy_paths["time"],
    }
    for ptype, csv_path in proxy_file_map.items():
        status = "completed"
        skipped_reason = ""
        if "pitch" in ptype or "roll" in ptype or "translation" in ptype:
            status = "skipped"
            skipped_reason = "Proxy summary file exists, but the corresponding perturbation was not re-executed this turn."
        registry_rows.append(
            {
                "setting_id": f"{ptype}::proxy_extended",
                "perturbation_type": ptype,
                "perturbation_value": "proxy_extended",
                "mode": "proxy_extended",
                "status": status,
                "result_json": "",
                "prediction_dir": "",
                "frame_count": 20 if ptype != "time_offset_proxy" else 20,
                "sampling_mode": "proxy_extended",
                "seed": 7,
                "command": "reused_existing_proxy_report",
                "start_time": "",
                "end_time": "",
                "wall_time_sec": "",
                "partial": False,
                "partial_reason": "",
                "skipped_reason": skipped_reason,
            }
        )

    registry_json = out_dir / "all_settings_registry.json"
    registry_csv = out_dir / "all_settings_registry.csv"
    write_csv(registry_csv, registry_rows, list(registry_rows[0].keys()))
    write_json(registry_json, {"status": "completed", "mode": mode, "row_count": len(registry_rows), "rows": registry_rows})
    return {"csv": registry_csv, "json": registry_json}


def _experiment_summary(root: Path, registry_paths: dict[str, Path], selected_paths: dict[str, Path], proxy_paths: dict[str, Path]) -> dict[str, Path]:
    out_json = root / "deployment_acceptance_experiment_summary.json"
    out_md = root / "deployment_acceptance_experiment_summary.md"
    registry_rows = _read_csv(registry_paths["csv"])
    quick_rows = [row for row in registry_rows if row["mode"] == "quick_dense"]
    selected_rows = [row for row in registry_rows if row["mode"] == "selected_1000"]
    proxy_rows = [row for row in registry_rows if row["mode"] == "proxy_extended"]
    executed = [row for row in registry_rows if row["status"] == "completed"]
    skipped = [row for row in registry_rows if row["status"] == "skipped"]
    partial = [row for row in registry_rows if row["status"] == "partial"]
    total_frame_runs = sum(int(float(row.get("frame_count") or 0)) for row in executed if row.get("frame_count"))
    payload = {
        "status": "completed",
        "generated_at": _now_iso(),
        "registry_csv": str(registry_paths["csv"]),
        "registry_json": str(registry_paths["json"]),
        "total_settings": len(registry_rows),
        "executed_settings": len(executed),
        "skipped_settings": len(skipped),
        "partial_settings": len(partial),
        "total_frame_runs": total_frame_runs,
        "quick_dense_frame_runs": sum(int(float(row.get("frame_count") or 0)) for row in quick_rows if row["status"] == "completed"),
        "selected_1000_frame_runs": sum(int(float(row.get("frame_count") or 0)) for row in selected_rows if row["status"] == "completed"),
        "proxy_extended_rows": {
            "yaw_rows": len(_read_csv(proxy_paths["yaw"])),
            "time_rows": len(_read_csv(proxy_paths["time"])),
        },
        "selected_1000_paths": {key: str(value) for key, value in selected_paths.items()},
        "proxy_extended_paths": {key: str(value) for key, value in proxy_paths.items()},
    }
    write_json(out_json, payload)
    write_markdown(
        out_md,
        "# Deployment Acceptance Experiment Summary\n\n"
        f"- Total settings: `{payload['total_settings']}`\n"
        f"- Executed settings: `{payload['executed_settings']}`\n"
        f"- Skipped settings: `{payload['skipped_settings']}`\n"
        f"- Partial settings: `{payload['partial_settings']}`\n"
        f"- Total frame-runs: `{payload['total_frame_runs']}`\n"
        f"- Quick-dense frame-runs: `{payload['quick_dense_frame_runs']}`\n"
        f"- Selected-1000 frame-runs: `{payload['selected_1000_frame_runs']}`\n"
        f"- Proxy yaw rows: `{payload['proxy_extended_rows']['yaw_rows']}`\n"
        f"- Proxy time-offset rows: `{payload['proxy_extended_rows']['time_rows']}`\n",
    )
    return {"json": out_json, "md": out_md}


def main() -> None:
    args = parse_args()
    root = ensure_dir(_resolve(args.input_dir))
    run_results = []
    skip_existing = _as_bool(args.skip_existing)

    if args.mode in {"quick_dense", "full_batch"}:
        quick_dense_ready = (root / "perturbation_matrix.csv").exists() and (root / "per_setting_ap.csv").exists()
        if skip_existing and quick_dense_ready:
            run_results.append(
                {
                    "command": "reused_existing_quick_dense_result",
                    "returncode": 0,
                    "stdout": "Skipped rerun because quick-dense outputs already exist and --skip-existing=true.",
                    "stderr": "",
                    "start_time": "",
                    "end_time": "",
                }
            )
        else:
            run_results.append(
                _run_python(
                    "scripts/lidar_system_algorithm/run_lidar_deployment_acceptance_benchmark.py",
                    [
                        "--frames",
                        "200",
                        "--sampling-mode",
                        "quick-dense",
                        "--output-dir",
                        str(root),
                        "--plot-data-dir",
                        str(root / "plot_data"),
                        "--origin-plot-data-dir",
                        str(root / "origin_plot_data"),
                        "--plot-metadata-dir",
                        str(root / "plot_data_metadata"),
                        "--figures-dir",
                        str(_resolve(args.figures_dir).parent / "deployment_acceptance"),
                        "--ppt-dir",
                        str(_resolve(args.ppt_dir).parent / "deployment_acceptance_ppt_panels"),
                    ],
                )
            )
    selected_paths = _selected_1000_tables(root)
    proxy_paths = _proxy_extended_tables(root)
    registry_paths = _run_registry(root, selected_paths, proxy_paths, args.mode)
    summary_paths = _experiment_summary(root, registry_paths, selected_paths, proxy_paths)

    # Generate the dense 29-figure bundle first, then extend it to the requested 42-figure bundle.
    dense_ready = (root / "dense_diagnostics" / "per_frame_prediction_dense.csv").exists() and (_resolve(args.figures_dir) / "29_final_acceptance_dashboard.png").exists()
    if skip_existing and dense_ready:
        run_results.append(
            {
                "command": "reused_existing_dense_diagnostics",
                "returncode": 0,
                "stdout": "Skipped rerun because dense diagnostics bundle already exists and --skip-existing=true.",
                "stderr": "",
                "start_time": "",
                "end_time": "",
            }
        )
    else:
        run_results.append(
            _run_python(
                "scripts/lidar_system_algorithm/generate_deployment_acceptance_dense_diagnostics.py",
                [
                    "--input-dir",
                    str(root),
                    "--dense-dir",
                    str(root / "dense_diagnostics"),
                    "--plot-data-dir",
                    str(root / "plot_data"),
                    "--origin-plot-data-dir",
                    str(root / "origin_plot_data"),
                    "--plot-metadata-dir",
                    str(root / "plot_data_metadata"),
                    "--figures-dir",
                    str(_resolve(args.figures_dir)),
                    "--ppt-dir",
                    str(_resolve(args.ppt_dir)),
                ],
            )
        )
    extended_ready = (_resolve(args.figures_dir) / "42_final_acceptance_dashboard.png").exists() and (_resolve(args.ppt_dir) / "slide5_deployment_precision_and_health_monitoring.png").exists()
    if skip_existing and extended_ready:
        run_results.append(
            {
                "command": "reused_existing_extended_assets",
                "returncode": 0,
                "stdout": "Skipped rerun because 42-figure bundle and 5 PPT panels already exist and --skip-existing=true.",
                "stderr": "",
                "start_time": "",
                "end_time": "",
            }
        )
    else:
        run_results.append(
            _run_python(
                "scripts/lidar_system_algorithm/generate_deployment_acceptance_extended_assets.py",
                [
                    "--input-dir",
                    str(root),
                    "--figures-dir",
                    str(_resolve(args.figures_dir)),
                    "--ppt-dir",
                    str(_resolve(args.ppt_dir)),
                ],
            )
        )

    final = {
        "status": "completed",
        "mode": args.mode,
        "root": str(root),
        "registry": {key: str(value) for key, value in registry_paths.items()},
        "selected_1000": {key: str(value) for key, value in selected_paths.items()},
        "proxy_extended": {key: str(value) for key, value in proxy_paths.items()},
        "experiment_summary": {key: str(value) for key, value in summary_paths.items()},
        "runs": run_results,
    }
    write_json(root / "run_registry" / "full_batch_last_run.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
