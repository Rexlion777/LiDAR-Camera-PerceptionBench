from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.deployment_acceptance import (
    CLASS_NAMES,
    RANGE_BINS,
    aggregate_failure_metrics,
    compute_prediction_health,
    copy_csv_to_origin,
    filter_prediction_dir_by_score,
    jensen_shannon_divergence,
    prediction_subset_stats,
    write_origin_friendly_csv,
    write_setting_json,
)
from runtime.lidar_system_algorithm.high_res_plot_utils import (
    apply_axis_style,
    figure_from_pixels,
    note_figure,
    save_contact_sheet,
    save_figure_triplet,
    write_plot_csv,
    write_plot_metadata,
)
from runtime.lidar_system_algorithm.kitti_io import locate_default_kitti_root
from runtime.lidar_system_algorithm.report_schema import ensure_dir, read_json_or_default, write_csv, write_json, write_markdown


REAL_PERTURBATIONS = {"point_dropout", "range_crop", "postprocess_score_threshold", "deployment_precision"}
QUICK_DENSE_POINT_DROPOUT = ["0.00", "0.05", "0.10", "0.15", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80"]
FULL_DENSE_POINT_DROPOUT = ["0.00", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.35", "0.40", "0.45", "0.50", "0.55", "0.60", "0.65", "0.70", "0.75", "0.80"]
QUICK_DENSE_RANGE_CROP = ["20", "25", "30", "35", "40", "45", "50", "55", "60", "70", "full"]
FULL_DENSE_RANGE_CROP = ["20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "full"]
QUICK_DENSE_SCORE_THRESHOLDS = ["default", "0.00", "0.02", "0.04", "0.06", "0.08", "0.10", "0.12", "0.15", "0.20", "0.30", "0.40", "0.60"]
FULL_DENSE_SCORE_THRESHOLDS = ["default", "0.00", "0.02", "0.04", "0.06", "0.08", "0.10", "0.12", "0.15", "0.18", "0.20", "0.25", "0.30", "0.35", "0.40", "0.50", "0.60"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LiDAR deployment acceptance benchmark.")
    parser.add_argument("--kitti-root", default="data/kitti_object_raw/extracted")
    parser.add_argument("--openpcdet-root", default="external/OpenPCDet")
    parser.add_argument("--cfg-file", default="external/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml")
    parser.add_argument("--ckpt", default="checkpoints/pointpillar_kitti.pth")
    parser.add_argument("--python-exe", default="python")
    parser.add_argument("--split-file", default="external/OpenPCDet/data/kitti/ImageSets/val.txt")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm/deployment_acceptance")
    parser.add_argument("--plot-data-dir", default="reports/lidar_system_algorithm/deployment_acceptance/plot_data")
    parser.add_argument("--origin-plot-data-dir", default="reports/lidar_system_algorithm/deployment_acceptance/origin_plot_data")
    parser.add_argument("--plot-metadata-dir", default="reports/lidar_system_algorithm/deployment_acceptance/plot_data_metadata")
    parser.add_argument("--figures-dir", default="projects/lidar_system_algorithm/figures/deployment_acceptance")
    parser.add_argument("--ppt-dir", default="projects/lidar_system_algorithm/figures/deployment_acceptance_ppt_panels")
    parser.add_argument("--baseline-pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt")
    parser.add_argument("--wrapper-pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt_wrapper_pytorch_core_v2")
    parser.add_argument("--trt-pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt_trt_backbone_head_only")
    parser.add_argument("--sampling-mode", choices=["quick-dense", "full-dense"], default="quick-dense")
    parser.add_argument("--skip-heavy-runs", action="store_true")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[-1]
    return f"/mnt/{drive}{tail}" if drive else resolved.as_posix()


def _split_ids(split_file: Path, frames: int) -> list[str]:
    ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids[:frames] if frames > 0 else ids


def _run(command: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(command, cwd=str(cwd or PROJECT_ROOT), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "command failed")


def _run_wsl_eval(openpcdet_root: Path, label_dir: Path, pred_dir: Path, split_file: Path, output_json: Path) -> dict:
    runner = PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "wsl_kitti_eval_runner.py"
    command = [
        sys.executable,
        str(runner),
        "--openpcdet-root",
        str(openpcdet_root),
        "--label-dir",
        str(label_dir),
        "--pred-dir",
        str(pred_dir),
        "--split-file",
        str(split_file),
        "--output-json",
        str(output_json),
    ]
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    stdout = completed.stdout
    stderr = completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"WSL eval failed\nstdout:\n{stdout}\nstderr:\n{stderr}")
    return json.loads(output_json.read_text(encoding="utf-8"))


def _run_acceptance_helper(
    args: argparse.Namespace,
    split_file: Path,
    perturbation_type: str,
    perturbation_value: str,
    output_json: Path,
    pred_dir: Path,
) -> dict:
    helper = PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "openpcdet_acceptance_eval_helper.py"
    command = [
        "wsl",
        "-d",
        "Ubuntu-24.04",
        "bash",
        "-lc",
        " ".join(
            [
                shlex.quote("python"),
                shlex.quote(_to_wsl_path(helper)),
                "--openpcdet-root",
                shlex.quote(_to_wsl_path(_resolve(args.openpcdet_root))),
                "--cfg-file",
                shlex.quote(_to_wsl_path(_resolve(args.cfg_file))),
                "--ckpt",
                shlex.quote("checkpoints/pointpillar_kitti.pth"),
                "--kitti-root",
                shlex.quote(_to_wsl_path(_resolve(args.kitti_root))),
                "--split-file",
                shlex.quote(_to_wsl_path(split_file)),
                "--output-json",
                shlex.quote(_to_wsl_path(output_json)),
                "--pred-dir",
                shlex.quote(_to_wsl_path(pred_dir)),
                "--max-frames",
                shlex.quote(str(args.frames)),
                "--perturbation-type",
                shlex.quote(perturbation_type),
                "--perturbation-value",
                shlex.quote(str(perturbation_value)),
                "--random-seed",
                shlex.quote(str(args.seed)),
            ]
        ),
    ]
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=False, check=False)
    stdout = completed.stdout.decode("utf-8", errors="ignore") if isinstance(completed.stdout, bytes) else str(completed.stdout)
    stderr = completed.stderr.decode("utf-8", errors="ignore") if isinstance(completed.stderr, bytes) else str(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(stderr.strip() or stdout.strip() or "WSL acceptance helper failed")
    return json.loads(output_json.read_text(encoding="utf-8"))


def _safe_mean_ap(result_dict: dict) -> float | None:
    keys = ["Car_3d/moderate_R40", "Pedestrian_3d/moderate_R40", "Cyclist_3d/moderate_R40"]
    values = [result_dict.get(key) for key in keys if result_dict.get(key) is not None]
    return float(np.mean(values)) if values else None


def _sampling_config(sampling_mode: str) -> dict[str, object]:
    if sampling_mode == "full-dense":
        return {
            "point_dropout_values": FULL_DENSE_POINT_DROPOUT,
            "range_crop_values": FULL_DENSE_RANGE_CROP,
            "score_threshold_values": FULL_DENSE_SCORE_THRESHOLDS,
            "reduced_sampling": False,
            "reduced_sampling_reason": None,
        }
    return {
        "point_dropout_values": QUICK_DENSE_POINT_DROPOUT,
        "range_crop_values": QUICK_DENSE_RANGE_CROP,
        "score_threshold_values": QUICK_DENSE_SCORE_THRESHOLDS,
        "reduced_sampling": True,
        "reduced_sampling_reason": "quick-dense mode uses a reduced but still dense sweep on a 200-frame slice to control runtime while keeping >=11 settings on primary trend curves.",
    }


def _plot_line_multi(base_path: Path, title: str, xlabel: str, ylabel: str, x_values: list, series: dict[str, list[float | None]], width_px: int, height_px: int, dpi: int, csv_path: Path, origin_dir: Path, metadata_path: Path, source_report: str, units: str, safe_caption: str, forbidden_claims: list[str]) -> None:
    fig = figure_from_pixels(width_px, height_px, dpi=dpi)
    ax = fig.add_subplot(111)
    numeric_x = np.arange(len(x_values), dtype=np.float64) if any(isinstance(x, str) for x in x_values) else np.asarray(x_values, dtype=np.float64)
    for label, values in series.items():
        y = np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float64)
        ax.plot(numeric_x, y, marker="o", linewidth=2.5, label=label)
    apply_axis_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    if any(isinstance(x, str) for x in x_values):
        ax.set_xticks(numeric_x)
        ax.set_xticklabels([str(x) for x in x_values], rotation=0)
    ax.legend(fontsize=12)
    saved = save_figure_triplet(fig, base_path, dpi=dpi)
    rows = []
    fieldnames = [xlabel] + list(series.keys())
    for idx, x_value in enumerate(x_values):
        row = {xlabel: x_value}
        for label, values in series.items():
            row[label] = values[idx]
        rows.append(row)
    csv_file, origin_csv = write_plot_csv(csv_path, rows, fieldnames, origin_dir)
    write_plot_metadata(
        metadata_path,
        {
            "figure_name": base_path.name,
            "figure_path_png": saved["png"],
            "figure_path_svg": saved["svg"],
            "figure_path_pdf": saved["pdf"],
            "source_csv": str(csv_file),
            "origin_csv": str(origin_csv),
            "source_report": source_report,
            "x_axis": xlabel,
            "y_axis": ylabel,
            "units": units,
            "plotted_columns": list(series.keys()),
            "skipped": False,
            "skipped_reason": None,
            "safe_caption": safe_caption,
            "forbidden_claims": forbidden_claims,
        },
    )


def _plot_bar(base_path: Path, title: str, xlabel: str, ylabel: str, categories: list[str], values: list[float | None], width_px: int, height_px: int, dpi: int, csv_path: Path, origin_dir: Path, metadata_path: Path, source_report: str, units: str, safe_caption: str, forbidden_claims: list[str]) -> None:
    fig = figure_from_pixels(width_px, height_px, dpi=dpi)
    ax = fig.add_subplot(111)
    x = np.arange(len(categories))
    y = np.asarray([0.0 if value is None else float(value) for value in values], dtype=np.float64)
    ax.bar(x, y, color="#3a86ff")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20)
    apply_axis_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    saved = save_figure_triplet(fig, base_path, dpi=dpi)
    rows = [{xlabel: categories[idx], ylabel: values[idx]} for idx in range(len(categories))]
    csv_file, origin_csv = write_plot_csv(csv_path, rows, [xlabel, ylabel], origin_dir)
    write_plot_metadata(
        metadata_path,
        {
            "figure_name": base_path.name,
            "figure_path_png": saved["png"],
            "figure_path_svg": saved["svg"],
            "figure_path_pdf": saved["pdf"],
            "source_csv": str(csv_file),
            "origin_csv": str(origin_csv),
            "source_report": source_report,
            "x_axis": xlabel,
            "y_axis": ylabel,
            "units": units,
            "plotted_columns": [ylabel],
            "skipped": False,
            "skipped_reason": None,
            "safe_caption": safe_caption,
            "forbidden_claims": forbidden_claims,
        },
    )


def _plot_scatter(base_path: Path, title: str, xlabel: str, ylabel: str, rows: list[dict], x_key: str, y_key: str, color_key: str | None, width_px: int, height_px: int, dpi: int, csv_path: Path, origin_dir: Path, metadata_path: Path, source_report: str, units: str, safe_caption: str, forbidden_claims: list[str]) -> None:
    fig = figure_from_pixels(width_px, height_px, dpi=dpi)
    ax = fig.add_subplot(111)
    xs = np.asarray([float(row[x_key]) for row in rows if row.get(x_key) is not None and row.get(y_key) is not None], dtype=np.float64)
    ys = np.asarray([float(row[y_key]) for row in rows if row.get(x_key) is not None and row.get(y_key) is not None], dtype=np.float64)
    if color_key:
        colors = [row.get(color_key, "steelblue") for row in rows if row.get(x_key) is not None and row.get(y_key) is not None]
        ax.scatter(xs, ys, c=colors, alpha=0.75, s=36)
    else:
        ax.scatter(xs, ys, alpha=0.75, s=36, color="#ff006e")
    apply_axis_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    saved = save_figure_triplet(fig, base_path, dpi=dpi)
    fieldnames = list(rows[0].keys()) if rows else [x_key, y_key]
    csv_file, origin_csv = write_plot_csv(csv_path, rows, fieldnames, origin_dir)
    write_plot_metadata(
        metadata_path,
        {
            "figure_name": base_path.name,
            "figure_path_png": saved["png"],
            "figure_path_svg": saved["svg"],
            "figure_path_pdf": saved["pdf"],
            "source_csv": str(csv_file),
            "origin_csv": str(origin_csv),
            "source_report": source_report,
            "x_axis": xlabel,
            "y_axis": ylabel,
            "units": units,
            "plotted_columns": [x_key, y_key] + ([color_key] if color_key else []),
            "skipped": False,
            "skipped_reason": None,
            "safe_caption": safe_caption,
            "forbidden_claims": forbidden_claims,
        },
    )


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _save_panel_triplet(image_paths: list[Path], output_base: Path, title: str) -> None:
    from PIL import Image, ImageOps, ImageDraw

    width_px = 3840
    height_px = 2160
    columns = 2
    rows = int(math.ceil(len(image_paths) / columns))
    panel = Image.new("RGB", (width_px, height_px), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    draw.text((60, 40), title, fill=(20, 20, 20))
    cell_w = width_px // columns
    cell_h = (height_px - 160) // max(rows, 1)
    y_offset = 120
    for index, path in enumerate(image_paths):
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, (cell_w - 40, cell_h - 40))
        x = (index % columns) * cell_w + 20
        y = (index // columns) * cell_h + y_offset
        panel.paste(image, (x, y))
    fig = figure_from_pixels(width_px, height_px, dpi=300)
    ax = fig.add_subplot(111)
    ax.imshow(np.asarray(panel))
    ax.axis("off")
    save_figure_triplet(fig, output_base, dpi=300)


def main() -> None:
    args = parse_args()
    sampling_cfg = _sampling_config(args.sampling_mode)
    kitti_root = _resolve(args.kitti_root) if args.kitti_root else locate_default_kitti_root()
    split_file = _resolve(args.split_file)
    openpcdet_root = _resolve(args.openpcdet_root)
    output_dir = ensure_dir(_resolve(args.output_dir))
    plot_data_dir = ensure_dir(_resolve(args.plot_data_dir))
    origin_dir = ensure_dir(_resolve(args.origin_plot_data_dir))
    plot_metadata_dir = ensure_dir(_resolve(args.plot_metadata_dir))
    figures_dir = ensure_dir(_resolve(args.figures_dir))
    ppt_dir = ensure_dir(_resolve(args.ppt_dir))
    settings_dir = ensure_dir(output_dir / "settings")
    pred_root = ensure_dir(output_dir / "prediction_dirs")
    label_dir = kitti_root / "training" / "label_2"
    sample_ids = _split_ids(split_file, args.frames)
    split_scope_file = output_dir / f"acceptance_eval_split_{args.frames}.txt"
    split_scope_file.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")

    perturbation_matrix_rows: list[dict] = []
    per_setting_ap_rows: list[dict] = []
    per_setting_health_rows: list[dict] = []
    per_setting_failure_range_rows: list[dict] = []
    per_setting_failure_class_rows: list[dict] = []
    runtime_health_rows: list[dict] = []
    per_frame_latency_rows: list[dict] = []
    per_frame_prediction_rows: list[dict] = []
    per_frame_diff_rows: list[dict] = []
    setting_payloads: dict[tuple[str, str], dict] = {}

    original_report = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_final_report.json", {})
    wrapper_report = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/wrapper_pytorch_core_eval_v2.json", {})
    trt_report = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_eval.json", {})
    trt_diff_report = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_diff.json", {})
    online_report = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_online_latency.json", {})
    robustness_report = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/calibration_sync_robustness.json", {})
    tracking_report = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tracking_optimized_summary.json", {})

    baseline_eval_json = settings_dir / "deployment_precision_original_baseline_eval.json"
    baseline_pred_dir = _resolve(args.baseline_pred_dir)
    baseline_eval = _run_wsl_eval(openpcdet_root, label_dir, baseline_pred_dir, split_scope_file, baseline_eval_json)
    baseline_stats = prediction_subset_stats(baseline_pred_dir, sample_ids)
    baseline_payload = {
        "status": "completed",
        "perturbation_type": "deployment_precision",
        "perturbation_value": "openpcdet_original",
        "frame_count": len(sample_ids),
        "eval_scope": f"{args.frames}-frame-slice",
        "is_full_val": False,
        "sampling_mode": args.sampling_mode,
        "reduced_sampling": bool(sampling_cfg["reduced_sampling"]),
        "reduced_sampling_reason": sampling_cfg["reduced_sampling_reason"],
        "prediction_dir": str(baseline_pred_dir),
        "official_result_dict": baseline_eval.get("official_result_dict", {}),
        "prediction_stats": baseline_stats,
    }
    write_setting_json(settings_dir / "deployment_precision_openpcdet_original.json", baseline_payload)
    setting_payloads[("deployment_precision", "openpcdet_original")] = baseline_payload

    experiments = [
        ("point_dropout", list(sampling_cfg["point_dropout_values"])),
        ("range_crop", list(sampling_cfg["range_crop_values"])),
    ]
    if not args.skip_heavy_runs:
        for perturbation_type, values in experiments:
            for value in values:
                key = (perturbation_type, value)
                pred_dir = ensure_dir(pred_root / f"{perturbation_type}_{str(value).replace('.', 'p')}")
                result_json = settings_dir / f"{perturbation_type}_{str(value).replace('.', 'p')}.json"
                raw_json = settings_dir / f"{perturbation_type}_{str(value).replace('.', 'p')}_raw.json"
                if result_json.exists():
                    payload = json.loads(result_json.read_text(encoding="utf-8"))
                else:
                    helper_payload = _run_acceptance_helper(args, split_scope_file, perturbation_type, value, raw_json, pred_dir)
                    eval_json = settings_dir / f"{perturbation_type}_{str(value).replace('.', 'p')}_eval.json"
                    eval_payload = _run_wsl_eval(openpcdet_root, label_dir, pred_dir, split_scope_file, eval_json)
                    pred_stats = prediction_subset_stats(pred_dir, sample_ids)
                    payload = {
                        "status": "completed",
                        "perturbation_type": perturbation_type,
                        "perturbation_value": value,
                        "frame_count": len(sample_ids),
                        "eval_scope": f"{args.frames}-frame-slice",
                        "is_full_val": False,
                        "sampling_mode": args.sampling_mode,
                        "reduced_sampling": bool(sampling_cfg["reduced_sampling"]),
                        "reduced_sampling_reason": sampling_cfg["reduced_sampling_reason"],
                        "prediction_dir": str(pred_dir),
                        "helper_output_json": str(raw_json),
                        "official_result_dict": eval_payload.get("official_result_dict", {}),
                        "prediction_stats": pred_stats,
                        "per_frame": helper_payload.get("per_frame", []),
                    }
                    write_setting_json(result_json, payload)
                setting_payloads[key] = payload

    for threshold in list(sampling_cfg["score_threshold_values"]):
        key = ("postprocess_score_threshold", threshold)
        result_json = settings_dir / f"postprocess_score_threshold_{str(threshold).replace('.', 'p')}.json"
        if result_json.exists():
            setting_payloads[key] = json.loads(result_json.read_text(encoding="utf-8"))
            continue
        if threshold == "default":
            filtered_pred_dir = pred_root / "score_threshold_default"
            ensure_dir(filtered_pred_dir)
            filter_prediction_dir_by_score(baseline_pred_dir, filtered_pred_dir, sample_ids, 0.1)
            threshold_value = 0.1
        else:
            threshold_value = float(threshold)
            filtered_pred_dir = pred_root / f"score_threshold_{str(threshold).replace('.', 'p')}"
            ensure_dir(filtered_pred_dir)
            filter_prediction_dir_by_score(baseline_pred_dir, filtered_pred_dir, sample_ids, threshold_value)
        eval_json = settings_dir / f"postprocess_score_threshold_{str(threshold).replace('.', 'p')}_eval.json"
        eval_payload = _run_wsl_eval(openpcdet_root, label_dir, filtered_pred_dir, split_scope_file, eval_json)
        pred_stats = prediction_subset_stats(filtered_pred_dir, sample_ids)
        payload = {
            "status": "completed",
            "perturbation_type": "postprocess_score_threshold",
            "perturbation_value": threshold,
            "frame_count": len(sample_ids),
            "eval_scope": f"{args.frames}-frame-slice",
            "is_full_val": False,
            "sampling_mode": args.sampling_mode,
            "reduced_sampling": bool(sampling_cfg["reduced_sampling"]),
            "reduced_sampling_reason": sampling_cfg["reduced_sampling_reason"],
            "prediction_dir": str(filtered_pred_dir),
            "official_result_dict": eval_payload.get("official_result_dict", {}),
            "prediction_stats": pred_stats,
        }
        write_setting_json(result_json, payload)
        setting_payloads[key] = payload

    wrapper_stats = prediction_subset_stats(_resolve(args.wrapper_pred_dir), sample_ids)
    wrapper_payload = {
        "status": "completed",
        "perturbation_type": "deployment_precision",
        "perturbation_value": "wrapper_pytorch",
        "frame_count": len(sample_ids),
        "eval_scope": f"{args.frames}-frame-slice",
        "is_full_val": False,
        "sampling_mode": args.sampling_mode,
        "reduced_sampling": bool(sampling_cfg["reduced_sampling"]),
        "reduced_sampling_reason": sampling_cfg["reduced_sampling_reason"],
        "prediction_dir": str(_resolve(args.wrapper_pred_dir)),
        "official_result_dict": wrapper_report.get("official_result_dict", {}),
        "prediction_stats": wrapper_stats,
    }
    write_setting_json(settings_dir / "deployment_precision_wrapper_pytorch.json", wrapper_payload)
    setting_payloads[("deployment_precision", "wrapper_pytorch")] = wrapper_payload

    trt_stats = prediction_subset_stats(_resolve(args.trt_pred_dir), sample_ids)
    trt_payload = {
        "status": "completed",
        "perturbation_type": "deployment_precision",
        "perturbation_value": "trt_backbone_head_only",
        "frame_count": len(sample_ids),
        "eval_scope": f"{args.frames}-frame-slice",
        "is_full_val": False,
        "sampling_mode": args.sampling_mode,
        "reduced_sampling": bool(sampling_cfg["reduced_sampling"]),
        "reduced_sampling_reason": sampling_cfg["reduced_sampling_reason"],
        "prediction_dir": str(_resolve(args.trt_pred_dir)),
        "official_result_dict": trt_report.get("official_result_dict", {}),
        "prediction_stats": trt_stats,
    }
    write_setting_json(settings_dir / "deployment_precision_trt_backbone_head_only.json", trt_payload)
    setting_payloads[("deployment_precision", "trt_backbone_head_only")] = trt_payload

    skipped_rows = [
        ("intensity_dropout_or_noise", "skipped", "KITTI intensity perturbation was not injected this turn to avoid mixing sensor-noise assumptions into the detector acceptance run."),
        ("calibration_translation_perturbation", "skipped", "Detector-input translation perturbation path is not safely implemented; only projection-level robustness exists and translation rows are unavailable."),
        ("postprocess_nms_threshold", "skipped", "OpenPCDet NMS threshold was not modified this turn to avoid threshold-hack ambiguity in acceptance claims."),
    ]
    for perturbation_type, value, reason in skipped_rows:
        payload = {
            "status": "skipped",
            "perturbation_type": perturbation_type,
            "perturbation_value": value,
            "frame_count": len(sample_ids),
            "eval_scope": f"{args.frames}-frame-slice",
            "is_full_val": False,
            "sampling_mode": args.sampling_mode,
            "reduced_sampling": True,
            "reduced_sampling_reason": reason,
            "prediction_dir": "",
            "skipped_reason": reason,
        }
        write_setting_json(settings_dir / f"{perturbation_type}.json", payload)
        setting_payloads[(perturbation_type, value)] = payload

    baseline_ap = baseline_eval.get("official_result_dict", {})
    baseline_mean_ap = _safe_mean_ap(baseline_ap)

    all_metrics = []
    for (perturbation_type, perturbation_value), payload in sorted(setting_payloads.items()):
        skipped = payload.get("status") != "completed"
        perturbation_matrix_rows.append(
            {
                "perturbation_type": perturbation_type,
                "perturbation_value": perturbation_value,
                "frame_count": payload.get("frame_count"),
                "eval_scope": payload.get("eval_scope"),
                "sampling_mode": payload.get("sampling_mode", args.sampling_mode),
                "model_path": args.ckpt if perturbation_type != "deployment_precision" or perturbation_value == "openpcdet_original" else perturbation_value,
                "config_path": args.cfg_file,
                "prediction_dir": payload.get("prediction_dir"),
                "result_json": str(settings_dir / f"{perturbation_type}_{str(perturbation_value).replace('.', 'p')}.json"),
                "skipped": skipped,
                "skipped_reason": payload.get("skipped_reason"),
                "reduced_sampling": payload.get("reduced_sampling", False),
                "reduced_sampling_reason": payload.get("reduced_sampling_reason"),
            }
        )
        if skipped:
            continue
        result_dict = payload.get("official_result_dict", {})
        mean_ap = _safe_mean_ap(result_dict)
        ap_row = {
            "perturbation_type": perturbation_type,
            "perturbation_value": perturbation_value,
            "frame_count": payload.get("frame_count"),
            "sampling_mode": payload.get("sampling_mode", args.sampling_mode),
            "car_ap_3d_moderate": result_dict.get("Car_3d/moderate_R40"),
            "ped_ap_3d_moderate": result_dict.get("Pedestrian_3d/moderate_R40"),
            "cyc_ap_3d_moderate": result_dict.get("Cyclist_3d/moderate_R40"),
            "mean_ap_3d_moderate": mean_ap,
            "delta_car": (result_dict.get("Car_3d/moderate_R40") - baseline_ap.get("Car_3d/moderate_R40")) if result_dict.get("Car_3d/moderate_R40") is not None else None,
            "delta_ped": (result_dict.get("Pedestrian_3d/moderate_R40") - baseline_ap.get("Pedestrian_3d/moderate_R40")) if result_dict.get("Pedestrian_3d/moderate_R40") is not None else None,
            "delta_cyc": (result_dict.get("Cyclist_3d/moderate_R40") - baseline_ap.get("Cyclist_3d/moderate_R40")) if result_dict.get("Cyclist_3d/moderate_R40") is not None else None,
            "delta_mean": (mean_ap - baseline_mean_ap) if mean_ap is not None and baseline_mean_ap is not None else None,
        }
        per_setting_ap_rows.append(ap_row)

        pred_stats = payload.get("prediction_stats", {})
        health = compute_prediction_health(pred_stats, baseline_stats)
        class_counts = pred_stats.get("per_class_box_count", {})
        health_row = {
            "perturbation_type": perturbation_type,
            "perturbation_value": perturbation_value,
            "empty_prediction_file_count": pred_stats.get("empty_prediction_file_count", 0),
            "invalid_geometry_count": pred_stats.get("invalid_geometry_count", 0),
            "total_box_count": pred_stats.get("total_box_count", 0),
            "car_box_count": class_counts.get("Car", 0),
            "ped_box_count": class_counts.get("Pedestrian", 0),
            "cyc_box_count": class_counts.get("Cyclist", 0),
            "score_mean": pred_stats.get("score_summary", {}).get("mean"),
            "score_p50": pred_stats.get("score_summary", {}).get("p50"),
            "score_p95": pred_stats.get("score_summary", {}).get("p95"),
            "prediction_count_drift": health["prediction_count_drift"],
            "score_distribution_drift": health["score_distribution_drift"],
            "class_distribution_drift": health["class_distribution_drift"],
            "range_distribution_drift": health["range_distribution_drift"],
            "nms_suppression_ratio": None,
        }
        per_setting_health_rows.append(health_row)

        failure = aggregate_failure_metrics(label_dir, Path(payload["prediction_dir"]), sample_ids)
        for row in failure.by_range:
            per_setting_failure_range_rows.append({"perturbation_type": perturbation_type, "perturbation_value": perturbation_value, **row})
        for row in failure.by_class:
            per_setting_failure_class_rows.append({"perturbation_type": perturbation_type, "perturbation_value": perturbation_value, **row})

        temporal_consistency_error = None
        tracking_association_residual = None
        latency_spike_rate = None
        if perturbation_type == "time_offset_proxy":
            temporal_consistency_error = float(perturbation_value)
        if perturbation_type == "deployment_precision" and perturbation_value == "openpcdet_original":
            tracking_association_residual = tracking_report.get("average_association_latency_ms")
        if payload.get("per_frame"):
            latencies = [float(row.get("pytorch_core_ms", 0.0)) for row in payload["per_frame"] if row.get("pytorch_core_ms") is not None]
            if latencies:
                p95 = float(np.percentile(latencies, 95.0))
                latency_spike_rate = float(np.mean(np.asarray(latencies) > p95))

        health_weights = {
            "prediction_count_drift": 0.20,
            "score_distribution_drift": 0.15,
            "class_distribution_drift": 0.15,
            "range_distribution_drift": 0.15,
            "invalid_geometry_rate": 0.10,
            "empty_prediction_rate": 0.10,
            "temporal_consistency_error": 0.10 if temporal_consistency_error is not None else 0.0,
            "latency_spike_rate": 0.05 if latency_spike_rate is not None else 0.0,
        }
        health_risk = (
            health_weights["prediction_count_drift"] * health["prediction_count_drift"]
            + health_weights["score_distribution_drift"] * health["score_distribution_drift"]
            + health_weights["class_distribution_drift"] * health["class_distribution_drift"]
            + health_weights["range_distribution_drift"] * health["range_distribution_drift"]
            + health_weights["invalid_geometry_rate"] * health["invalid_geometry_rate"]
            + health_weights["empty_prediction_rate"] * health["empty_prediction_rate"]
            + health_weights["temporal_consistency_error"] * (temporal_consistency_error or 0.0)
            + health_weights["latency_spike_rate"] * (latency_spike_rate or 0.0)
        )
        runtime_row = {
            "perturbation_type": perturbation_type,
            "perturbation_value": perturbation_value,
            "label_free_health_risk": health_risk,
            "prediction_count_drift": health["prediction_count_drift"],
            "score_distribution_drift": health["score_distribution_drift"],
            "class_distribution_drift": health["class_distribution_drift"],
            "range_distribution_drift": health["range_distribution_drift"],
            "invalid_geometry_rate": health["invalid_geometry_rate"],
            "empty_prediction_rate": health["empty_prediction_rate"],
            "temporal_consistency_error": temporal_consistency_error,
            "tracking_association_residual": tracking_association_residual,
            "latency_spike_rate": latency_spike_rate,
            "mean_ap_drop": abs(ap_row["delta_mean"]) if ap_row["delta_mean"] is not None else None,
        }
        runtime_health_rows.append(runtime_row)
        all_metrics.append(runtime_row)

        for frame_row in payload.get("per_frame", []):
            per_frame_latency_rows.append(
                {
                    "frame_id": frame_row.get("frame_id"),
                    "perturbation_type": perturbation_type,
                    "perturbation_value": perturbation_value,
                    "pytorch_core_ms": frame_row.get("pytorch_core_ms"),
                    "trt_core_ms": None,
                    "core_speedup": None,
                    "pytorch_online_total_ms": None,
                    "trt_online_total_ms": None,
                    "online_speedup": None,
                    "preprocess_ms": None,
                    "vfe_ms": None,
                    "scatter_ms": None,
                    "backbone_head_ms": None,
                    "postprocess_ms": None,
                    "tracking_ms": None,
                    "visualization_excluded": True,
                }
            )
            per_frame_prediction_rows.append(
                {
                    "frame_id": frame_row.get("frame_id"),
                    "perturbation_type": perturbation_type,
                    "perturbation_value": perturbation_value,
                    "point_count": frame_row.get("point_count"),
                    "predicted_box_count": frame_row.get("predicted_box_count"),
                    "car_count": frame_row.get("car_count"),
                    "ped_count": frame_row.get("ped_count"),
                    "cyc_count": frame_row.get("cyc_count"),
                    "score_mean": frame_row.get("score_mean"),
                    "score_p50": frame_row.get("score_p50"),
                    "score_p95": frame_row.get("score_p95"),
                    "max_range": frame_row.get("max_range"),
                    "mean_range": frame_row.get("mean_range"),
                }
            )

    trt_online_rows = online_report.get("rows_preview", [])
    for row in trt_online_rows:
        per_frame_latency_rows.append(
            {
                "frame_id": row.get("frame_id"),
                "perturbation_type": "deployment_precision",
                "perturbation_value": "trt_backbone_head_only",
                "pytorch_core_ms": row.get("pytorch_backbone_head_ms"),
                "trt_core_ms": row.get("trt_backbone_head_ms"),
                "core_speedup": (row.get("pytorch_backbone_head_ms") / row.get("trt_backbone_head_ms")) if row.get("trt_backbone_head_ms") else None,
                "pytorch_online_total_ms": row.get("pytorch_online_total_ms"),
                "trt_online_total_ms": row.get("trt_online_total_ms"),
                "online_speedup": (row.get("pytorch_online_total_ms") / row.get("trt_online_total_ms")) if row.get("trt_online_total_ms") else None,
                "preprocess_ms": row.get("preprocess_voxelization_ms"),
                "vfe_ms": row.get("vfe_ms"),
                "scatter_ms": row.get("scatter_ms"),
                "backbone_head_ms": row.get("trt_backbone_head_ms"),
                "postprocess_ms": row.get("trt_postprocess_nms_ms"),
                "tracking_ms": row.get("tracking_ms"),
                "visualization_excluded": True,
            }
        )

    diff_summary = trt_diff_report.get("summary", {})
    for preview_row in trt_diff_report.get("frame_rows_preview", []):
        per_frame_diff_rows.append(
            {
                "frame_id": preview_row.get("frame_id"),
                "topk_center_diff_mean": preview_row.get("topk_center_diff_mean"),
                "topk_center_diff_p95": preview_row.get("topk_center_diff_p95"),
                "topk_score_diff_mean": preview_row.get("topk_score_diff_mean"),
                "rotation_y_diff_mean": preview_row.get("topk_rotation_y_diff_mean"),
                "box_count_diff": preview_row.get("box_count_diff"),
                "class_count_diff": preview_row.get("class_count_diff"),
                "is_outlier": preview_row.get("frame_id") in set(diff_summary.get("outlier_frame_ids", [])),
            }
        )

    def _corr(metric_name: str, class_key: str | None = None) -> tuple[float | None, int]:
        xs = []
        ys = []
        for row, ap_row in zip(runtime_health_rows, per_setting_ap_rows):
            x = row.get(metric_name)
            if class_key is None:
                y = row.get("mean_ap_drop")
            else:
                y = abs(ap_row.get(class_key)) if ap_row.get(class_key) is not None else None
            if x is None or y is None:
                continue
            xs.append(float(x))
            ys.append(float(y))
        if len(xs) < 3 or np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
            return None, len(xs)
        return float(np.corrcoef(xs, ys)[0, 1]), len(xs)

    health_metric_rows = []
    for metric_name in [
        "prediction_count_drift",
        "score_distribution_drift",
        "class_distribution_drift",
        "range_distribution_drift",
        "invalid_geometry_rate",
        "empty_prediction_rate",
        "temporal_consistency_error",
        "latency_spike_rate",
        "label_free_health_risk",
    ]:
        mean_corr, n_points = _corr(metric_name)
        car_corr, _ = _corr(metric_name, "delta_car")
        ped_corr, _ = _corr(metric_name, "delta_ped")
        cyc_corr, _ = _corr(metric_name, "delta_cyc")
        health_metric_rows.append(
            {
                "metric_name": metric_name,
                "correlation_with_mean_ap_drop": mean_corr,
                "correlation_with_car_ap_drop": car_corr,
                "correlation_with_ped_ap_drop": ped_corr,
                "correlation_with_cyc_ap_drop": cyc_corr,
                "n_points_used": n_points,
            }
        )

    perturbation_matrix_csv = output_dir / "perturbation_matrix.csv"
    per_setting_ap_csv = output_dir / "per_setting_ap.csv"
    per_setting_health_csv = output_dir / "per_setting_prediction_health.csv"
    per_setting_failure_range_csv = output_dir / "per_setting_failure_by_range.csv"
    per_setting_failure_class_csv = output_dir / "per_setting_failure_by_class.csv"
    runtime_health_csv = output_dir / "runtime_health_metrics.csv"
    health_metric_correlation_csv = output_dir / "health_metric_correlation.csv"
    per_frame_latency_csv = output_dir / "per_frame_latency.csv"
    per_frame_prediction_csv = output_dir / "per_frame_prediction_stats.csv"
    per_frame_diff_csv = output_dir / "per_frame_diff_stats.csv"

    write_csv(perturbation_matrix_csv, perturbation_matrix_rows, list(perturbation_matrix_rows[0].keys()))
    write_csv(per_setting_ap_csv, per_setting_ap_rows, list(per_setting_ap_rows[0].keys()))
    write_csv(per_setting_health_csv, per_setting_health_rows, list(per_setting_health_rows[0].keys()))
    write_csv(per_setting_failure_range_csv, per_setting_failure_range_rows, list(per_setting_failure_range_rows[0].keys()))
    write_csv(per_setting_failure_class_csv, per_setting_failure_class_rows, list(per_setting_failure_class_rows[0].keys()))
    write_csv(runtime_health_csv, runtime_health_rows, list(runtime_health_rows[0].keys()))
    write_csv(health_metric_correlation_csv, health_metric_rows, list(health_metric_rows[0].keys()))
    write_csv(per_frame_latency_csv, per_frame_latency_rows, list(per_frame_latency_rows[0].keys()))
    write_csv(per_frame_prediction_csv, per_frame_prediction_rows, list(per_frame_prediction_rows[0].keys()))
    write_csv(per_frame_diff_csv, per_frame_diff_rows, list(per_frame_diff_rows[0].keys() if per_frame_diff_rows else ["frame_id"]))

    for csv_path in [
        perturbation_matrix_csv,
        per_setting_ap_csv,
        per_setting_health_csv,
        per_setting_failure_range_csv,
        per_setting_failure_class_csv,
        runtime_health_csv,
        health_metric_correlation_csv,
        per_frame_latency_csv,
        per_frame_prediction_csv,
        per_frame_diff_csv,
    ]:
        copy_csv_to_origin(csv_path, origin_dir)

    # High-res figures
    forbidden = [
        "new 3D detection model",
        "full TensorRT detector",
        "full-val if only 1000-frame slice",
        "end-to-end latency when referring to core latency",
    ]
    figure_paths: list[Path] = []
    def add_note(fig_name: str, title: str, lines: list[str], width_px: int = 3840, height_px: int = 2160):
        base = figures_dir / fig_name
        saved = note_figure(base, title, lines, width_px, height_px)
        write_plot_metadata(
            plot_metadata_dir / f"{fig_name}.json",
            {
                "figure_name": fig_name,
                "figure_path_png": saved["png"],
                "figure_path_svg": saved["svg"],
                "figure_path_pdf": saved["pdf"],
                "source_csv": None,
                "source_report": str(output_dir / "deployment_acceptance_final_report.md"),
                "x_axis": None,
                "y_axis": None,
                "units": None,
                "plotted_columns": [],
                "skipped": False,
                "skipped_reason": None,
                "safe_caption": title,
                "forbidden_claims": forbidden,
            },
        )
        figure_paths.append(Path(saved["png"]))

    add_note("01_application_scenario_deployment_acceptance", "Application Scenario", [
        "Offline AP normal does not imply deployment reliability.",
        "Target scenario: unmanned platforms / robotics / intelligent hardware LiDAR detection acceptance.",
        "This project is an acceptance and abnormal-attribution toolchain, not a new detector.",
    ])
    add_note("02_perturbation_taxonomy", "Perturbation Taxonomy", [
        "Point cloud quality: point dropout, range crop.",
        "Calibration / timing: projection-level yaw sensitivity, adjacent-frame time-offset proxy.",
        "Postprocess / deployment precision: score-threshold drift, wrapper parity, TRT parity, latency drift.",
    ])
    add_note("03_acceptance_benchmark_pipeline", "Acceptance Benchmark Pipeline", [
        "Research checkpoint / runtime variant -> perturbation injection -> official eval / matcher / health metrics.",
        "Outputs: AP sensitivity, failure by range/class, distribution drift, latency traces, label-free health risk.",
        "All x/y figures export raw CSV, Origin-friendly CSV, PNG/SVG/PDF, and metadata JSON.",
    ])

    def _ap_rows_for(ptype: str):
        rows = [row for row in per_setting_ap_rows if row["perturbation_type"] == ptype]
        rows.sort(key=lambda row: str(row["perturbation_value"]))
        return rows

    dropout_rows = sorted(_ap_rows_for("point_dropout"), key=lambda row: float(row["perturbation_value"]))
    if dropout_rows:
        _plot_line_multi(
            figures_dir / "04_ap_vs_point_dropout",
            "AP vs Point Dropout",
            "dropout_ratio",
            "moderate 3D AP_R40",
            [float(row["perturbation_value"]) for row in dropout_rows],
            {
                "car_ap": [row["car_ap_3d_moderate"] for row in dropout_rows],
                "ped_ap": [row["ped_ap_3d_moderate"] for row in dropout_rows],
                "cyc_ap": [row["cyc_ap_3d_moderate"] for row in dropout_rows],
                "mean_ap": [row["mean_ap_3d_moderate"] for row in dropout_rows],
            },
            3200,
            2200,
            600,
            plot_data_dir / "04_ap_vs_point_dropout.csv",
            origin_dir,
            plot_metadata_dir / "04_ap_vs_point_dropout.json",
            str(per_setting_ap_csv),
            "AP_R40",
            "Moderate 3D AP sensitivity under random point dropout on a 200-frame slice.",
            forbidden,
        )
        figure_paths.append(figures_dir / "04_ap_vs_point_dropout.png")

    range_rows = sorted(_ap_rows_for("range_crop"), key=lambda row: 999.0 if row["perturbation_value"] == "full" else float(row["perturbation_value"]))
    if range_rows:
        _plot_line_multi(
            figures_dir / "05_ap_vs_range_crop",
            "AP vs Range Crop",
            "max_range_m",
            "moderate 3D AP_R40",
            [row["perturbation_value"] for row in range_rows],
            {
                "car_ap": [row["car_ap_3d_moderate"] for row in range_rows],
                "ped_ap": [row["ped_ap_3d_moderate"] for row in range_rows],
                "cyc_ap": [row["cyc_ap_3d_moderate"] for row in range_rows],
                "mean_ap": [row["mean_ap_3d_moderate"] for row in range_rows],
            },
            3200,
            2200,
            600,
            plot_data_dir / "05_ap_vs_range_crop.csv",
            origin_dir,
            plot_metadata_dir / "05_ap_vs_range_crop.json",
            str(per_setting_ap_csv),
            "AP_R40",
            "AP sensitivity when far-range points are removed before detection.",
            forbidden,
        )
        figure_paths.append(figures_dir / "05_ap_vs_range_crop.png")

    yaw_rows = robustness_report.get("yaw_summary", [])
    if yaw_rows:
        _plot_line_multi(
            figures_dir / "06_calibration_yaw_sensitivity",
            "Projection-level Calibration Yaw Sensitivity",
            "yaw_deg",
            "avg_reprojection_shift_px",
            [row["yaw_deg"] for row in yaw_rows],
            {"reprojection_shift_px": [row["avg_reprojection_shift_px"] for row in yaw_rows]},
            3200,
            2200,
            600,
            plot_data_dir / "06_calibration_yaw_sensitivity.csv",
            origin_dir,
            plot_metadata_dir / "06_calibration_yaw_sensitivity.json",
            str(PROJECT_ROOT / "reports/lidar_system_algorithm/calibration_sync_robustness.json"),
            "pixels",
            "Projection-level sensitivity only; not claimed as detector AP perturbation.",
            forbidden,
        )
        figure_paths.append(figures_dir / "06_calibration_yaw_sensitivity.png")

    time_rows = robustness_report.get("time_offset_summary", [])
    if time_rows:
        _plot_line_multi(
            figures_dir / "07_time_offset_proxy_curve",
            "Time Offset Proxy",
            "frame_offset",
            "avg_center_drift_m",
            [row["frame_offset"] for row in time_rows],
            {"center_drift_m": [row["avg_box_center_displacement_bev_m"] for row in time_rows]},
            3200,
            2200,
            600,
            plot_data_dir / "07_time_offset_proxy_curve.csv",
            origin_dir,
            plot_metadata_dir / "07_time_offset_proxy_curve.json",
            str(PROJECT_ROOT / "reports/lidar_system_algorithm/calibration_sync_robustness.json"),
            "meters",
            "Adjacent-frame proxy only; not claimed as true sensor time-sync with IMU compensation.",
            forbidden,
        )
        figure_paths.append(figures_dir / "07_time_offset_proxy_curve.png")

    score_rows = []
    for row in per_setting_ap_rows:
        if row["perturbation_type"] == "postprocess_score_threshold":
            match = next(item for item in per_setting_health_rows if item["perturbation_type"] == row["perturbation_type"] and item["perturbation_value"] == row["perturbation_value"])
            score_rows.append((row, match))
    score_rows.sort(key=lambda item: 0.1 if item[0]["perturbation_value"] == "default" else float(item[0]["perturbation_value"]))
    if score_rows:
        x_vals = [0.1 if row["perturbation_value"] == "default" else float(row["perturbation_value"]) for row, _ in score_rows]
        _plot_line_multi(
            figures_dir / "08_ap_vs_score_threshold",
            "AP / Box Count vs Score Threshold",
            "score_threshold",
            "value",
            x_vals,
            {
                "mean_ap": [row["mean_ap_3d_moderate"] for row, _ in score_rows],
                "predicted_box_count": [health["total_box_count"] for _, health in score_rows],
            },
            3200,
            2200,
            600,
            plot_data_dir / "08_ap_vs_score_threshold.csv",
            origin_dir,
            plot_metadata_dir / "08_ap_vs_score_threshold.json",
            str(per_setting_ap_csv),
            "mixed",
            "Score-threshold perturbation is reported as postprocess sensitivity, not a threshold hack.",
            forbidden,
        )
        figure_paths.append(figures_dir / "08_ap_vs_score_threshold.png")

    note_figure(figures_dir / "09_ap_vs_nms_threshold", "NMS Threshold Perturbation", [
        "Skipped this turn.",
        "Reason: NMS threshold was not modified to avoid threshold-hack ambiguity in acceptance claims.",
        "This figure is intentionally a skipped-note panel, not a data curve.",
    ], 3200, 2200)
    write_plot_metadata(
        plot_metadata_dir / "09_ap_vs_nms_threshold.json",
        {
            "figure_name": "09_ap_vs_nms_threshold",
            "figure_path_png": str((figures_dir / "09_ap_vs_nms_threshold.png")),
            "figure_path_svg": str((figures_dir / "09_ap_vs_nms_threshold.svg")),
            "figure_path_pdf": str((figures_dir / "09_ap_vs_nms_threshold.pdf")),
            "source_csv": None,
            "source_report": str(output_dir / "deployment_acceptance_final_report.md"),
            "x_axis": "nms_threshold",
            "y_axis": "AP / predicted_box_count",
            "units": "mixed",
            "plotted_columns": [],
            "skipped": True,
            "skipped_reason": "NMS threshold was intentionally not changed this turn.",
            "safe_caption": "Skipped note for NMS threshold perturbation.",
            "forbidden_claims": forbidden,
        },
    )
    figure_paths.append(figures_dir / "09_ap_vs_nms_threshold.png")

    deployment_rows = [row for row in per_setting_ap_rows if row["perturbation_type"] == "deployment_precision"]
    deployment_rows.sort(key=lambda row: row["perturbation_value"])
    if deployment_rows:
        _plot_line_multi(
            figures_dir / "10_deployment_precision_ap_delta",
            "Deployment Precision AP Delta",
            "runtime_variant",
            "delta_mean_ap",
            [row["perturbation_value"] for row in deployment_rows],
            {"delta_mean_ap": [row["delta_mean"] for row in deployment_rows]},
            3200,
            2200,
            600,
            plot_data_dir / "10_deployment_precision_ap_delta.csv",
            origin_dir,
            plot_metadata_dir / "10_deployment_precision_ap_delta.json",
            str(per_setting_ap_csv),
            "AP_R40",
            "Wrapper and TRT parity are compared against the original OpenPCDet runtime on the same 200-frame slice.",
            forbidden,
        )
        figure_paths.append(figures_dir / "10_deployment_precision_ap_delta.png")

        _plot_line_multi(
            figures_dir / "11_deployment_precision_latency_delta",
            "Deployment Precision Latency",
            "runtime_variant",
            "latency_ms",
            ["openpcdet_original", "wrapper_pytorch", "trt_backbone_head_only"],
            {
                "core_latency_ms": [
                    original_report.get("latency_comparison", {}).get("pytorch_core_ms_mean"),
                    original_report.get("latency_comparison", {}).get("pytorch_core_ms_mean"),
                    original_report.get("latency_comparison", {}).get("trt_core_ms_mean"),
                ],
                "online_latency_ms": [
                    online_report.get("pytorch_summary", {}).get("online_total_ms", {}).get("mean"),
                    online_report.get("pytorch_summary", {}).get("online_total_ms", {}).get("mean"),
                    online_report.get("trt_summary", {}).get("online_total_ms", {}).get("mean"),
                ],
            },
            3200,
            2200,
            600,
            plot_data_dir / "11_deployment_precision_latency_delta.csv",
            origin_dir,
            plot_metadata_dir / "11_deployment_precision_latency_delta.json",
            str(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_online_latency.json"),
            "ms",
            "Core latency and online latency are separated; core numbers are not written as end-to-end detector latency.",
            forbidden,
        )
        figure_paths.append(figures_dir / "11_deployment_precision_latency_delta.png")

    def _filter_rows(rows, ptype, values=None):
        out = [row for row in rows if row["perturbation_type"] == ptype]
        if values is not None:
            out = [row for row in out if row["perturbation_value"] in values]
        return out

    dropout_range_rows = _filter_rows(per_setting_failure_range_rows, "point_dropout", {"0.0", "0.3", "0.7"})
    if dropout_range_rows:
        grouped = {}
        x = list(RANGE_BINS)
        for value in ["0.0", "0.3", "0.7"]:
            subset = [row for row in dropout_range_rows if row["perturbation_value"] == value]
            subset_map = {row["range_bin"]: row["recall"] for row in subset}
            grouped[f"recall_dropout_{value}"] = [subset_map.get(bin_name) for bin_name in x]
        _plot_line_multi(figures_dir / "12_failure_by_range_under_dropout", "Failure by Range under Dropout", "range_bin", "recall", x, grouped, 3200, 2200, 600, plot_data_dir / "12_failure_by_range_under_dropout.csv", origin_dir, plot_metadata_dir / "12_failure_by_range_under_dropout.json", str(per_setting_failure_range_csv), "ratio", "Range-segment recall under point dropout on a 200-frame slice.", forbidden)
        figure_paths.append(figures_dir / "12_failure_by_range_under_dropout.png")

    dropout_class_rows = _filter_rows(per_setting_failure_class_rows, "point_dropout", {"0.0", "0.3", "0.7"})
    if dropout_class_rows:
        grouped = {}
        x = list(CLASS_NAMES)
        for value in ["0.0", "0.3", "0.7"]:
            subset = [row for row in dropout_class_rows if row["perturbation_value"] == value]
            subset_map = {row["class_name"]: row["recall"] for row in subset}
            grouped[f"recall_dropout_{value}"] = [subset_map.get(bin_name) for bin_name in x]
        _plot_line_multi(figures_dir / "13_failure_by_class_under_dropout", "Failure by Class under Dropout", "class_name", "recall", x, grouped, 3200, 2200, 600, plot_data_dir / "13_failure_by_class_under_dropout.csv", origin_dir, plot_metadata_dir / "13_failure_by_class_under_dropout.json", str(per_setting_failure_class_csv), "ratio", "Class-level recall under point dropout on a 200-frame slice.", forbidden)
        figure_paths.append(figures_dir / "13_failure_by_class_under_dropout.png")

    note_figure(figures_dir / "14_failure_by_range_under_yaw", "Failure by Range under Yaw", [
        "Skipped as detector-level TP/FP/FN perturbation.",
        "Safe available evidence is projection-level yaw sensitivity only.",
        "See Figure 06 for real reprojection-shift data under yaw perturbation.",
    ], 3200, 2200)
    write_plot_metadata(plot_metadata_dir / "14_failure_by_range_under_yaw.json", {
        "figure_name": "14_failure_by_range_under_yaw",
        "figure_path_png": str(figures_dir / "14_failure_by_range_under_yaw.png"),
        "figure_path_svg": str(figures_dir / "14_failure_by_range_under_yaw.svg"),
        "figure_path_pdf": str(figures_dir / "14_failure_by_range_under_yaw.pdf"),
        "source_csv": None,
        "source_report": str(PROJECT_ROOT / "reports/lidar_system_algorithm/calibration_sync_robustness.json"),
        "x_axis": "range_bin",
        "y_axis": "TP/FP/FN",
        "units": "count",
        "plotted_columns": [],
        "skipped": True,
        "skipped_reason": "Detector-level yaw perturbation was not injected safely this turn.",
        "safe_caption": "Skipped note; only projection-level yaw sensitivity is claimed.",
        "forbidden_claims": forbidden,
    })
    figure_paths.append(figures_dir / "14_failure_by_range_under_yaw.png")

    score_class_rows = _filter_rows(per_setting_failure_class_rows, "postprocess_score_threshold", {"default", "0.2", "0.5"})
    if score_class_rows:
        grouped = {}
        x = list(CLASS_NAMES)
        for value in ["default", "0.2", "0.5"]:
            subset = [row for row in score_class_rows if row["perturbation_value"] == value]
            subset_map = {row["class_name"]: row["precision"] for row in subset}
            grouped[f"precision_thr_{value}"] = [subset_map.get(bin_name) for bin_name in x]
        _plot_line_multi(figures_dir / "15_failure_by_class_under_postprocess", "Failure by Class under Score Threshold", "class_name", "precision", x, grouped, 3200, 2200, 600, plot_data_dir / "15_failure_by_class_under_postprocess.csv", origin_dir, plot_metadata_dir / "15_failure_by_class_under_postprocess.json", str(per_setting_failure_class_csv), "ratio", "Class precision changes under score-threshold perturbation.", forbidden)
        figure_paths.append(figures_dir / "15_failure_by_class_under_postprocess.png")

    if dropout_rows:
        _plot_line_multi(figures_dir / "16_prediction_count_drift", "Prediction Count Drift", "dropout_ratio", "prediction_count_drift", [float(row["perturbation_value"]) for row in dropout_rows], {"prediction_count_drift": [next(item["prediction_count_drift"] for item in per_setting_health_rows if item["perturbation_type"] == "point_dropout" and item["perturbation_value"] == row["perturbation_value"]) for row in dropout_rows]}, 3200, 2200, 600, plot_data_dir / "16_prediction_count_drift.csv", origin_dir, plot_metadata_dir / "16_prediction_count_drift.json", str(per_setting_health_csv), "ratio", "Prediction count drift grows with stronger point dropout.", forbidden)
        figure_paths.append(figures_dir / "16_prediction_count_drift.png")

        _plot_line_multi(figures_dir / "17_score_distribution_drift", "Score Distribution Drift", "dropout_ratio", "score_distribution_drift", [float(row["perturbation_value"]) for row in dropout_rows], {"score_distribution_drift": [next(item["score_distribution_drift"] for item in per_setting_health_rows if item["perturbation_type"] == "point_dropout" and item["perturbation_value"] == row["perturbation_value"]) for row in dropout_rows]}, 3200, 2200, 600, plot_data_dir / "17_score_distribution_drift.csv", origin_dir, plot_metadata_dir / "17_score_distribution_drift.json", str(per_setting_health_csv), "JS divergence", "Score-distribution drift is computed as Jensen-Shannon divergence against the baseline prediction distribution.", forbidden)
        figure_paths.append(figures_dir / "17_score_distribution_drift.png")

        _plot_line_multi(figures_dir / "18_class_distribution_drift", "Class Distribution Drift", "dropout_ratio", "class_distribution_drift", [float(row["perturbation_value"]) for row in dropout_rows], {"class_distribution_drift": [next(item["class_distribution_drift"] for item in per_setting_health_rows if item["perturbation_type"] == "point_dropout" and item["perturbation_value"] == row["perturbation_value"]) for row in dropout_rows]}, 3200, 2200, 600, plot_data_dir / "18_class_distribution_drift.csv", origin_dir, plot_metadata_dir / "18_class_distribution_drift.json", str(per_setting_health_csv), "JS divergence", "Class-count drift is computed against baseline predictions, not against labels.", forbidden)
        figure_paths.append(figures_dir / "18_class_distribution_drift.png")

        _plot_line_multi(figures_dir / "19_range_distribution_drift", "Range Distribution Drift", "dropout_ratio", "range_distribution_drift", [float(row["perturbation_value"]) for row in dropout_rows], {"range_distribution_drift": [next(item["range_distribution_drift"] for item in per_setting_health_rows if item["perturbation_type"] == "point_dropout" and item["perturbation_value"] == row["perturbation_value"]) for row in dropout_rows]}, 3200, 2200, 600, plot_data_dir / "19_range_distribution_drift.csv", origin_dir, plot_metadata_dir / "19_range_distribution_drift.json", str(per_setting_health_csv), "JS divergence", "Range-bin drift is computed from prediction distance bins.", forbidden)
        figure_paths.append(figures_dir / "19_range_distribution_drift.png")

    health_points = [row for row in runtime_health_rows if row.get("mean_ap_drop") is not None]
    if health_points:
        _plot_scatter(figures_dir / "20_health_risk_vs_ap_drop", "Health Risk vs AP Drop", "label_free_health_risk", "mean_ap_drop", health_points, "label_free_health_risk", "mean_ap_drop", None, 3200, 2200, 600, plot_data_dir / "20_health_risk_vs_ap_drop.csv", origin_dir, plot_metadata_dir / "20_health_risk_vs_ap_drop.json", str(runtime_health_csv), "mixed", "Heuristic label-free health risk is correlated against AP drop on the evaluated perturbation settings only.", forbidden)
        figure_paths.append(figures_dir / "20_health_risk_vs_ap_drop.png")

    _plot_bar(figures_dir / "21_health_metric_correlation", "Health Metric Correlation", "metric_name", "correlation_with_mean_ap_drop", [row["metric_name"] for row in health_metric_rows], [row["correlation_with_mean_ap_drop"] for row in health_metric_rows], 3200, 2200, 600, plot_data_dir / "21_health_metric_correlation.csv", origin_dir, plot_metadata_dir / "21_health_metric_correlation.json", str(health_metric_correlation_csv), "Pearson r", "Correlation is reported as an acceptance-analysis heuristic, not a universal theorem.", forbidden)
    figure_paths.append(figures_dir / "21_health_metric_correlation.png")

    burst_rows = []
    for row in per_frame_prediction_rows:
        if row["perturbation_type"] != "point_dropout" or row["perturbation_value"] != "0.7":
            continue
        burst_rows.append(
            {
                "frame_id": row["frame_id"],
                "failure_count_proxy": max(int(row["predicted_box_count"] or 0) - int(np.median([r["predicted_box_count"] for r in per_frame_prediction_rows if r["perturbation_type"] == "point_dropout" and r["perturbation_value"] == "0.0"] or [0])), 0),
                "prediction_count": row["predicted_box_count"],
                "health_risk_proxy": next(item["label_free_health_risk"] for item in runtime_health_rows if item["perturbation_type"] == "point_dropout" and item["perturbation_value"] == "0.7"),
            }
        )
    if burst_rows:
        _plot_line_multi(figures_dir / "22_per_frame_failure_burst", "Per-frame Failure Burst", "frame_index", "value", list(range(len(burst_rows))), {"failure_count_proxy": [row["failure_count_proxy"] for row in burst_rows], "prediction_count": [row["prediction_count"] for row in burst_rows]}, 3200, 2200, 600, plot_data_dir / "22_per_frame_failure_burst.csv", origin_dir, plot_metadata_dir / "22_per_frame_failure_burst.json", str(per_frame_prediction_csv), "count", "Per-frame burst chart uses 0.7 dropout rows on the 200-frame slice.", forbidden)
        figure_paths.append(figures_dir / "22_per_frame_failure_burst.png")

    latency_scatter_rows = []
    for row in per_frame_prediction_rows:
        if row["perturbation_type"] != "point_dropout":
            continue
        latency_match = next((item for item in per_frame_latency_rows if item["frame_id"] == row["frame_id"] and item["perturbation_type"] == row["perturbation_type"] and item["perturbation_value"] == row["perturbation_value"]), None)
        if latency_match and latency_match.get("pytorch_core_ms") is not None:
            latency_scatter_rows.append(
                {
                    "frame_id": row["frame_id"],
                    "predicted_box_count": row["predicted_box_count"],
                    "point_count": row["point_count"],
                    "online_total_ms": latency_match.get("pytorch_core_ms"),
                    "perturbation_value": row["perturbation_value"],
                }
            )
    if latency_scatter_rows:
        _plot_scatter(figures_dir / "23_latency_vs_prediction_count", "Latency vs Prediction Count", "predicted_box_count", "online_total_ms", latency_scatter_rows, "predicted_box_count", "online_total_ms", None, 3200, 2200, 600, plot_data_dir / "23_latency_vs_prediction_count.csv", origin_dir, plot_metadata_dir / "23_latency_vs_prediction_count.json", str(per_frame_latency_csv), "ms", "For detector-perturbation runs, only PyTorch inference latency is available; visualization is excluded.", forbidden)
        figure_paths.append(figures_dir / "23_latency_vs_prediction_count.png")

    top_metric = max((row for row in health_metric_rows if row["correlation_with_mean_ap_drop"] is not None), key=lambda row: abs(row["correlation_with_mean_ap_drop"]), default=None)
    worst_ap = min((row for row in per_setting_ap_rows if row["delta_mean"] is not None), key=lambda row: row["delta_mean"], default=None)
    worst_class = min(
        (
            {"class_name": name, "mean_delta": _mean([row[f"delta_{name.lower()[:3] if name != 'Pedestrian' else 'ped'}"] for row in per_setting_ap_rows if row.get(f"delta_{name.lower()[:3] if name != 'Pedestrian' else 'ped'}") is not None])}
            for name in ["Car", "Pedestrian", "Cyclist"]
        ),
        key=lambda item: item["mean_delta"],
        default=None,
    )
    add_note("24_deployment_acceptance_dashboard", "Deployment Acceptance Dashboard", [
        f"Most sensitive perturbation: {worst_ap['perturbation_type']}={worst_ap['perturbation_value']} (delta_mean={worst_ap['delta_mean']:.3f})" if worst_ap else "Most sensitive perturbation: unavailable",
        f"Most correlated health metric: {top_metric['metric_name']} (r={top_metric['correlation_with_mean_ap_drop']:.3f})" if top_metric else "Health-metric correlation: unavailable",
        f"TRT AP parity on 200-frame slice: Car {trt_report.get('official_result_dict', {}).get('Car_3d/moderate_R40')}, Ped {trt_report.get('official_result_dict', {}).get('Pedestrian_3d/moderate_R40')}, Cyc {trt_report.get('official_result_dict', {}).get('Cyclist_3d/moderate_R40')}",
        "Boundary: backbone/head-only TRT is valid; full TensorRT detector is not claimed.",
    ])
    figure_paths.append(figures_dir / "24_deployment_acceptance_dashboard.png")

    contact_sheet_path = figures_dir / "deployment_acceptance_contact_sheet.png"
    save_contact_sheet(figure_paths, contact_sheet_path, columns=4, width_px=6400, dpi=300)

    # PPT panels
    slide_map = {
        "slide1_application_and_problem": [
            figures_dir / "01_application_scenario_deployment_acceptance.png",
            figures_dir / "02_perturbation_taxonomy.png",
            figures_dir / "03_acceptance_benchmark_pipeline.png",
        ],
        "slide2_perturbation_sensitivity": [
            figures_dir / "04_ap_vs_point_dropout.png",
            figures_dir / "05_ap_vs_range_crop.png",
            figures_dir / "06_calibration_yaw_sensitivity.png",
            figures_dir / "07_time_offset_proxy_curve.png",
            figures_dir / "08_ap_vs_score_threshold.png",
            figures_dir / "10_deployment_precision_ap_delta.png",
        ],
        "slide3_failure_attribution": [
            figures_dir / "12_failure_by_range_under_dropout.png",
            figures_dir / "13_failure_by_class_under_dropout.png",
            figures_dir / "16_prediction_count_drift.png",
            figures_dir / "17_score_distribution_drift.png",
            figures_dir / "18_class_distribution_drift.png",
            figures_dir / "24_deployment_acceptance_dashboard.png",
        ],
        "slide4_runtime_health_monitoring": [
            figures_dir / "20_health_risk_vs_ap_drop.png",
            figures_dir / "21_health_metric_correlation.png",
            figures_dir / "22_per_frame_failure_burst.png",
            figures_dir / "23_latency_vs_prediction_count.png",
            figures_dir / "11_deployment_precision_latency_delta.png",
            figures_dir / "24_deployment_acceptance_dashboard.png",
        ],
    }
    for slide_name, images in slide_map.items():
        _save_panel_triplet(images, ppt_dir / slide_name, slide_name.replace("_", " ").title())

    # Reports
    most_sensitive_perturbation = worst_ap["perturbation_type"] + "=" + str(worst_ap["perturbation_value"]) if worst_ap else None
    by_class_avg = []
    for class_name, delta_key in [("Car", "delta_car"), ("Pedestrian", "delta_ped"), ("Cyclist", "delta_cyc")]:
        vals = [row[delta_key] for row in per_setting_ap_rows if row.get(delta_key) is not None]
        by_class_avg.append((class_name, float(np.mean(vals)) if vals else 0.0))
    most_sensitive_class = min(by_class_avg, key=lambda item: item[1])[0] if by_class_avg else None
    range_fpfn = {}
    for row in per_setting_failure_range_rows:
        bucket = row["range_bin"]
        value = row["fp"] + row["fn"]
        range_fpfn[bucket] = range_fpfn.get(bucket, 0) + value
    most_sensitive_range = max(range_fpfn.items(), key=lambda item: item[1])[0] if range_fpfn else None
    top_metric_name = top_metric["metric_name"] if top_metric else None

    report_payload = {
        "status": "completed",
        "application_scenario": "Pre-deployment acceptance for LiDAR 3D detection models on unmanned platforms and intelligent hardware.",
        "frame_count": len(sample_ids),
        "eval_scope": f"{args.frames}-frame-slice",
        "is_full_val": False,
        "real_perturbations_run": sorted(list(REAL_PERTURBATIONS)),
        "skipped_perturbations": [
            {"perturbation_type": row[0], "reason": row[2]} for row in skipped_rows
        ],
        "most_sensitive_perturbation": most_sensitive_perturbation,
        "most_sensitive_class": most_sensitive_class,
        "most_sensitive_range": most_sensitive_range,
        "best_health_metric": top_metric_name,
        "paths": {
            "perturbation_matrix": str(perturbation_matrix_csv),
            "per_setting_ap": str(per_setting_ap_csv),
            "per_setting_prediction_health": str(per_setting_health_csv),
            "per_setting_failure_by_range": str(per_setting_failure_range_csv),
            "per_setting_failure_by_class": str(per_setting_failure_class_csv),
            "runtime_health_metrics": str(runtime_health_csv),
            "health_metric_correlation": str(health_metric_correlation_csv),
            "plot_data_dir": str(plot_data_dir),
            "origin_plot_data_dir": str(origin_dir),
            "figures_dir": str(figures_dir),
            "contact_sheet": str(contact_sheet_path),
            "ppt_dir": str(ppt_dir),
        },
        "safe_claims": [
            "This project builds a deployment acceptance and abnormal-attribution toolchain around an existing LiDAR detector.",
            "Official AP, runtime health metrics, perturbation sensitivity, and deployment precision parity are separated explicitly.",
            "Backbone/head-only TRT parity is reported as a deployment milestone, not as a full TensorRT detector.",
        ],
        "forbidden_claims": forbidden,
    }
    write_json(output_dir / "deployment_acceptance_final_report.json", report_payload)
    write_markdown(
        output_dir / "deployment_acceptance_final_report.md",
        "# Deployment Acceptance Final Report\n\n"
        "## 1. Application Scenario\n\n"
        "- Target use case: unmanned vehicles, robotics, intelligent hardware, and fixed LiDAR perception systems before model rollout.\n"
        "- Question: if offline AP is normal, how do we know the deployed runtime is still trustworthy?\n\n"
        "## 2. Why Offline AP Is Not Enough\n\n"
        "- Calibration error, time offset, sparse point clouds, postprocess drift, runtime latency spikes, and deployment-precision changes can all degrade behavior after training is finished.\n\n"
        "## 3. Acceptance Matrix\n\n"
        f"- Eval scope: `{args.frames}-frame slice`, not full-val.\n"
        f"- Real perturbations run: `{', '.join(sorted(REAL_PERTURBATIONS))}`\n"
        f"- Skipped perturbations are recorded with reasons in `perturbation_matrix.csv`.\n\n"
        "## 4. AP Sensitivity\n\n"
        f"- Most sensitive perturbation observed this turn: `{most_sensitive_perturbation}`\n"
        f"- Most sensitive class: `{most_sensitive_class}`\n"
        f"- Most sensitive range bin: `{most_sensitive_range}`\n\n"
        "## 5. Failure Attribution\n\n"
        "- Failure by range/class is exported per setting for dropout, score-threshold, and deployment-precision comparisons.\n"
        "- Detector-level yaw failure attribution was skipped; only projection-level yaw sensitivity is claimed.\n\n"
        "## 6. Runtime Health Metrics\n\n"
        f"- Most correlated label-free metric this turn: `{top_metric_name}`\n"
        "- Health risk is a heuristic and is not claimed as a general theory.\n\n"
        "## 7. Deployment Precision\n\n"
        "- Original / Wrapper / TRT backbone-head-only are compared on the same 200-frame slice.\n"
        "- Full TensorRT detector is not claimed.\n\n"
        "## 8. Limitations\n\n"
        "- 1000-frame slice results from earlier work are not rewritten as full-val.\n"
        "- Score-threshold perturbation is analysis only, not a threshold hack.\n"
        "- NMS-threshold and detector-input calibration-translation perturbations were skipped this turn.\n\n"
        "## 9. Safe Claims / Forbidden Claims\n\n"
        f"- Safe claims: `{report_payload['safe_claims']}`\n"
        f"- Forbidden claims: `{report_payload['forbidden_claims']}`\n",
    )

    storyline_payload = {
        "status": "completed",
        "language": "zh-CN",
        "slides": [
            {
                "title": "为什么离线 AP 正常不等于部署可信",
                "takeaway": "上线前需要系统性验收，而不是只看一次离线 AP。",
                "bullets": ["部署后会遇到点云稀疏、标定误差、时序偏移和后处理漂移。", "这些问题会直接影响漏检、误检、tracking 稳定性和输出分布。", "因此需要一套可复现、可审计的部署验收矩阵。"],
                "recommended_panel": "slide1_application_and_problem",
                "speaker_notes": "先把问题边界讲清楚：我做的不是新 detector，而是模型上线前的部署验收系统。",
                "interviewer_followups": ["你这个和普通 benchmark 有什么不同？", "为什么不能只看一次 KITTI AP？"],
                "answer_guidance": "强调 offline AP、online latency、runtime health metrics、扰动敏感性是不同口径。",
                "forbidden_claims": forbidden,
            },
            {
                "title": "不同扰动对 AP 和输出分布的影响并不一样",
                "takeaway": "点云退化、时序偏移、后处理和部署精度差异会呈现不同失效模式。",
                "bullets": ["point dropout 和 range crop 会直接影响 AP 和预测数量。", "yaw 这里只做 projection-level sensitivity，不冒充 detector AP 扰动。", "deployment precision 复用 original / wrapper / TRT 做 parity 分析。"],
                "recommended_panel": "slide2_perturbation_sensitivity",
                "speaker_notes": "这里要强调哪些是真实 detector perturbation，哪些只是 proxy 或 projection-level 分析。",
                "interviewer_followups": ["哪些扰动是真实跑的？", "哪些是 skipped？为什么？"],
                "answer_guidance": "直接对照 perturbation_matrix.csv 讲真实跑了什么、为什么跳过什么。",
                "forbidden_claims": forbidden,
            },
            {
                "title": "AP 下降之后还要继续做异常归因",
                "takeaway": "只知道 AP 掉了不够，还要回答是哪个距离段、哪个类别、哪种输出分布先出问题。",
                "bullets": ["按 range/class 导出 TP/FP/FN 和 precision/recall。", "再看 prediction count、score/class/range distribution drift。", "对同帧 deployment diff 和 per-frame burst 做异常定位。"],
                "recommended_panel": "slide3_failure_attribution",
                "speaker_notes": "把 failure attribution 讲成上线前验收报告，而不是学术刷榜分析。",
                "interviewer_followups": ["没有 GT 时你怎么发现异常？", "为什么要看 distribution drift？"],
                "answer_guidance": "过渡到第四页的 label-free runtime health risk。",
                "forbidden_claims": forbidden,
            },
            {
                "title": "没有 GT 时，用输出分布和时序一致性做健康监控",
                "takeaway": "上线后可以先用无标签 health metrics 做风险预警，再决定是否回收人工标注或回退部署。",
                "bullets": ["构造 heuristic label-free runtime health risk。", "看 health risk 和 AP drop 的相关性。", "把 backbone/head-only TRT 作为已验证的部署边界，而不是 full detector。"],
                "recommended_panel": "slide4_runtime_health_monitoring",
                "speaker_notes": "强调这个 health risk 是 heuristic，不是通用理论，但它足够工程化、可解释、可审计。",
                "interviewer_followups": ["这个 health risk 能直接替代 GT 吗？", "TRT 这条线为什么只保留 backbone/head？"],
                "answer_guidance": "明确说不能替代 GT，也不能写成 full TensorRT detector。",
                "forbidden_claims": forbidden,
            },
        ],
    }
    write_json(output_dir / "deployment_acceptance_ppt_storyline.json", storyline_payload)
    story_md_lines = ["# Deployment Acceptance PPT Storyline", ""]
    for index, slide in enumerate(storyline_payload["slides"], start=1):
        story_md_lines.extend(
            [
                f"## Slide {index}: {slide['title']}",
                "",
                f"- Takeaway: {slide['takeaway']}",
                f"- Bullets: {slide['bullets']}",
                f"- Recommended panel: {slide['recommended_panel']}",
                f"- Speaker notes: {slide['speaker_notes']}",
                f"- Interviewer followups: {slide['interviewer_followups']}",
                f"- Answer guidance: {slide['answer_guidance']}",
                f"- Forbidden claims: {slide['forbidden_claims']}",
                "",
            ]
        )
    write_markdown(output_dir / "deployment_acceptance_ppt_storyline.md", "\n".join(story_md_lines))

    print(json.dumps({"status": "completed", "output_dir": str(output_dir), "figure_count": len(figure_paths)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
