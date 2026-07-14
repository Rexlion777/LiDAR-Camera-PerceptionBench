from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import ensure_dir, read_json_or_default, write_csv, write_json, write_markdown


REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"
FIGURE_ROOT = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures" / "training_finetune"
PLOT_DATA_ROOT = REPORT_ROOT / "plot_data"
ORIGIN_ROOT = REPORT_ROOT / "origin_plot_data"
DEPLOYMENT_REPORT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "deployment_acceptance" / "deployment_acceptance_final_report.md"
README_PATH = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "README.md"
RESUME_PATH = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "resume_bullets.md"
INTERVIEW_QA_PATH = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "interview_qa.md"


@dataclass
class ExperimentArtifacts:
    name: str
    mode: str
    exp_dir: Path
    run_config: Path
    dataset_split_path: Path
    metrics_summary: dict[str, Any]
    train_log: Path | None
    baseline_eval: dict[str, Any]
    final_eval: dict[str, Any]
    train_rows: list[dict[str, Any]]
    lr_rows: list[dict[str, Any]]
    val_ids: list[str]


def list_experiment_dirs(pattern: str) -> list[Path]:
    return sorted(REPORT_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)


def select_best_experiment(prefix: str) -> Path | None:
    candidates = []
    for exp_dir in list_experiment_dirs(f"{prefix}_*"):
        final_eval = exp_dir / "final_official_eval" / "kitti_official_eval.json"
        metrics_summary = exp_dir / "metrics_summary.json"
        if final_eval.exists() and metrics_summary.exists():
            candidates.append(exp_dir)
    return candidates[-1] if candidates else None


def select_latest_experiment(prefix: str) -> Path | None:
    candidates = list_experiment_dirs(f"{prefix}_*")
    return candidates[-1] if candidates else None


def parse_split_ids(dataset_split_path: Path) -> list[str]:
    if not dataset_split_path.exists():
        return []
    text = dataset_split_path.read_text(encoding="utf-8", errors="ignore")
    if "[val_ids]" not in text:
        return []
    tail = text.split("[val_ids]", 1)[1]
    return [line.strip() for line in tail.splitlines() if line.strip()]


def parse_train_log(log_path: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if log_path is None or not log_path.exists():
        return [], []
    loss_rows: list[dict[str, Any]] = []
    lr_rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?Train:\s+"
        r"(?P<epoch>\d+)/(?P<epoch_total>\d+).*?Loss:\s*(?P<loss>[0-9eE+.\-]+)\s+\((?P<loss_avg>[0-9eE+.\-]+)\)\s+"
        r"LR:\s*(?P<lr>[0-9eE+.\-]+).*?Acc_iter\s+(?P<iter>\d+)"
        r".*?Data time:\s*(?P<data>[0-9eE+.\-]+)\((?P<data_avg>[0-9eE+.\-]+)\)\s+"
        r"Forward time:\s*(?P<fwd>[0-9eE+.\-]+)\((?P<fwd_avg>[0-9eE+.\-]+)\)\s+"
        r"Batch time:\s*(?P<batch>[0-9eE+.\-]+)\((?P<batch_avg>[0-9eE+.\-]+)\)"
    )
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        timestamp = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
        row = {
            "timestamp": timestamp.isoformat(),
            "iteration": int(match.group("iter")),
            "epoch": int(match.group("epoch")),
            "epoch_total": int(match.group("epoch_total")),
            "training_loss": float(match.group("loss")),
            "training_loss_avg": float(match.group("loss_avg")),
            "learning_rate": float(match.group("lr")),
            "data_time": max(float(match.group("data")), 0.0),
            "data_time_avg": max(float(match.group("data_avg")), 0.0),
            "forward_time": max(float(match.group("fwd")), 0.0),
            "forward_time_avg": max(float(match.group("fwd_avg")), 0.0),
            "batch_time": max(float(match.group("batch")), 0.0),
            "batch_time_avg": max(float(match.group("batch_avg")), 0.0),
        }
        loss_rows.append({k: row[k] for k in ("timestamp", "iteration", "epoch", "epoch_total", "training_loss", "training_loss_avg", "data_time", "forward_time", "batch_time")})
        lr_rows.append({k: row[k] for k in ("timestamp", "iteration", "epoch", "epoch_total", "learning_rate")})
    return loss_rows, lr_rows


def load_experiment(exp_dir: Path) -> ExperimentArtifacts:
    summary = read_json_or_default(exp_dir / "metrics_summary.json", {})
    mode = str(summary.get("training_mode") or exp_dir.name.split("_", 1)[0])
    train_log = exp_dir / "train.log"
    baseline_eval = read_json_or_default(exp_dir / "baseline_official_eval" / "kitti_official_eval.json", {})
    final_eval = read_json_or_default(exp_dir / "final_official_eval" / "kitti_official_eval.json", {})
    loss_rows, lr_rows = parse_train_log(train_log if train_log.exists() else None)
    dataset_split_path = exp_dir / "dataset_split_used.txt"
    return ExperimentArtifacts(
        name=exp_dir.name,
        mode=mode,
        exp_dir=exp_dir,
        run_config=exp_dir / "run_config.yaml",
        dataset_split_path=dataset_split_path,
        metrics_summary=summary if isinstance(summary, dict) else {},
        train_log=train_log if train_log.exists() else None,
        baseline_eval=baseline_eval if isinstance(baseline_eval, dict) else {},
        final_eval=final_eval if isinstance(final_eval, dict) else {},
        train_rows=loss_rows,
        lr_rows=lr_rows,
        val_ids=parse_split_ids(dataset_split_path),
    )


def eval_metric(payload: dict[str, Any], key: str) -> float | None:
    return payload.get("official_result_dict", {}).get(key) if isinstance(payload, dict) else None


def mean_ap(payload: dict[str, Any]) -> float | None:
    vals = [
        eval_metric(payload, "Car_3d/moderate_R40"),
        eval_metric(payload, "Pedestrian_3d/moderate_R40"),
        eval_metric(payload, "Cyclist_3d/moderate_R40"),
    ]
    numeric = [v for v in vals if isinstance(v, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


def parse_prediction_dir(pred_dir: Path, expected_ids: list[str]) -> dict[str, Any]:
    class_counts = {"Car": 0, "Pedestrian": 0, "Cyclist": 0}
    scores: list[float] = []
    ranges: list[float] = []
    empty_count = 0
    invalid_geometry = 0
    total_boxes = 0
    existing_files = 0
    for frame_id in expected_ids:
        txt_path = pred_dir / f"{frame_id}.txt"
        if not txt_path.exists():
            empty_count += 1
            continue
        existing_files += 1
        lines = [line.strip() for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
        if not lines:
            empty_count += 1
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 16:
                invalid_geometry += 1
                continue
            class_name = parts[0]
            if class_name in class_counts:
                class_counts[class_name] += 1
            try:
                height = float(parts[8])
                width = float(parts[9])
                length = float(parts[10])
                x = float(parts[11])
                z = float(parts[13])
                score = float(parts[15])
            except Exception:
                invalid_geometry += 1
                continue
            total_boxes += 1
            scores.append(score)
            ranges.append(math.sqrt(x * x + z * z))
            if not np.isfinite([height, width, length, x, z, score]).all() or min(height, width, length) <= 0:
                invalid_geometry += 1
    score_mean = float(np.mean(scores)) if scores else None
    score_p50 = float(np.percentile(scores, 50)) if scores else None
    score_p95 = float(np.percentile(scores, 95)) if scores else None
    range_mean = float(np.mean(ranges)) if ranges else None
    range_p95 = float(np.percentile(ranges, 95)) if ranges else None
    return {
        "prediction_dir": str(pred_dir),
        "expected_frame_count": len(expected_ids),
        "existing_prediction_file_count": existing_files,
        "empty_prediction_file_count": empty_count,
        "invalid_geometry_count": invalid_geometry,
        "total_box_count": total_boxes,
        "class_counts": class_counts,
        "score_mean": score_mean,
        "score_p50": score_p50,
        "score_p95": score_p95,
        "range_mean": range_mean,
        "range_p95": range_p95,
    }


def save_plot_data(name: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> tuple[Path, Path]:
    plot_csv = PLOT_DATA_ROOT / f"{name}.csv"
    origin_csv = ORIGIN_ROOT / f"{name}_origin.csv"
    write_csv(plot_csv, rows, fieldnames)
    write_csv(origin_csv, rows, fieldnames)
    return plot_csv, origin_csv


def save_figure(fig: plt.Figure, name: str) -> dict[str, str]:
    ensure_dir(FIGURE_ROOT)
    outputs = {}
    for suffix in ("png", "svg", "pdf"):
        path = FIGURE_ROOT / f"{name}.{suffix}"
        if suffix == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        outputs[suffix] = str(path)
    plt.close(fig)
    return outputs


def summary_card(title: str, lines: list[str]) -> plt.Figure:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.05, 0.92, title, fontsize=24, weight="bold", va="top")
    y = 0.84
    for line in lines:
        ax.text(0.06, y, line, fontsize=15, va="top")
        y -= 0.08
    return fig


def generate_figures(smoke: ExperimentArtifacts, finetune: ExperimentArtifacts, comparison_rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    ensure_dir(FIGURE_ROOT)
    ensure_dir(PLOT_DATA_ROOT)
    ensure_dir(ORIGIN_ROOT)
    outputs: dict[str, dict[str, str]] = {}

    fig = summary_card(
        "PointPillars Training / Fine-tuning Pipeline",
        [
            "KITTI subset split -> OpenPCDet config clone -> subset training / fine-tuning",
            "checkpoint save -> external official AP eval -> deployment diagnostics reuse",
            "Purpose: close the training-eval-deployment loop, not claim KITTI full training or SOTA",
        ],
    )
    outputs["01_training_pipeline"] = save_figure(fig, "01_training_pipeline")

    loss_rows = []
    for tag, artifacts in (("smoke_train", smoke), ("subset_finetune", finetune)):
        for row in artifacts.train_rows:
            loss_rows.append(
                {
                    "series": tag,
                    "iteration": row["iteration"],
                    "epoch": row["epoch"],
                    "training_loss": row["training_loss"],
                    "training_loss_avg": row["training_loss_avg"],
                }
            )
    save_plot_data("02_loss_curve", loss_rows, ["series", "iteration", "epoch", "training_loss", "training_loss_avg"])
    fig, ax = plt.subplots(figsize=(12, 9))
    for tag, artifacts in (("smoke_train", smoke), ("subset_finetune", finetune)):
        xs = [row["iteration"] for row in artifacts.train_rows]
        ys = [row["training_loss_avg"] for row in artifacts.train_rows]
        if xs and ys:
            ax.plot(xs, ys, label=tag, linewidth=2.2)
    ax.set_title("Training Loss Curve")
    ax.set_xlabel("Accumulated Iteration")
    ax.set_ylabel("Average Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    outputs["02_loss_curve"] = save_figure(fig, "02_loss_curve")

    lr_plot_rows = []
    for tag, artifacts in (("smoke_train", smoke), ("subset_finetune", finetune)):
        for row in artifacts.lr_rows:
            lr_plot_rows.append({"series": tag, "iteration": row["iteration"], "epoch": row["epoch"], "learning_rate": row["learning_rate"]})
    save_plot_data("03_lr_schedule", lr_plot_rows, ["series", "iteration", "epoch", "learning_rate"])
    fig, ax = plt.subplots(figsize=(12, 9))
    for tag, artifacts in (("smoke_train", smoke), ("subset_finetune", finetune)):
        xs = [row["iteration"] for row in artifacts.lr_rows]
        ys = [row["learning_rate"] for row in artifacts.lr_rows]
        if xs and ys:
            ax.plot(xs, ys, label=tag, linewidth=2.2)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Accumulated Iteration")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    outputs["03_lr_schedule"] = save_figure(fig, "03_lr_schedule")

    ap_rows = [
        {
            "model_version": "pretrained_baseline",
            "car_ap": eval_metric(finetune.baseline_eval, "Car_3d/moderate_R40"),
            "ped_ap": eval_metric(finetune.baseline_eval, "Pedestrian_3d/moderate_R40"),
            "cyc_ap": eval_metric(finetune.baseline_eval, "Cyclist_3d/moderate_R40"),
            "mean_ap": mean_ap(finetune.baseline_eval),
        },
        {
            "model_version": "subset_finetune_epoch3",
            "car_ap": eval_metric(finetune.final_eval, "Car_3d/moderate_R40"),
            "ped_ap": eval_metric(finetune.final_eval, "Pedestrian_3d/moderate_R40"),
            "cyc_ap": eval_metric(finetune.final_eval, "Cyclist_3d/moderate_R40"),
            "mean_ap": mean_ap(finetune.final_eval),
        },
    ]
    save_plot_data("04_pretrained_vs_finetuned_ap", ap_rows, ["model_version", "car_ap", "ped_ap", "cyc_ap", "mean_ap"])
    fig, ax = plt.subplots(figsize=(12, 9))
    categories = ["car_ap", "ped_ap", "cyc_ap", "mean_ap"]
    labels = ["Car", "Pedestrian", "Cyclist", "Mean"]
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width / 2, [ap_rows[0][c] for c in categories], width, label="pretrained_baseline")
    ax.bar(x + width / 2, [ap_rows[1][c] for c in categories], width, label="subset_finetune_epoch3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("3D AP_R40")
    ax.set_title("Pretrained vs Fine-tuned AP on Subset Val")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    outputs["04_pretrained_vs_finetuned_ap"] = save_figure(fig, "04_pretrained_vs_finetuned_ap")

    trend_rows = [
        {"experiment": smoke.name, "checkpoint": "checkpoint_epoch_1", "car_ap": eval_metric(smoke.final_eval, "Car_3d/moderate_R40"), "ped_ap": eval_metric(smoke.final_eval, "Pedestrian_3d/moderate_R40"), "cyc_ap": eval_metric(smoke.final_eval, "Cyclist_3d/moderate_R40")},
        {"experiment": finetune.name, "checkpoint": "checkpoint_epoch_3", "car_ap": eval_metric(finetune.final_eval, "Car_3d/moderate_R40"), "ped_ap": eval_metric(finetune.final_eval, "Pedestrian_3d/moderate_R40"), "cyc_ap": eval_metric(finetune.final_eval, "Cyclist_3d/moderate_R40")},
    ]
    save_plot_data("05_checkpoint_eval_trend", trend_rows, ["experiment", "checkpoint", "car_ap", "ped_ap", "cyc_ap"])
    fig = summary_card(
        "Checkpoint Eval Trend",
        [
            "This turn evaluated one smoke checkpoint and one fine-tuned checkpoint.",
            "No fake multi-checkpoint trend is drawn because only one external official AP point was measured per experiment.",
            f"Smoke epoch1 mean AP: {mean_ap(smoke.final_eval):.2f}" if mean_ap(smoke.final_eval) is not None else "Smoke epoch1 mean AP: unavailable",
            f"Fine-tune epoch3 mean AP: {mean_ap(finetune.final_eval):.2f}" if mean_ap(finetune.final_eval) is not None else "Fine-tune epoch3 mean AP: unavailable",
        ],
    )
    outputs["05_checkpoint_eval_trend"] = save_figure(fig, "05_checkpoint_eval_trend")

    runtime_rows = []
    for artifacts in (smoke, finetune):
        summary = artifacts.metrics_summary
        start = summary.get("start_time")
        end = summary.get("end_time")
        wall_minutes = None
        if start and end:
            wall_minutes = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60.0
        epoch_count = summary.get("epochs")
        runtime_rows.append(
            {
                "experiment": artifacts.name,
                "mode": artifacts.mode,
                "epochs": epoch_count,
                "wall_minutes": wall_minutes,
                "batch_size": summary.get("batch_size"),
                "train_sample_count": summary.get("train_sample_count"),
            }
        )
    save_plot_data("06_training_runtime_summary", runtime_rows, ["experiment", "mode", "epochs", "wall_minutes", "batch_size", "train_sample_count"])
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.bar([row["mode"] for row in runtime_rows], [row["wall_minutes"] or 0.0 for row in runtime_rows], color=["#4C78A8", "#F58518"])
    ax.set_ylabel("Wall Time (minutes)")
    ax.set_title("Training Runtime Summary")
    ax.grid(True, axis="y", alpha=0.3)
    outputs["06_training_runtime_summary"] = save_figure(fig, "06_training_runtime_summary")

    fig = summary_card(
        "Training to Deployment Loop",
        [
            "subset_finetune checkpoint -> external official eval -> prediction export",
            "same checkpoint -> deployment diagnostics: box count / score / range / empty file / invalid geometry",
            "This closes the model-iteration-to-deployment-analysis loop without claiming full KITTI training",
        ],
    )
    outputs["07_training_to_deployment_loop"] = save_figure(fig, "07_training_to_deployment_loop")
    return outputs


def build_comparison_rows(smoke: ExperimentArtifacts, finetune: ExperimentArtifacts) -> list[dict[str, Any]]:
    rows = []
    for artifacts in (smoke, finetune):
        summary = artifacts.metrics_summary
        for variant_name, eval_payload, ckpt in (
            ("pretrained_baseline", artifacts.baseline_eval, summary.get("pretrained_checkpoint")),
            ("trained_checkpoint", artifacts.final_eval, summary.get("trained_checkpoint")),
        ):
            rows.append(
                {
                    "experiment_name": f"{artifacts.name}::{variant_name}",
                    "training_mode": artifacts.mode,
                    "init_checkpoint": summary.get("pretrained_checkpoint"),
                    "trained_checkpoint": ckpt,
                    "train_sample_count": summary.get("train_sample_count"),
                    "val_sample_count": summary.get("val_sample_count"),
                    "epochs": summary.get("epochs"),
                    "batch_size": summary.get("batch_size"),
                    "learning_rate": summary.get("learning_rate"),
                    "car_ap_3d_moderate": eval_metric(eval_payload, "Car_3d/moderate_R40"),
                    "ped_ap_3d_moderate": eval_metric(eval_payload, "Pedestrian_3d/moderate_R40"),
                    "cyc_ap_3d_moderate": eval_metric(eval_payload, "Cyclist_3d/moderate_R40"),
                    "mean_ap_3d_moderate": mean_ap(eval_payload),
                    "eval_scope": "subset-val",
                    "full_val": False,
                    "notes": "subset training / fine-tuning experiment",
                }
            )
    return rows


def write_experiment_csvs(artifacts: ExperimentArtifacts) -> None:
    write_csv(
        artifacts.exp_dir / "loss_curve.csv",
        artifacts.train_rows or [{"timestamp": "", "iteration": 0, "epoch": 0, "epoch_total": 0, "training_loss": "", "training_loss_avg": "", "data_time": "", "forward_time": "", "batch_time": ""}],
        ["timestamp", "iteration", "epoch", "epoch_total", "training_loss", "training_loss_avg", "data_time", "forward_time", "batch_time"],
    )
    write_csv(
        artifacts.exp_dir / "lr_schedule.csv",
        artifacts.lr_rows or [{"timestamp": "", "iteration": 0, "epoch": 0, "epoch_total": 0, "learning_rate": ""}],
        ["timestamp", "iteration", "epoch", "epoch_total", "learning_rate"],
    )
    final_dict = artifacts.final_eval.get("official_result_dict", {}) if isinstance(artifacts.final_eval, dict) else {}
    eval_rows = [{"metric": key, "value": value} for key, value in sorted(final_dict.items())]
    write_csv(artifacts.exp_dir / "eval_ap_summary.csv", eval_rows or [{"metric": "status", "value": ""}], ["metric", "value"])


def build_deployment_diagnostics(finetune: ExperimentArtifacts) -> dict[str, Any]:
    baseline_pred_dir = Path(finetune.baseline_eval.get("prediction_dir", ""))
    final_pred_dir = Path(finetune.final_eval.get("prediction_dir", ""))
    baseline_stats = parse_prediction_dir(baseline_pred_dir, finetune.val_ids)
    final_stats = parse_prediction_dir(final_pred_dir, finetune.val_ids)
    ap_delta = {
        "car_ap_delta": (eval_metric(finetune.final_eval, "Car_3d/moderate_R40") or 0.0) - (eval_metric(finetune.baseline_eval, "Car_3d/moderate_R40") or 0.0),
        "ped_ap_delta": (eval_metric(finetune.final_eval, "Pedestrian_3d/moderate_R40") or 0.0) - (eval_metric(finetune.baseline_eval, "Pedestrian_3d/moderate_R40") or 0.0),
        "cyc_ap_delta": (eval_metric(finetune.final_eval, "Cyclist_3d/moderate_R40") or 0.0) - (eval_metric(finetune.baseline_eval, "Cyclist_3d/moderate_R40") or 0.0),
        "mean_ap_delta": (mean_ap(finetune.final_eval) or 0.0) - (mean_ap(finetune.baseline_eval) or 0.0),
    }
    payload = {
        "experiment_name": finetune.name,
        "scope": "subset-val deployment-style diagnostics",
        "baseline": {
            "checkpoint": finetune.metrics_summary.get("pretrained_checkpoint"),
            "ap": {
                "car": eval_metric(finetune.baseline_eval, "Car_3d/moderate_R40"),
                "pedestrian": eval_metric(finetune.baseline_eval, "Pedestrian_3d/moderate_R40"),
                "cyclist": eval_metric(finetune.baseline_eval, "Cyclist_3d/moderate_R40"),
                "mean": mean_ap(finetune.baseline_eval),
            },
            "prediction_health": baseline_stats,
        },
        "fine_tuned": {
            "checkpoint": finetune.metrics_summary.get("trained_checkpoint"),
            "ap": {
                "car": eval_metric(finetune.final_eval, "Car_3d/moderate_R40"),
                "pedestrian": eval_metric(finetune.final_eval, "Pedestrian_3d/moderate_R40"),
                "cyclist": eval_metric(finetune.final_eval, "Cyclist_3d/moderate_R40"),
                "mean": mean_ap(finetune.final_eval),
            },
            "prediction_health": final_stats,
        },
        "ap_delta": ap_delta,
        "notes": [
            "This diagnostic reuses the same subset-val split used for fine-tuning evaluation.",
            "Purpose: verify that the fine-tuned checkpoint can enter the same deployment diagnostics path.",
            "This is not a claim that fine-tuning universally improves PointPillars.",
        ],
    }
    write_json(REPORT_ROOT / "finetune_deployment_diagnostics.json", payload)
    lines = [
        "# Fine-tuned Checkpoint Deployment Diagnostics",
        "",
        f"- Experiment: `{finetune.name}`",
        "- Scope: `subset-val deployment-style diagnostics`",
        f"- Baseline mean AP_R40: `{mean_ap(finetune.baseline_eval):.4f}`",
        f"- Fine-tuned mean AP_R40: `{mean_ap(finetune.final_eval):.4f}`",
        f"- Mean AP delta: `{ap_delta['mean_ap_delta']:.4f}`",
        f"- Baseline total boxes: `{baseline_stats['total_box_count']}`",
        f"- Fine-tuned total boxes: `{final_stats['total_box_count']}`",
        f"- Baseline empty prediction files: `{baseline_stats['empty_prediction_file_count']}`",
        f"- Fine-tuned empty prediction files: `{final_stats['empty_prediction_file_count']}`",
        f"- Baseline invalid geometry: `{baseline_stats['invalid_geometry_count']}`",
        f"- Fine-tuned invalid geometry: `{final_stats['invalid_geometry_count']}`",
        "",
        "## Boundary",
        "",
        "- This step proves the fine-tuned checkpoint can flow into the same deployment diagnostics stack.",
        "- It does not claim full KITTI training or a generally better detector.",
    ]
    write_markdown(REPORT_ROOT / "finetune_deployment_diagnostics.md", "\n".join(lines))
    return payload


def append_deployment_addendum(payload: dict[str, Any]) -> None:
    if not DEPLOYMENT_REPORT.exists():
        return
    marker = "## Training / Fine-tuning Loop Addendum"
    content = DEPLOYMENT_REPORT.read_text(encoding="utf-8", errors="ignore")
    if marker in content:
        return
    addendum = "\n\n" + marker + "\n\n" + "\n".join(
        [
            "- Added a real PointPillars subset training / fine-tuning smoke loop on KITTI subsets.",
            "- Training artifacts now include real OpenPCDet logs, checkpoints, external official AP eval JSON, and loss curve CSV.",
            "- `smoke_train` was used as a one-epoch executable smoke test, not a convergence claim.",
            "- `subset_finetune` reused the pretrained PointPillars checkpoint and was evaluated on the same subset-val split.",
            "- Built-in WSL `tools/test.py` eval remained partial because the KITTI numba CUDA evaluator still hit a `Missing libdevice file` blocker; official AP used the external helper path already present in the project.",
        ]
    )
    DEPLOYMENT_REPORT.write_text(content + addendum, encoding="utf-8")


def update_material_files(finetune: ExperimentArtifacts) -> None:
    readme = f"""# LiDAR 3D 检测模型部署验收与异常归因系统

## Current Scope

- OpenPCDet / PointPillars baseline with KITTI runtime, official eval, deployment acceptance and diagnostics
- TensorRT milestone is **backbone/head-only**: `PyTorch VFE/scatter + TensorRT backbone/dense_head + OpenPCDet native post_processing/export`
- Added **PointPillars subset training / fine-tuning smoke loop** to close the data-prep -> training -> checkpoint -> official-eval -> deployment-diagnostics chain

## Training / Fine-tuning Loop

- Training env audit: `reports/lidar_system_algorithm/training_finetune/training_env_audit.md`
- Smoke train: one epoch / subset scope, used to prove loader, loss, backward, checkpoint and external eval path
- Subset fine-tune: pretrained PointPillars init, subset-train / subset-val scope, not full KITTI training
- Main report: `reports/lidar_system_algorithm/training_finetune/pointpillars_training_finetune_report.md`

## Safe Claims

- Completed real subset training / fine-tuning experiments with logs, checkpoints and official AP on subset-val splits
- Closed the training-eval-deployment loop for PointPillars in the current repo
- Fine-tuned checkpoint can be fed into the same deployment diagnostics stack used by the deployment acceptance project

## Forbidden Claims

- Do not call this a new 3D detection model
- Do not call subset training full KITTI training
- Do not call smoke train convergence
- Do not call backbone/head-only TensorRT a full TensorRT detector
"""
    README_PATH.write_text(readme, encoding="utf-8")

    resume = """# Resume Bullets

## Training Loop Add-on

### 短版
补齐 PointPillars 从 KITTI 数据准备、模型训练/微调、checkpoint 管理、官方 AP 评估到部署诊断的闭环，基于真实训练日志输出 loss 曲线、AP 对比和微调后运行质量分析。

### 长版
完成 PointPillars subset 训练/微调与评估复现，打通 KITTI 数据准备、训练配置、checkpoint 保存、官方 AP 评估和部署验收流程；基于真实训练日志生成 loss 曲线和 AP 对比，并将微调后 checkpoint 接入运行质量诊断工具，分析预测框数量、score/range 分布、类别分布和异常输出变化。

### 面试边界
- 这是 subset training / smoke fine-tuning 闭环，不是 KITTI full training，也不是 SOTA 训练。
- smoke_train 只证明训练代码、损失回传、checkpoint 和评估链路可执行，不能写成模型收敛。
- subset_finetune 结果只在 subset-val 范围内解释。
"""
    RESUME_PATH.write_text(resume, encoding="utf-8")

    qa = """# Interview Q&A

## 你做过模型训练/微调吗？

做过，但我会明确说是 PointPillars 的 subset training / fine-tuning smoke experiment，不是完整 KITTI full training。目的不是刷榜，而是补齐数据准备、训练配置、checkpoint、官方 AP 评估和部署诊断闭环。

## 训练后你怎么验证？

先在同一 subset-val split 上做 external official AP 评估，再把 fine-tuned checkpoint 接入已有 deployment diagnostics，比较预测框数量、score/range 分布、类别分布、empty prediction 和 invalid geometry。

## 你会把这段写成模型收敛吗？

不会。`smoke_train` 只说明训练链路可运行，`subset_finetune` 只说明我把训练-评估-部署迭代闭环真实打通。

## 这能和 TensorRT / 部署验收主线怎么接？

主线仍然是部署验收与异常归因系统。训练这部分的价值是：模型每次微调后，可以进入同一套 official AP、prediction health、latency 和 deployment acceptance 检查，不需要重新搭一套验证框架。
"""
    INTERVIEW_QA_PATH.write_text(qa, encoding="utf-8")


def generate_report(smoke: ExperimentArtifacts, finetune: ExperimentArtifacts, diagnostics: dict[str, Any], comparison_rows: list[dict[str, Any]], figure_outputs: dict[str, dict[str, str]]) -> dict[str, Any]:
    audit = read_json_or_default(REPORT_ROOT / "training_env_audit.json", {})
    from_scratch_dir = select_latest_experiment("subset_train_from_scratch")
    from_scratch_status = read_json_or_default(from_scratch_dir / "train_status.json", {}) if from_scratch_dir else {"status": "skipped", "reason": "not_run"}
    report_payload = {
        "status": "completed",
        "why_add_training_loop": "Close the data-prep / training / checkpoint / official-eval / deployment-diagnostics loop for PointPillars.",
        "environment_audit_path": str(REPORT_ROOT / "training_env_audit.json"),
        "experiments": {
            "smoke_train": {
                "experiment_dir": str(smoke.exp_dir),
                "train_sample_count": smoke.metrics_summary.get("train_sample_count"),
                "val_sample_count": smoke.metrics_summary.get("val_sample_count"),
                "epochs": smoke.metrics_summary.get("epochs"),
                "batch_size": smoke.metrics_summary.get("batch_size"),
                "learning_rate": smoke.metrics_summary.get("learning_rate"),
                "used_pretrained_checkpoint": smoke.metrics_summary.get("used_pretrained_checkpoint"),
                "pretrained_checkpoint": smoke.metrics_summary.get("pretrained_checkpoint"),
                "trained_checkpoint": smoke.metrics_summary.get("trained_checkpoint"),
                "status": smoke.metrics_summary.get("status", "partial"),
                "boundary_note": "One-epoch smoke train. Not a convergence claim.",
                "baseline_mean_ap": mean_ap(smoke.baseline_eval),
                "final_mean_ap": mean_ap(smoke.final_eval),
            },
            "subset_finetune": {
                "experiment_dir": str(finetune.exp_dir),
                "train_sample_count": finetune.metrics_summary.get("train_sample_count"),
                "val_sample_count": finetune.metrics_summary.get("val_sample_count"),
                "epochs": finetune.metrics_summary.get("epochs"),
                "batch_size": finetune.metrics_summary.get("batch_size"),
                "learning_rate": finetune.metrics_summary.get("learning_rate"),
                "used_pretrained_checkpoint": finetune.metrics_summary.get("used_pretrained_checkpoint"),
                "pretrained_checkpoint": finetune.metrics_summary.get("pretrained_checkpoint"),
                "trained_checkpoint": finetune.metrics_summary.get("trained_checkpoint"),
                "status": finetune.metrics_summary.get("status", "partial"),
                "baseline_mean_ap": mean_ap(finetune.baseline_eval),
                "final_mean_ap": mean_ap(finetune.final_eval),
            },
            "subset_train_from_scratch": from_scratch_status,
        },
        "comparison_csv": str(REPORT_ROOT / "training_eval_comparison.csv"),
        "finetune_deployment_diagnostics": diagnostics,
        "figures": figure_outputs,
        "limitations": [
            "Subset scope only. No full KITTI training claim.",
            "Built-in WSL test.py official eval remained partial due missing libdevice in the numba CUDA evaluator path.",
            "External official eval helper was used for measured AP outputs.",
        ],
        "safe_claims": [
            "Completed PointPillars subset training / fine-tuning smoke experiment.",
            "Closed the data-prep / training / checkpoint / official-eval / deployment-diagnostics loop.",
            "Generated real loss curves and AP comparison from logs and official eval artifacts.",
        ],
        "forbidden_claims": [
            "Do not claim KITTI full training.",
            "Do not claim SOTA.",
            "Do not claim smoke_train convergence.",
            "Do not claim subset-val as full-val.",
        ],
    }
    write_json(REPORT_ROOT / "pointpillars_training_finetune_report.json", report_payload)
    lines = [
        "# PointPillars Training / Fine-tuning Report",
        "",
        "## Why add training / fine-tuning loop",
        "",
        "- The deployment acceptance project already covered runtime, official eval, deployment diagnostics and TensorRT boundaries.",
        "- This add-on closes the missing model-iteration evidence: data prep -> subset training / fine-tuning -> checkpoint -> official AP -> deployment diagnostics.",
        "",
        "## Environment audit",
        "",
        f"- Audit path: `{REPORT_ROOT / 'training_env_audit.md'}`",
        f"- CUDA available in WSL training env: `{audit.get('cuda_available')}`",
        f"- spconv available: `{audit.get('spconv_available')}`",
        f"- pcdet ops available: `{audit.get('pcdet_ops_available')}`",
        "",
        "## Dataset subset and split",
        "",
        f"- Smoke train: `{smoke.metrics_summary.get('train_sample_count')}` train / `{smoke.metrics_summary.get('val_sample_count')}` val",
        f"- Subset fine-tune: `{finetune.metrics_summary.get('train_sample_count')}` train / `{finetune.metrics_summary.get('val_sample_count')}` val",
        "- Eval scope: `subset-val`; `full_val=false`",
        "",
        "## Training modes",
        "",
        f"- `smoke_train`: 1 epoch, batch size {smoke.metrics_summary.get('batch_size')}, lr {smoke.metrics_summary.get('learning_rate')}, pretrained init `{smoke.metrics_summary.get('used_pretrained_checkpoint')}`",
        f"- `subset_finetune`: 3 epochs, batch size {finetune.metrics_summary.get('batch_size')}, lr {finetune.metrics_summary.get('learning_rate')}, pretrained init `{finetune.metrics_summary.get('used_pretrained_checkpoint')}`",
        "- `subset_train_from_scratch`: supported by wrapper but skipped in this run to keep runtime bounded.",
        "",
        "## Evaluation results",
        "",
        f"- Smoke baseline mean AP_R40: `{mean_ap(smoke.baseline_eval):.4f}`",
        f"- Smoke final mean AP_R40: `{mean_ap(smoke.final_eval):.4f}`",
        f"- Fine-tune baseline mean AP_R40: `{mean_ap(finetune.baseline_eval):.4f}`",
        f"- Fine-tune final mean AP_R40: `{mean_ap(finetune.final_eval):.4f}`",
        f"- Comparison CSV: `{REPORT_ROOT / 'training_eval_comparison.csv'}`",
        "",
        "## Deployment diagnostics after fine-tuning",
        "",
        f"- Diagnostics report: `{REPORT_ROOT / 'finetune_deployment_diagnostics.md'}`",
        f"- Fine-tuned total boxes: `{diagnostics['fine_tuned']['prediction_health']['total_box_count']}`",
        f"- Fine-tuned empty prediction files: `{diagnostics['fine_tuned']['prediction_health']['empty_prediction_file_count']}`",
        "",
        "## Limitations",
        "",
        "- This is subset training / subset fine-tuning evidence, not KITTI full training.",
        "- `smoke_train` is a chain verification experiment, not a convergence claim.",
        "- WSL built-in `tools/test.py` official eval remained partial because the KITTI numba CUDA evaluator still hit `Missing libdevice file`; measured AP came from the external official eval wrapper path.",
        "",
        "## Safe claims",
        "",
        "- Completed real PointPillars subset training / fine-tuning smoke experiments with logs, checkpoints and official AP outputs.",
        "- Closed the training-eval-deployment loop inside the current repo.",
        "",
        "## Forbidden claims",
        "",
        "- Do not claim SOTA.",
        "- Do not claim KITTI full training.",
        "- Do not call smoke train convergence.",
        "- Do not call subset-val full-val.",
        "",
        "## Resume command",
        "",
        "```bash",
        'cd "."',
        "PYTHONPATH=. python scripts/lidar_system_algorithm/run_pointpillars_subset_training.py --mode subset_finetune --max_samples 200 --epochs 3 --batch_size 2 --seed 42 --workers 0",
        "```",
    ]
    write_markdown(REPORT_ROOT / "pointpillars_training_finetune_report.md", "\n".join(lines))
    return report_payload


def main() -> None:
    smoke_dir = select_best_experiment("smoke_train")
    finetune_dir = select_best_experiment("subset_finetune")
    if smoke_dir is None or finetune_dir is None:
        raise SystemExit("Missing completed smoke_train or subset_finetune experiment directories.")
    smoke = load_experiment(smoke_dir)
    finetune = load_experiment(finetune_dir)
    write_experiment_csvs(smoke)
    write_experiment_csvs(finetune)
    comparison_rows = build_comparison_rows(smoke, finetune)
    write_csv(REPORT_ROOT / "training_eval_comparison.csv", comparison_rows, list(comparison_rows[0].keys()))
    figure_outputs = generate_figures(smoke, finetune, comparison_rows)
    diagnostics = build_deployment_diagnostics(finetune)
    update_material_files(finetune)
    append_deployment_addendum(diagnostics)
    report_payload = generate_report(smoke, finetune, diagnostics, comparison_rows, figure_outputs)
    print(json.dumps({"status": "completed", "report": str(REPORT_ROOT / "pointpillars_training_finetune_report.md"), "comparison_csv": str(REPORT_ROOT / "training_eval_comparison.csv")}, indent=2))


if __name__ == "__main__":
    main()
