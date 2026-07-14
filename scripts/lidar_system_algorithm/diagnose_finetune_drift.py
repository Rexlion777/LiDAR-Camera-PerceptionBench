from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.failure_matcher import distance_bin, read_kitti_objects
from runtime.lidar_system_algorithm.report_schema import ensure_dir, read_json_or_default, write_csv, write_json, write_markdown


REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"
DIAG_ROOT = REPORT_ROOT / "diagnose_finetune_drift"
SPLIT_ROOT = REPORT_ROOT / "expanded_splits"
FIG_ROOT = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures" / "training_finetune" / "diagnose_finetune_drift"
PLOT_ROOT = DIAG_ROOT / "plot_data"
ORIGIN_ROOT = DIAG_ROOT / "origin_plot_data"
PRED_ROOT = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "results" / "diagnose_finetune_drift"
PRETRAINED_CKPT = Path(r"checkpoints/pointpillar_kitti.pth")
CLASS_ORDER = ("Car", "Pedestrian", "Cyclist")
RANGE_ORDER = ("0-20m", "20-40m", "40-60m", "60m+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose per-class drift after expanded PointPillars fine-tuning.")
    parser.add_argument("--holdout-split", default="holdout_eval_500")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def latest_exp(prefix: str) -> Path:
    matches = sorted(REPORT_ROOT.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"Missing experiment directory for prefix={prefix}")
    return matches[-1]


def split_ids(name: str) -> list[str]:
    return [line.strip() for line in (SPLIT_ROOT / f"{name}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def run_eval(model_name: str, cfg_file: Path, ckpt: Path, split_name: str, skip_existing: bool) -> dict[str, Any]:
    out_dir = ensure_dir(DIAG_ROOT / "holdout_eval_outputs" / model_name)
    pred_dir = ensure_dir(PRED_ROOT / f"{model_name}_{split_name}_txt")
    eval_json = out_dir / "kitti_official_eval.json"
    if skip_existing and eval_json.exists():
        payload = read_json_or_default(eval_json, {})
        if isinstance(payload, dict) and payload.get("status") == "completed":
            return {"model_name": model_name, "status": "completed", "payload": payload, "reused": True}
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "run_kitti_official_eval.py"),
        "--kitti-root",
        str(PROJECT_ROOT / "data" / "kitti_object_raw" / "extracted"),
        "--openpcdet-root",
        "external/OpenPCDet",
        "--cfg-file",
        str(cfg_file),
        "--ckpt",
        str(ckpt),
        "--split-file",
        str(SPLIT_ROOT / f"{split_name}.txt"),
        "--output-dir",
        str(out_dir),
        "--pred-dir",
        str(pred_dir),
        "--max-frames",
        "0",
    ]
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=14400, check=False)
    payload = read_json_or_default(eval_json, {})
    status = "completed" if completed.returncode == 0 and isinstance(payload, dict) and payload.get("status") == "completed" else "failed"
    return {
        "model_name": model_name,
        "status": status,
        "command": " ".join(cmd),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "payload": payload,
        "reused": False,
    }


def metric(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get("official_result_dict", {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def build_models() -> list[dict[str, Any]]:
    subset_dir = latest_exp("subset_finetune")
    expanded_dir = REPORT_ROOT / "expanded_finetune_1000_20260630_150229"
    lr0004_dir = latest_exp("expanded_finetune_1000_lr0004")
    subset_status = read_json_or_default(subset_dir / "train_status.json", {})
    expanded_status = read_json_or_default(expanded_dir / "train_status.json", {})
    lr0004_status = read_json_or_default(lr0004_dir / "train_status.json", {})

    models: list[dict[str, Any]] = [
        {
            "model_name": "pretrained_baseline",
            "group": "baseline",
            "lr": None,
            "epoch": 0,
            "cfg_file": expanded_dir / "run_config.yaml",
            "checkpoint_path": PRETRAINED_CKPT,
            "train_samples": 0,
            "val_samples": 0,
            "notes": "Public pretrained checkpoint; no subset fine-tuning.",
        },
        {
            "model_name": "subset_finetune_200_50_epoch3",
            "group": "current_200_50",
            "lr": float(subset_status.get("learning_rate", 0.0008)),
            "epoch": 3,
            "cfg_file": subset_dir / "run_config.yaml",
            "checkpoint_path": Path(str(subset_status["trained_checkpoint"])),
            "train_samples": int(subset_status["train_sample_count"]),
            "val_samples": int(subset_status["val_sample_count"]),
            "notes": "Earlier 200/50 subset_finetune checkpoint.",
        },
    ]
    for exp_dir, status, group, lr in (
        (expanded_dir, expanded_status, "expanded_lr0008", 0.0008),
        (lr0004_dir, lr0004_status, "expanded_lr0004", 0.0004),
    ):
        ckpt_dir = Path(str(status["opencpdet_output_dir"])) / "ckpt"
        for epoch in (1, 2, 3):
            ckpt = ckpt_dir / f"checkpoint_epoch_{epoch}.pth"
            models.append(
                {
                    "model_name": f"{group}_epoch{epoch}",
                    "group": group,
                    "lr": lr,
                    "epoch": epoch,
                    "cfg_file": exp_dir / "run_config.yaml",
                    "checkpoint_path": ckpt,
                    "train_samples": int(status["train_sample_count"]),
                    "val_samples": int(status["val_sample_count"]),
                    "notes": f"{group} checkpoint epoch {epoch}.",
                }
            )
    return models


def normalize_hist(values: list[int]) -> list[float]:
    total = float(sum(values))
    return [float(v) / total if total > 0 else 0.0 for v in values]


def gt_counts_by_class(sample_ids: list[str]) -> dict[str, int]:
    label_dir = PROJECT_ROOT / "data" / "kitti_object_raw" / "extracted" / "training" / "label_2"
    counts = {name: 0 for name in CLASS_ORDER}
    for sample_id in sample_ids:
        for obj in read_kitti_objects(label_dir / f"{sample_id}.txt", is_prediction=False):
            if obj.class_name in counts:
                counts[obj.class_name] += 1
    return counts


def per_class_prediction_stats(pred_dir: Path, sample_ids: list[str], gt_counts: dict[str, int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    score_values = {name: [] for name in CLASS_ORDER}
    range_counts = {name: {bucket: 0 for bucket in RANGE_ORDER} for name in CLASS_ORDER}
    class_counts = {name: 0 for name in CLASS_ORDER}
    empty_frames = 0
    invalid_geometry = 0
    total_boxes = 0
    for sample_id in sample_ids:
        path = pred_dir / f"{sample_id}.txt"
        objects = read_kitti_objects(path, is_prediction=True)
        if not path.exists() or not objects:
            empty_frames += 1
        for obj in objects:
            if obj.class_name not in CLASS_ORDER:
                continue
            total_boxes += 1
            class_counts[obj.class_name] += 1
            if obj.score is not None:
                score_values[obj.class_name].append(float(obj.score))
            range_counts[obj.class_name][distance_bin(obj.distance_m)] += 1
            dims = [obj.height, obj.width, obj.length, *obj.location_camera_xyz, obj.rotation_y]
            if obj.height <= 0 or obj.width <= 0 or obj.length <= 0 or not all(np.isfinite(dims)):
                invalid_geometry += 1
    rows = []
    for class_name in CLASS_ORDER:
        arr = np.asarray(score_values[class_name], dtype=np.float64)
        pred_count = class_counts[class_name]
        rows.append(
            {
                "class_name": class_name,
                "total_boxes_class": pred_count,
                "boxes_per_frame_class": pred_count / max(len(sample_ids), 1),
                "score_mean_class": float(arr.mean()) if arr.size else None,
                "score_median_class": float(np.percentile(arr, 50)) if arr.size else None,
                "score_p10_class": float(np.percentile(arr, 10)) if arr.size else None,
                "score_p90_class": float(np.percentile(arr, 90)) if arr.size else None,
                "near_count_class": range_counts[class_name]["0-20m"],
                "mid_count_class": range_counts[class_name]["20-40m"],
                "far_count_class": range_counts[class_name]["40-60m"] + range_counts[class_name]["60m+"],
                "prediction_gt_ratio_class": pred_count / max(gt_counts[class_name], 1),
                "gt_count_class": gt_counts[class_name],
            }
        )
    summary = {
        "total_boxes": total_boxes,
        "boxes_per_frame": total_boxes / max(len(sample_ids), 1),
        "empty_prediction_frames": empty_frames,
        "invalid_geometry_boxes": invalid_geometry,
        "class_counts": class_counts,
    }
    return rows, summary


def eval_config_diff(models: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        ("CLASS_NAMES",),
        ("DATA_CONFIG", "POINT_CLOUD_RANGE"),
        ("DATA_CONFIG", "DATA_PROCESSOR"),
        ("MODEL", "POST_PROCESSING"),
        ("MODEL", "DENSE_HEAD", "ANCHOR_GENERATOR_CONFIG"),
        ("MODEL", "DENSE_HEAD", "TARGET_ASSIGNER_CONFIG"),
        ("DATA_CONFIG", "DATA_SPLIT"),
        ("DATA_CONFIG", "INFO_PATH"),
    ]
    loaded = {}
    for model in models[:3]:
        path = Path(model["cfg_file"])
        loaded[model["model_name"]] = yaml.safe_load(path.read_text(encoding="utf-8"))

    def dig(cfg: dict[str, Any], path: tuple[str, ...]) -> Any:
        value: Any = cfg
        for key in path:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    diffs = []
    for key_path in keys:
        values = {name: dig(cfg, key_path) for name, cfg in loaded.items()}
        serialized = {name: json.dumps(value, sort_keys=True, ensure_ascii=False, default=str) for name, value in values.items()}
        consistent = len(set(serialized.values())) == 1
        diffs.append({"field": ".".join(key_path), "consistent": consistent, "values": values})
    payload = {"status": "completed", "overall_consistent": all(item["consistent"] for item in diffs if not item["field"].endswith("DATA_SPLIT") and not item["field"].endswith("INFO_PATH")), "diffs": diffs}
    write_json(DIAG_ROOT / "eval_config_diff.json", payload)
    lines = ["# Eval Config Diff", "", f"- Overall core eval config consistent: `{payload['overall_consistent']}`", ""]
    for item in diffs:
        lines.extend([f"## {item['field']}", "", f"- consistent: `{item['consistent']}`", ""])
        if not item["consistent"]:
            lines.append("- Difference is reported for audit only; no existing result is modified.")
            lines.append("")
    write_markdown(DIAG_ROOT / "eval_config_diff.md", "\n".join(lines))
    return payload


def select_best(checkpoint_rows: list[dict[str, Any]], diag_rows: list[dict[str, Any]]) -> dict[str, Any]:
    diag_by_model = {}
    for row in diag_rows:
        diag_by_model.setdefault(row["model_name"], {})[row["class_name"]] = row
    baseline = next(row for row in checkpoint_rows if row["model_name"] == "pretrained_baseline")
    baseline_car = float(baseline["car_ap_3d_moderate"])
    baseline_boxes = next(row for row in diag_rows if row["model_name"] == "pretrained_baseline" and row["class_name"] == "Car")
    best_mean = max(checkpoint_rows, key=lambda row: float(row["mean_ap_3d_moderate"]))
    candidates = []
    for row in checkpoint_rows:
        if row["model_name"] == "pretrained_baseline":
            car_drop = 0.0
        else:
            car_drop = baseline_car - float(row["car_ap_3d_moderate"])
        car_diag = diag_by_model[row["model_name"]]["Car"]
        box_ratio = float(car_diag["prediction_gt_ratio_class"]) / max(float(baseline_boxes["prediction_gt_ratio_class"]), 1e-9)
        score_penalty = max(0.0, 0.30 - float(car_diag["score_mean_class"] or 0.0))
        balanced_score = float(row["mean_ap_3d_moderate"]) - 0.6 * max(0.0, car_drop) - 2.0 * max(0.0, box_ratio - 1.20) - 8.0 * score_penalty
        candidates.append({**row, "car_drop_vs_pretrained": car_drop, "car_box_ratio_vs_pretrained": box_ratio, "balanced_score": balanced_score})
    best_balanced = max(candidates, key=lambda row: float(row["balanced_score"]))
    return {
        "best_mean_ap_model": best_mean["model_name"],
        "best_mean_ap": best_mean["mean_ap_3d_moderate"],
        "best_balanced_checkpoint": best_balanced["model_name"],
        "best_balanced_score": best_balanced["balanced_score"],
        "balanced_candidates": candidates,
    }


def write_checkpoint_sweep(rows: list[dict[str, Any]], selection: dict[str, Any]) -> None:
    fields = [
        "model_name",
        "group",
        "lr",
        "checkpoint_path",
        "epoch",
        "train_samples",
        "val_samples",
        "holdout_samples",
        "car_ap_3d_moderate",
        "ped_ap_3d_moderate",
        "cyc_ap_3d_moderate",
        "mean_ap_3d_moderate",
        "best_mean_ap",
        "best_balanced_checkpoint",
        "notes",
    ]
    out_rows = []
    for row in rows:
        out_rows.append({**row, "best_mean_ap": selection["best_mean_ap_model"] == row["model_name"], "best_balanced_checkpoint": selection["best_balanced_checkpoint"] == row["model_name"]})
    write_csv(DIAG_ROOT / "checkpoint_sweep_holdout_ap.csv", out_rows, fields)
    write_json(DIAG_ROOT / "checkpoint_sweep_holdout_ap.json", {"status": "completed", "selection": selection, "rows": out_rows})
    lines = ["# Checkpoint Sweep Holdout AP", "", f"- Best mean AP: `{selection['best_mean_ap_model']}`", f"- Best balanced checkpoint: `{selection['best_balanced_checkpoint']}`", ""]
    for row in out_rows:
        lines.append(f"- {row['model_name']}: Car `{float(row['car_ap_3d_moderate']):.4f}`, Ped `{float(row['ped_ap_3d_moderate']):.4f}`, Cyc `{float(row['cyc_ap_3d_moderate']):.4f}`, Mean `{float(row['mean_ap_3d_moderate']):.4f}`")
    write_markdown(DIAG_ROOT / "checkpoint_sweep_holdout_ap.md", "\n".join(lines))


def save_plot_csv(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    write_csv(PLOT_ROOT / f"{name}.csv", rows, fields)
    write_csv(ORIGIN_ROOT / f"{name}_origin.csv", rows, fields)


def save_fig(fig: plt.Figure, name: str) -> None:
    ensure_dir(FIG_ROOT)
    for ext in ("png", "svg", "pdf"):
        path = FIG_ROOT / f"{name}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_figures(ap_rows: list[dict[str, Any]], diag_rows: list[dict[str, Any]], lr_rows: list[dict[str, Any]], selection: dict[str, Any]) -> None:
    ensure_dir(PLOT_ROOT)
    ensure_dir(ORIGIN_ROOT)
    ap_plot_rows = [row for row in ap_rows if row["group"].startswith("expanded_lr") or row["model_name"] == "pretrained_baseline"]
    save_plot_csv("checkpoint_sweep_ap_by_class", ap_plot_rows, list(ap_plot_rows[0].keys()))
    fig, ax = plt.subplots(figsize=(12, 9))
    x = np.arange(len(ap_plot_rows))
    width = 0.24
    ax.bar(x - width, [float(r["car_ap_3d_moderate"]) for r in ap_plot_rows], width, label="Car")
    ax.bar(x, [float(r["ped_ap_3d_moderate"]) for r in ap_plot_rows], width, label="Pedestrian")
    ax.bar(x + width, [float(r["cyc_ap_3d_moderate"]) for r in ap_plot_rows], width, label="Cyclist")
    ax.set_xticks(x)
    ax.set_xticklabels([r["model_name"] for r in ap_plot_rows], rotation=30, ha="right")
    ax.set_title("holdout_eval_500 checkpoint sweep AP by class")
    ax.set_ylabel("3D AP_R40 moderate")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, "checkpoint_sweep_ap_by_class")

    save_plot_csv("checkpoint_sweep_mean_ap", ap_plot_rows, list(ap_plot_rows[0].keys()))
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot([r["model_name"] for r in ap_plot_rows], [float(r["mean_ap_3d_moderate"]) for r in ap_plot_rows], marker="o")
    ax.set_title("holdout_eval_500 checkpoint sweep mean AP")
    ax.set_ylabel("Mean 3D AP_R40 moderate")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3)
    save_fig(fig, "checkpoint_sweep_mean_ap")

    class_rows = [row for row in diag_rows if row["class_name"] in CLASS_ORDER and row["model_name"] in {r["model_name"] for r in ap_plot_rows}]
    save_plot_csv("per_class_box_count_comparison", class_rows, list(class_rows[0].keys()))
    fig, ax = plt.subplots(figsize=(12, 9))
    xlabels = [r["model_name"] for r in ap_plot_rows]
    x = np.arange(len(xlabels))
    width = 0.24
    for idx, cls in enumerate(CLASS_ORDER):
        vals = [float(next(r for r in class_rows if r["model_name"] == name and r["class_name"] == cls)["boxes_per_frame_class"]) for name in xlabels]
        ax.bar(x + (idx - 1) * width, vals, width, label=cls)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_title("holdout_eval_500 per-class boxes/frame")
    ax.set_ylabel("Boxes / frame")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, "per_class_box_count_comparison")

    save_plot_csv("per_class_score_mean_comparison", class_rows, list(class_rows[0].keys()))
    fig, ax = plt.subplots(figsize=(12, 9))
    for cls in CLASS_ORDER:
        vals = [float(next(r for r in class_rows if r["model_name"] == name and r["class_name"] == cls)["score_mean_class"]) for name in xlabels]
        ax.plot(xlabels, vals, marker="o", label=cls)
    ax.set_title("holdout_eval_500 per-class score mean")
    ax.set_ylabel("Score mean")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "per_class_score_mean_comparison")

    save_plot_csv("prediction_gt_ratio_by_class", class_rows, list(class_rows[0].keys()))
    fig, ax = plt.subplots(figsize=(12, 9))
    for cls in CLASS_ORDER:
        vals = [float(next(r for r in class_rows if r["model_name"] == name and r["class_name"] == cls)["prediction_gt_ratio_class"]) for name in xlabels]
        ax.plot(xlabels, vals, marker="o", label=cls)
    ax.set_title("holdout_eval_500 prediction / GT ratio by class")
    ax.set_ylabel("Prediction / GT ratio")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "prediction_gt_ratio_by_class")

    save_plot_csv("lr_sweep_ap_comparison", lr_rows, list(lr_rows[0].keys()))
    fig, ax = plt.subplots(figsize=(12, 9))
    for lr in sorted({r["lr"] for r in lr_rows if r["lr"] not in ("", None)}):
        subset = [r for r in lr_rows if r["lr"] == lr]
        subset.sort(key=lambda r: int(r["epoch"]))
        ax.plot([int(r["epoch"]) for r in subset], [float(r["mean_ap_3d_moderate"]) for r in subset], marker="o", label=f"lr={lr}")
    ax.axhline(float(next(r for r in ap_rows if r["model_name"] == "pretrained_baseline")["mean_ap_3d_moderate"]), color="black", linestyle="--", label="pretrained")
    ax.set_title("holdout_eval_500 learning-rate sweep AP")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean 3D AP_R40 moderate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "lr_sweep_ap_comparison")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.04, 0.94, "holdout_eval_500 drift diagnosis summary", fontsize=22, weight="bold", va="top")
    summary_lines = [
        f"Best mean AP checkpoint: {selection['best_mean_ap_model']}",
        f"Best balanced checkpoint: {selection['best_balanced_checkpoint']}",
        "Diagnosis: mean AP can improve while Car drops because Ped/Cyc gains and higher box count change the class balance.",
        "Deployment warning: higher boxes/frame is a potential FP-risk signal, not a direct performance claim.",
        "Boundary: subset fine-tuning + holdout_eval_500 only; not full KITTI training or full val.",
    ]
    y = 0.82
    for line in summary_lines:
        ax.text(0.05, y, line, fontsize=15, va="top")
        y -= 0.11
    save_fig(fig, "drift_diagnosis_summary")


def generate_report(ap_rows: list[dict[str, Any]], diag_rows: list[dict[str, Any]], config_payload: dict[str, Any], selection: dict[str, Any], lr_rows: list[dict[str, Any]]) -> None:
    baseline = next(row for row in ap_rows if row["model_name"] == "pretrained_baseline")
    lr0004_best = max([r for r in ap_rows if r["group"] == "expanded_lr0004"], key=lambda r: float(r["mean_ap_3d_moderate"]))
    lr0008_best = max([r for r in ap_rows if r["group"] == "expanded_lr0008"], key=lambda r: float(r["mean_ap_3d_moderate"]))
    payload = {
        "status": "completed",
        "eval_scope": "holdout_eval_500",
        "full_val": False,
        "config_consistent": config_payload["overall_consistent"],
        "selection": selection,
        "lr0004_best": lr0004_best,
        "lr0008_best": lr0008_best,
        "safe_claims": [
            "Diagnosed subset fine-tuning drift using the same holdout_eval_500 split.",
            "Compared AP, per-class prediction count, score distribution, and prediction/GT ratio.",
            "Found a more stable checkpoint only when supported by holdout AP and deployment diagnostics.",
        ],
        "forbidden_claims": [
            "Do not claim SOTA.",
            "Do not claim full KITTI training.",
            "Do not call holdout_eval_500 full-val.",
            "Do not interpret higher box count as direct performance improvement.",
        ],
    }
    write_json(DIAG_ROOT / "finetune_drift_diagnosis_report.json", payload)
    lines = [
        "# Fine-tune Drift Diagnosis Report",
        "",
        "## 1. Problem statement",
        "",
        "expanded_finetune_1000 lr=0.0008 produced a small mean AP gain on holdout_eval_500, but Car AP dropped while Pedestrian/Cyclist improved. This report diagnoses whether the model became more aggressive and whether a more conservative fine-tuning setting is safer.",
        "",
        "## 2. Observed AP changes",
        "",
        f"- Pretrained mean AP_R40: `{float(baseline['mean_ap_3d_moderate']):.4f}`",
        f"- lr=0.0008 best mean AP checkpoint: `{lr0008_best['model_name']}` mean `{float(lr0008_best['mean_ap_3d_moderate']):.4f}`",
        f"- lr=0.0004 best mean AP checkpoint: `{lr0004_best['model_name']}` mean `{float(lr0004_best['mean_ap_3d_moderate']):.4f}`",
        "",
        "## 3. Box count and score distribution changes",
        "",
        "Box count increases are interpreted as a potential recall/FP trade-off, not as a direct performance improvement. Per-class prediction/GT ratios and score means are reported in `per_class_diagnostics.csv`.",
        "",
        "## 4. Config consistency check",
        "",
        f"- Core eval config consistent: `{config_payload['overall_consistent']}`",
        "- DATA_SPLIT / INFO_PATH can differ by experiment split, but all holdout comparisons use the same `holdout_eval_500` split file.",
        "",
        "## 5. Checkpoint sweep",
        "",
        f"- Best mean AP: `{selection['best_mean_ap_model']}`",
        f"- Best balanced checkpoint: `{selection['best_balanced_checkpoint']}`",
        "",
        "## 6. Per-class diagnostics",
        "",
        "- Check `per_class_diagnostics.csv` for class-level boxes/frame, score p10/p50/p90, range distribution, and prediction/GT ratio.",
        "- If Car prediction/GT ratio increases while Car AP drops, that is a concrete FP-risk / localization-quality warning.",
        "",
        "## 7. Learning-rate sweep",
        "",
        f"- lr=0.0004 was run as a new experiment and compared on the same holdout_eval_500.",
        f"- lr=0.0002 was skipped in this turn to avoid another long training run after lr=0.0004 already provided a conservative setting.",
        "",
        "## 8. Best checkpoint recommendation",
        "",
        f"- Recommended checkpoint: `{selection['best_balanced_checkpoint']}`.",
        "- This recommendation balances mean AP, Car AP drop, box-count drift, and score decline. If this is the pretrained baseline, it means fine-tuning did not produce a safer deployment candidate.",
        "",
        "## 9. Why fine-tuning can hurt pretrained models",
        "",
        "- A pretrained KITTI PointPillars checkpoint is already well calibrated for common Car patterns.",
        "- Short subset fine-tuning can shift class balance toward Pedestrian/Cyclist or subset-specific examples.",
        "- A higher number of predictions can raise recall for some classes while increasing false-positive risk or reducing precision for Car.",
        "",
        "## 10. Deployment implication",
        "",
        "- For deployment acceptance, AP must be read together with class-wise output distribution and score/range drift.",
        "- The safer candidate is not automatically the highest mean AP checkpoint if it creates large Car degradation or aggressive low-score outputs.",
        "",
        "## 11. Safe claims",
        "",
        *[f"- {item}" for item in payload["safe_claims"]],
        "",
        "## 12. Forbidden claims",
        "",
        *[f"- {item}" for item in payload["forbidden_claims"]],
        "",
        "## 13. Resume wording",
        "",
        "- 完成 PointPillars 1000-sample subset fine-tuning 与固定 500-frame holdout 评估，打通 KITTI 数据准备、训练/微调、native OpenPCDet 官方评估与部署诊断闭环，并进一步分析微调后 per-class AP、预测框数量、score/range 分布和潜在 false-positive 风险。",
    ]
    write_markdown(DIAG_ROOT / "finetune_drift_diagnosis_report.md", "\n".join(lines))


def update_materials() -> None:
    resume_path = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "resume_bullets.md"
    interview_path = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "interview_qa.md"
    resume_text = resume_path.read_text(encoding="utf-8", errors="replace")
    bullet = "\n\n## Fine-tune Drift Diagnosis\n\n完成 PointPillars 1000-sample subset fine-tuning 与固定 500-frame holdout 评估，打通 KITTI 数据准备、训练/微调、native OpenPCDet 官方评估与部署诊断闭环，并进一步分析微调后 per-class AP、预测框数量、score/range 分布和潜在 false-positive 风险。\n"
    if "Fine-tune Drift Diagnosis" not in resume_text:
        resume_path.write_text(resume_text.rstrip() + bullet, encoding="utf-8")
    interview_text = interview_path.read_text(encoding="utf-8", errors="replace")
    qa = "\n\n## 为什么 fine-tune 后 Car AP 下降？\n\n我的结论不是简单说 fine-tune 变好或变差，而是它改变了类别间 trade-off。需要同时看 holdout AP、per-class boxes/frame、score 分布和 prediction/GT ratio；如果框数增加但 Car AP 下降，就要按 false-positive 或定位质量风险解释，而不能把 box count 增加写成性能提升。\n"
    if "为什么 fine-tune 后 Car AP 下降" not in interview_text:
        interview_path.write_text(interview_text.rstrip() + qa, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(DIAG_ROOT)
    models = build_models()
    sample_ids = split_ids(args.holdout_split)
    gt_counts = gt_counts_by_class(sample_ids)
    attempts = []
    ap_rows = []
    diag_rows = []
    for model in models:
        attempt = run_eval(model["model_name"], Path(model["cfg_file"]), Path(model["checkpoint_path"]), args.holdout_split, args.skip_existing)
        attempts.append(attempt)
        if attempt["status"] != "completed":
            continue
        payload = attempt["payload"]
        pred_dir = Path(payload["prediction_dir"])
        car = metric(payload, "Car_3d/moderate_R40")
        ped = metric(payload, "Pedestrian_3d/moderate_R40")
        cyc = metric(payload, "Cyclist_3d/moderate_R40")
        vals = [v for v in (car, ped, cyc) if isinstance(v, (int, float))]
        ap_rows.append(
            {
                "model_name": model["model_name"],
                "group": model["group"],
                "lr": model["lr"] if model["lr"] is not None else "",
                "checkpoint_path": str(model["checkpoint_path"]),
                "epoch": model["epoch"],
                "train_samples": model["train_samples"],
                "val_samples": model["val_samples"],
                "holdout_samples": len(sample_ids),
                "car_ap_3d_moderate": car,
                "ped_ap_3d_moderate": ped,
                "cyc_ap_3d_moderate": cyc,
                "mean_ap_3d_moderate": sum(vals) / len(vals) if vals else None,
                "notes": model["notes"],
            }
        )
        class_rows, summary = per_class_prediction_stats(pred_dir, sample_ids, gt_counts)
        for class_row in class_rows:
            diag_rows.append(
                {
                    "model_name": model["model_name"],
                    "group": model["group"],
                    "lr": model["lr"] if model["lr"] is not None else "",
                    "epoch": model["epoch"],
                    "holdout_samples": len(sample_ids),
                    "total_boxes": summary["total_boxes"],
                    "boxes_per_frame": summary["boxes_per_frame"],
                    "empty_prediction_frames": summary["empty_prediction_frames"],
                    "invalid_geometry_boxes": summary["invalid_geometry_boxes"],
                    **class_row,
                }
            )
    config_payload = eval_config_diff(models)
    selection = select_best(ap_rows, diag_rows)
    write_checkpoint_sweep(ap_rows, selection)
    fields = list(diag_rows[0].keys())
    write_csv(DIAG_ROOT / "per_class_diagnostics.csv", diag_rows, fields)
    write_json(DIAG_ROOT / "per_class_diagnostics.json", {"status": "completed", "gt_counts": gt_counts, "rows": diag_rows})
    md_lines = ["# Per-class Diagnostics", "", f"- Holdout split: `{args.holdout_split}`", f"- GT counts: `{gt_counts}`", ""]
    for row in diag_rows:
        if row["class_name"] == "Car":
            md_lines.append(f"- {row['model_name']} Car boxes/frame `{float(row['boxes_per_frame_class']):.3f}`, score mean `{float(row['score_mean_class']):.4f}`, pred/GT `{float(row['prediction_gt_ratio_class']):.3f}`")
    write_markdown(DIAG_ROOT / "per_class_diagnostics.md", "\n".join(md_lines))
    lr_rows = [row for row in ap_rows if str(row["group"]).startswith("expanded_lr")]
    write_csv(DIAG_ROOT / "lr_sweep_summary.csv", lr_rows, list(lr_rows[0].keys()))
    write_json(DIAG_ROOT / "lr_sweep_report.json", {"status": "completed", "rows": lr_rows, "selection": selection, "skipped_lr0002_reason": "Skipped to keep runtime bounded after lr=0.0004 completed."})
    write_markdown(DIAG_ROOT / "lr_sweep_report.md", "\n".join(["# LR Sweep Report", "", f"- Best mean AP: `{selection['best_mean_ap_model']}`", f"- Best balanced checkpoint: `{selection['best_balanced_checkpoint']}`", "- lr=0.0002 skipped to keep runtime bounded."]))
    write_json(DIAG_ROOT / "holdout_eval_attempts.json", {"status": "completed", "attempts": attempts})
    make_figures(ap_rows, diag_rows, lr_rows, selection)
    generate_report(ap_rows, diag_rows, config_payload, selection, lr_rows)
    update_materials()
    print(json.dumps({"status": "completed", "diag_root": str(DIAG_ROOT), "best_balanced": selection["best_balanced_checkpoint"]}, indent=2))


if __name__ == "__main__":
    main()
