from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.deployment_acceptance import compute_prediction_health, prediction_subset_stats
from runtime.lidar_system_algorithm.report_schema import ensure_dir, read_json_or_default, write_csv, write_json, write_markdown


REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"
SPLIT_ROOT = REPORT_ROOT / "expanded_splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deployment diagnostics comparison on fixed holdout predictions.")
    parser.add_argument("--holdout-split", default="holdout_eval_500")
    return parser.parse_args()


def split_ids(name: str) -> list[str]:
    path = SPLIT_ROOT / f"{name}.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_holdout_eval_rows() -> list[dict[str, Any]]:
    csv_path = REPORT_ROOT / "expanded_holdout_eval_comparison.csv"
    rows = []
    import csv

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def eval_payload_for(model_name: str) -> dict[str, Any]:
    comparison = read_json_or_default(REPORT_ROOT / "expanded_holdout_eval_comparison.json", {})
    attempts = comparison.get("attempts", []) if isinstance(comparison, dict) else []
    for attempt in attempts:
        if attempt.get("model_name") == model_name:
            return attempt.get("payload", {})
    return {}


def normalize_hist(hist: list[float]) -> list[float]:
    arr = np.asarray(hist, dtype=np.float64)
    if arr.sum() <= 0:
        return arr.tolist()
    return (arr / arr.sum()).tolist()


def main() -> None:
    args = parse_args()
    sample_ids = split_ids(args.holdout_split)
    baseline_name = "pretrained_baseline"
    eval_rows = load_holdout_eval_rows()
    baseline_payload = eval_payload_for(baseline_name)
    baseline_stats = prediction_subset_stats(Path(baseline_payload["prediction_dir"]), sample_ids)

    rows = []
    payload_rows = []
    for row in eval_rows:
        model_name = row["model_name"]
        payload = eval_payload_for(model_name)
        pred_dir = Path(payload["prediction_dir"])
        stats = prediction_subset_stats(pred_dir, sample_ids)
        drift = compute_prediction_health(stats, baseline_stats)
        payload_row = {
            "model_name": model_name,
            "prediction_dir": str(pred_dir),
            "checkpoint_path": row["checkpoint_path"],
            "total_boxes": stats["total_box_count"],
            "mean_boxes_per_frame": stats["total_box_count"] / max(len(sample_ids), 1),
            "empty_prediction_frames": stats["empty_prediction_file_count"],
            "invalid_geometry_boxes": stats["invalid_geometry_count"],
            "score_mean": stats["score_summary"]["mean"],
            "score_p50": stats["score_summary"]["p50"],
            "score_p95": stats["score_summary"]["p95"],
            "car_box_count": stats["per_class_box_count"]["Car"],
            "ped_box_count": stats["per_class_box_count"]["Pedestrian"],
            "cyc_box_count": stats["per_class_box_count"]["Cyclist"],
            "range_hist_norm": normalize_hist(stats["histograms"]["range"]),
            "class_hist_norm": normalize_hist(stats["histograms"]["class"]),
            "prediction_count_drift": drift["prediction_count_drift"],
            "score_distribution_drift": drift["score_distribution_drift"],
            "class_distribution_drift": drift["class_distribution_drift"],
            "range_distribution_drift": drift["range_distribution_drift"],
            "invalid_geometry_rate": drift["invalid_geometry_rate"],
            "empty_prediction_rate": drift["empty_prediction_rate"],
            "mean_ap_3d_moderate": float(row["mean_ap_3d_moderate"]) if row["mean_ap_3d_moderate"] else None,
        }
        payload_rows.append(payload_row)
        rows.append(
            {
                "model_name": model_name,
                "total_boxes": payload_row["total_boxes"],
                "mean_boxes_per_frame": payload_row["mean_boxes_per_frame"],
                "empty_prediction_frames": payload_row["empty_prediction_frames"],
                "invalid_geometry_boxes": payload_row["invalid_geometry_boxes"],
                "score_mean": payload_row["score_mean"],
                "score_p50": payload_row["score_p50"],
                "score_p95": payload_row["score_p95"],
                "car_box_count": payload_row["car_box_count"],
                "ped_box_count": payload_row["ped_box_count"],
                "cyc_box_count": payload_row["cyc_box_count"],
                "prediction_count_drift": payload_row["prediction_count_drift"],
                "score_distribution_drift": payload_row["score_distribution_drift"],
                "class_distribution_drift": payload_row["class_distribution_drift"],
                "range_distribution_drift": payload_row["range_distribution_drift"],
                "mean_ap_3d_moderate": payload_row["mean_ap_3d_moderate"],
            }
        )

    csv_path = REPORT_ROOT / "expanded_deployment_diagnostics.csv"
    json_path = REPORT_ROOT / "expanded_deployment_diagnostics.json"
    md_path = REPORT_ROOT / "expanded_deployment_diagnostics.md"
    fields = list(rows[0].keys()) if rows else []
    if rows:
        write_csv(csv_path, rows, fields)
    payload = {
        "status": "completed",
        "holdout_split": args.holdout_split,
        "frame_count": len(sample_ids),
        "baseline_model": baseline_name,
        "rows": payload_rows,
        "analysis": {
            "box_count_interpretation": "Box count increase may indicate higher recall, but may also increase false positive risk. It must be interpreted together with holdout AP and distribution drift.",
            "score_interpretation": "A lower score distribution can indicate a less confident detector; an unusually high score distribution can also signal calibration drift or overconfident outputs.",
            "range_interpretation": "Range distribution drift shows whether detections move toward near-field or far-field concentration.",
        },
    }
    write_json(json_path, payload)

    baseline_row = next((item for item in payload_rows if item["model_name"] == baseline_name), None)
    subset_row = next((item for item in payload_rows if item["model_name"] == "subset_finetune_200_50_epoch3"), None)
    expanded_row = next((item for item in payload_rows if item["model_name"] == "expanded_finetune_1000_200_epoch3"), None)
    lines = [
        "# Expanded Deployment Diagnostics",
        "",
        f"- Holdout split: `{args.holdout_split}`",
        f"- Frame count: `{len(sample_ids)}`",
        "",
        "## Comparison",
        "",
    ]
    for item in payload_rows:
        lines.extend(
            [
                f"### {item['model_name']}",
                "",
                f"- total boxes: `{item['total_boxes']}`",
                f"- mean boxes per frame: `{item['mean_boxes_per_frame']:.3f}`",
                f"- empty prediction frames: `{item['empty_prediction_frames']}`",
                f"- invalid geometry boxes: `{item['invalid_geometry_boxes']}`",
                f"- score mean/p50/p95: `{item['score_mean']:.4f}` / `{item['score_p50']:.4f}` / `{item['score_p95']:.4f}`",
                f"- class counts: `Car={item['car_box_count']}, Pedestrian={item['ped_box_count']}, Cyclist={item['cyc_box_count']}`",
                f"- drifts vs baseline: `count={item['prediction_count_drift']:.4f}, score={item['score_distribution_drift']:.4f}, class={item['class_distribution_drift']:.4f}, range={item['range_distribution_drift']:.4f}`",
                "",
            ]
        )
    if expanded_row and baseline_row:
        box_delta = expanded_row["total_boxes"] - baseline_row["total_boxes"]
        ap_delta = (expanded_row["mean_ap_3d_moderate"] or 0.0) - (baseline_row["mean_ap_3d_moderate"] or 0.0)
        risk_note = "Box count increased while mean AP dropped; this is a concrete false-positive risk signal." if box_delta > 0 and ap_delta < 0 else "Box count/AP change is mixed and should not be simplified into a direct performance claim."
        lines.extend(
            [
                "## Risk Interpretation",
                "",
                f"- expanded vs baseline total box delta: `{box_delta}`",
                f"- expanded vs baseline mean AP delta: `{ap_delta:.4f}`",
                f"- interpretation: {risk_note}",
                "",
            ]
        )
    write_markdown(md_path, "\n".join(lines))
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
