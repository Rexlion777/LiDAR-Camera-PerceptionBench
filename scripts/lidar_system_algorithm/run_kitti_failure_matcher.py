from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.failure_matcher import (
    SUPPORTED_CLASSES,
    as_detection_box,
    difficulty_of,
    distance_bin,
    greedy_match_objects,
    official_eval_filter_status,
    read_kitti_objects,
    score_bin,
    size_bin,
)
from runtime.lidar_system_algorithm.kitti_io import locate_default_kitti_root, read_kitti_bin
from runtime.lidar_system_algorithm.report_schema import write_csv, write_json, write_markdown
from runtime.lidar_system_algorithm.visualization import compose_grid, draw_bar_chart, render_bev, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KITTI GT/prediction failure matcher for analysis.")
    parser.add_argument("--kitti-root", default="", help="KITTI root directory.")
    parser.add_argument("--pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt", help="KITTI prediction txt directory.")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm", help="Output directory.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap for analysis.")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _frame_ids(label_dir: Path, pred_dir: Path, max_frames: int) -> list[str]:
    ids = sorted({path.stem for path in label_dir.glob("*.txt")} | {path.stem for path in pred_dir.glob("*.txt")})
    return ids[:max_frames] if max_frames > 0 else ids


def _inc(table: dict[str, dict[str, int]], bucket: str, key: str) -> None:
    row = table.setdefault(bucket, {"tp": 0, "fp": 0, "fn": 0})
    row[key] += 1


def main() -> None:
    args = parse_args()
    kitti_root = _resolve(args.kitti_root) if args.kitti_root else locate_default_kitti_root().resolve()
    pred_dir = _resolve(args.pred_dir)
    output_dir = _resolve(args.output_dir)
    figures_dir = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    label_dir = kitti_root / "training" / "label_2"
    velodyne_dir = kitti_root / "training" / "velodyne"
    summary_csv = output_dir / "failure_matcher_summary.csv"
    summary_json = output_dir / "failure_matcher_summary.json"
    by_range_csv = output_dir / "failure_cases_by_range.csv"
    by_class_csv = output_dir / "failure_cases_by_class.csv"
    report_md = output_dir / "failure_analysis_report.md"
    report_json = output_dir / "failure_analysis_report.json"
    fp_fn_range_fig = figures_dir / "fp_fn_by_range.png"
    fp_fn_class_fig = figures_dir / "fp_fn_by_class.png"
    gallery_fig = figures_dir / "failure_case_gallery.png"

    if not pred_dir.exists():
        payload = {"status": "skipped", "reason": f"Prediction directory not found: {pred_dir}"}
        write_json(summary_json, payload)
        write_markdown(report_md, f"# Failure Analysis Report\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
        print(json.dumps(payload, indent=2))
        return

    frame_ids = _frame_ids(label_dir, pred_dir, args.max_frames)
    if not frame_ids:
        payload = {"status": "skipped", "reason": "No KITTI label/prediction frames were found for failure analysis."}
        write_json(summary_json, payload)
        write_markdown(report_md, f"# Failure Analysis Report\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
        print(json.dumps(payload, indent=2))
        return

    per_match_rows: list[dict] = []
    by_class: dict[str, dict[str, int]] = {name: {"tp": 0, "fp": 0, "fn": 0} for name in SUPPORTED_CLASSES}
    by_range: dict[str, dict[str, int]] = {}
    by_difficulty: dict[str, dict[str, int]] = {}
    by_official_filter: dict[str, dict[str, int]] = {}
    by_occlusion: dict[str, dict[str, int]] = {}
    by_truncation: dict[str, dict[str, int]] = {}
    by_size: dict[str, dict[str, int]] = {}
    by_score: dict[str, dict[str, int]] = {}
    gallery_images = []
    gallery_notes = []

    for frame_id in frame_ids:
        gt_objects = read_kitti_objects(label_dir / f"{frame_id}.txt", is_prediction=False)
        pred_objects = read_kitti_objects(pred_dir / f"{frame_id}.txt", is_prediction=True)
        matches, unmatched_preds, unmatched_gt = greedy_match_objects(gt_objects, pred_objects)

        for match in matches:
            gt = match["gt"]
            pred = match["pred"]
            cls = gt.class_name
            by_class[cls]["tp"] += 1
            _inc(by_range, distance_bin(gt.distance_m), "tp")
            _inc(by_difficulty, difficulty_of(gt), "tp")
            for diff_name in ("easy", "moderate", "hard"):
                status = official_eval_filter_status(gt, cls, diff_name)
                if status == 0:
                    _inc(by_official_filter, f"{cls}_{diff_name}_valid", "tp")
                elif status == 1:
                    _inc(by_official_filter, f"{cls}_{diff_name}_ignored", "tp")
            _inc(by_occlusion, f"occ_{gt.occlusion}", "tp")
            _inc(by_truncation, f"trunc_{min(int(gt.truncation * 10) * 10, 90)}pct", "tp")
            _inc(by_size, size_bin(gt), "tp")
            _inc(by_score, score_bin(pred.score), "tp")
            per_match_rows.append(
                {
                    "frame_id": frame_id,
                    "class_name": cls,
                    "match_type": "tp",
                    "bev_iou": match["bev_iou"],
                    "gt_distance_m": gt.distance_m,
                    "pred_score": pred.score,
                    "difficulty": difficulty_of(gt),
                    "range_bin": distance_bin(gt.distance_m),
                    "score_bin": score_bin(pred.score),
                }
            )

        for pred in unmatched_preds:
            cls = pred.class_name
            by_class.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
            _inc(by_range, distance_bin(pred.distance_m), "fp")
            _inc(by_occlusion, "pred_only", "fp")
            _inc(by_truncation, "pred_only", "fp")
            _inc(by_size, size_bin(pred), "fp")
            _inc(by_score, score_bin(pred.score), "fp")
            per_match_rows.append(
                {
                    "frame_id": frame_id,
                    "class_name": cls,
                    "match_type": "fp",
                    "bev_iou": 0.0,
                    "gt_distance_m": "",
                    "pred_distance_m": pred.distance_m,
                    "pred_score": pred.score,
                    "difficulty": "",
                    "range_bin": distance_bin(pred.distance_m),
                    "score_bin": score_bin(pred.score),
                }
            )

        for gt in unmatched_gt:
            cls = gt.class_name
            by_class.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
            _inc(by_range, distance_bin(gt.distance_m), "fn")
            _inc(by_difficulty, difficulty_of(gt), "fn")
            for diff_name in ("easy", "moderate", "hard"):
                status = official_eval_filter_status(gt, cls, diff_name)
                if status == 0:
                    _inc(by_official_filter, f"{cls}_{diff_name}_valid", "fn")
                elif status == 1:
                    _inc(by_official_filter, f"{cls}_{diff_name}_ignored", "fn")
            _inc(by_occlusion, f"occ_{gt.occlusion}", "fn")
            _inc(by_truncation, f"trunc_{min(int(gt.truncation * 10) * 10, 90)}pct", "fn")
            _inc(by_size, size_bin(gt), "fn")
            _inc(by_score, "unmatched_gt", "fn")
            per_match_rows.append(
                {
                    "frame_id": frame_id,
                    "class_name": cls,
                    "match_type": "fn",
                    "bev_iou": 0.0,
                    "gt_distance_m": gt.distance_m,
                    "pred_score": "",
                    "difficulty": difficulty_of(gt),
                    "range_bin": distance_bin(gt.distance_m),
                    "score_bin": "unmatched_gt",
                }
            )

        if len(gallery_images) < 6 and (unmatched_preds or unmatched_gt):
            lidar_path = velodyne_dir / f"{frame_id}.bin"
            if lidar_path.exists():
                points = read_kitti_bin(lidar_path)[:, :3]
            else:
                points = np.zeros((0, 3), dtype=np.float32)
            gt_boxes = [as_detection_box(obj, idx, "dbscan") for idx, obj in enumerate(unmatched_gt[:12])]
            pred_boxes = [as_detection_box(obj, idx, "pointpillars") for idx, obj in enumerate(unmatched_preds[:12])]
            gallery_images.append(
                render_bev(
                    points,
                    boxes=gt_boxes + pred_boxes,
                    title=f"Failure Case {frame_id}",
                    extra_lines=[
                        f"FN={len(unmatched_gt)} FP={len(unmatched_preds)}",
                        "orange=GT FN, green=Pred FP",
                    ],
                )
            )
            gallery_notes.append(f"{frame_id}: FN={len(unmatched_gt)} FP={len(unmatched_preds)}")

    class_rows = [
        {"bucket": cls, "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"]}
        for cls, counts in by_class.items()
    ]
    range_rows = [
        {"bucket": bucket, "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"]}
        for bucket, counts in sorted(by_range.items())
    ]
    write_csv(summary_csv, per_match_rows, sorted({key for row in per_match_rows for key in row.keys()}))
    write_csv(by_class_csv, class_rows, ["bucket", "tp", "fp", "fn"])
    write_csv(by_range_csv, range_rows, ["bucket", "tp", "fp", "fn"])

    save_image(fp_fn_class_fig, draw_bar_chart([{"stage": f"{row['bucket']}_FP", "mean_ms": row["fp"]} for row in class_rows] + [{"stage": f"{row['bucket']}_FN", "mean_ms": row["fn"]} for row in class_rows], title="Failure Analysis by Class"))
    save_image(fp_fn_range_fig, draw_bar_chart([{"stage": f"{row['bucket']}_FP", "mean_ms": row["fp"]} for row in range_rows] + [{"stage": f"{row['bucket']}_FN", "mean_ms": row["fn"]} for row in range_rows], title="Failure Analysis by Range"))
    save_image(gallery_fig, compose_grid(gallery_images, columns=2, label_lines=["GT matcher gallery", *gallery_notes[:4]]))

    totals = {
        "tp": sum(row["tp"] for row in class_rows),
        "fp": sum(row["fp"] for row in class_rows),
        "fn": sum(row["fn"] for row in class_rows),
    }
    summary_payload = {
        "status": "completed",
        "scope": "Greedy BEV IoU matcher for KITTI prediction-vs-GT error attribution. This does not replace the KITTI official evaluator.",
        "frame_count": len(frame_ids),
        "prediction_dir": str(pred_dir),
        "label_dir": str(label_dir),
        "totals": totals,
        "by_class": class_rows,
        "by_range": range_rows,
        "by_difficulty": [{"bucket": bucket, **counts} for bucket, counts in sorted(by_difficulty.items())],
        "by_official_filter": [{"bucket": bucket, **counts} for bucket, counts in sorted(by_official_filter.items())],
        "by_occlusion": [{"bucket": bucket, **counts} for bucket, counts in sorted(by_occlusion.items())],
        "by_truncation": [{"bucket": bucket, **counts} for bucket, counts in sorted(by_truncation.items())],
        "by_size": [{"bucket": bucket, **counts} for bucket, counts in sorted(by_size.items())],
        "by_score_bin": [{"bucket": bucket, **counts} for bucket, counts in sorted(by_score.items())],
        "limitations": [
            "This matcher is an analysis tool and is not equivalent to the KITTI official evaluator.",
            "Official AP remains the source of record for ranking-style metrics.",
            "Current matching is BEV IoU greedy matching by score order.",
            "Difficulty-side summaries now include an official-eval-style filter status approximation for GT objects.",
        ],
        "gallery_figure": str(gallery_fig),
        "class_figure": str(fp_fn_class_fig),
        "range_figure": str(fp_fn_range_fig),
    }
    write_json(summary_json, summary_payload)
    write_json(report_json, summary_payload)
    write_markdown(
        report_md,
        "# Failure Analysis Report\n\n"
        f"- Status: `{summary_payload['status']}`\n"
        f"- Frames analyzed: `{summary_payload['frame_count']}`\n"
        f"- TP / FP / FN: `{totals['tp']}` / `{totals['fp']}` / `{totals['fn']}`\n"
        "\n## Notes\n\n"
        "- This matcher is for error attribution and dashboard analysis.\n"
        "- KITTI official AP is still taken from the native evaluator.\n"
        "- Matching here is BEV IoU greedy matching by score order, not a claim of evaluator equivalence.\n",
    )
    print(f"Saved failure matcher summary: {summary_json}")


if __name__ == "__main__":
    main()
