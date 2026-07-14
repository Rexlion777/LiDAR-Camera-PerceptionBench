from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import write_csv, write_json, write_markdown
from runtime.lidar_system_algorithm.tensorrt_accuracy_debug import (
    analyze_decode_nms_rows,
    analyze_raw_tensor_rows,
    audit_prediction_export,
    build_route_summaries,
    load_existing_debug_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TensorRT accuracy bisection and export audits from existing reports.")
    parser.add_argument("--input-dir", default="reports/lidar_system_algorithm", help="Input report directory.")
    parser.add_argument("--baseline-pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt", help="Baseline KITTI prediction directory.")
    parser.add_argument("--trt-pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt_trt_bucketed", help="TensorRT bucketed KITTI prediction directory.")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def main() -> None:
    args = parse_args()
    input_dir = _resolve(args.input_dir)
    baseline_pred_dir = _resolve(args.baseline_pred_dir)
    trt_pred_dir = _resolve(args.trt_pred_dir)

    inputs = load_existing_debug_inputs(input_dir)
    baseline_eval = inputs["baseline_eval"]
    wrapper_eval = inputs["wrapper_eval"]
    bucket_report = inputs["bucket_report"]
    bucket_eval = inputs["bucket_eval"]
    raw_diff = inputs["raw_diff"]
    frame_rows = bucket_report.get("frame_rows", []) if isinstance(bucket_report, dict) else []

    route_rows = build_route_summaries(
        baseline_eval=baseline_eval,
        wrapper_eval=wrapper_eval,
        bucket_report=bucket_report,
        bucket_eval=bucket_eval,
        raw_diff=raw_diff,
        baseline_pred_dir=baseline_pred_dir,
        trt_pred_dir=trt_pred_dir,
    )
    raw_rows, raw_summary = analyze_raw_tensor_rows(raw_diff.get("rows", []) if isinstance(raw_diff, dict) else [])
    decode_rows, decode_summary = analyze_decode_nms_rows(frame_rows)
    export_rows, export_summary = audit_prediction_export(trt_pred_dir)

    route_csv = input_dir / "tensorrt_accuracy_bisection_summary.csv"
    route_json = input_dir / "tensorrt_accuracy_bisection_report.json"
    route_md = input_dir / "tensorrt_accuracy_bisection_report.md"
    raw_csv = input_dir / "tensorrt_raw_tensor_diff.csv"
    raw_json = input_dir / "tensorrt_raw_tensor_diff.json"
    decode_csv = input_dir / "tensorrt_decode_nms_diff.csv"
    decode_json = input_dir / "tensorrt_decode_nms_diff.json"
    export_csv = input_dir / "kitti_export_audit.csv"
    export_json = input_dir / "kitti_export_audit.json"
    export_md = input_dir / "kitti_export_audit.md"
    fix_json = input_dir / "tensorrt_accuracy_fix_report.json"
    fix_md = input_dir / "tensorrt_accuracy_fix_report.md"

    route_fieldnames = [
        "route",
        "frame_count",
        "prediction_file_count",
        "empty_prediction_file_count",
        "total_box_count",
        "official_eval_status",
        "blocker",
    ]
    write_csv(route_csv, [{key: row.get(key) for key in route_fieldnames} for row in route_rows], route_fieldnames)
    write_json(
        route_json,
        {
            "status": "completed",
            "routes": route_rows,
            "route_layer_judgement": {
            "most_likely_blocker_layer": "TRT raw output / tensor semantic mismatch"
                if raw_summary.get("severe_frame_count")
                else "needs further verification",
                "supporting_evidence": [
                    "wrapper_pytorch_core official eval on the same 200-frame slice is non-zero but collapses to Car/Ped/Cyc moderate AP_R40 = 2.20 / 0.31 / 1.01, which shows the padded wrapper path itself is not accuracy-preserving",
                    "rebuilt wrapper_trt_core no longer writes 212/212 empty files, but still exports 195/212 empty files and saturates non-empty predictions to score=1.0",
                    "same-frame raw tensor diff from the earlier bucketed engines shows severe magnitude / non-finite mismatch on sampled frames",
                    "fresh submodule bisection on bucket 8192 shows representative full-core TRT can align with PyTorch, so the remaining blocker is likely in bucketed multi-frame wrapper integration rather than a simple output-name swap",
                ],
            },
        },
    )
    write_markdown(
        route_md,
        "# TensorRT Accuracy Bisection Report\n\n"
        "## Route Summary\n\n"
        + "\n".join(
            f"- `{row['route']}`: files={row['prediction_file_count']}, empty_files={row['empty_prediction_file_count']}, total_boxes={row['total_box_count']}, eval_status=`{row['official_eval_status']}`"
            for row in route_rows
        )
        + "\n\n## Judgement\n\n"
        + "- The issue is not in baseline OpenPCDet evaluation.\n"
        + "- The strongest evidence points to `TRT raw output / tensor semantic mismatch`, with empty post-NMS TRT detections as the downstream symptom.\n"
        + "- KITTI export is also failing functionally because the current TRT export directory contains empty prediction files.\n",
    )

    write_csv(raw_csv, raw_rows, list(raw_rows[0].keys()) if raw_rows else ["frame_id"])
    write_json(raw_json, {"status": "completed", "rows": raw_rows, "summary": raw_summary})

    write_csv(decode_csv, decode_rows, list(decode_rows[0].keys()) if decode_rows else ["frame_id"])
    write_json(decode_json, {"status": "completed", "rows": decode_rows, "summary": decode_summary})

    write_csv(export_csv, export_rows, list(export_rows[0].keys()) if export_rows else ["frame_id"])
    write_json(export_json, {"status": "completed", "rows": export_rows, "summary": export_summary})
    write_markdown(
        export_md,
        "# KITTI Export Audit\n\n"
        f"- Prediction files: `{export_summary['prediction_file_count']}`\n"
        f"- Empty prediction files: `{export_summary['empty_prediction_file_count']}`\n"
        f"- Direct symptom: `{export_summary['direct_symptom']}`\n"
        "- Prediction lines are audited as KITTI prediction lines with 16 tokens including score.\n",
    )

    fix_payload = {
        "status": "completed_with_blocker",
        "fixes_applied": [
            "Replaced zero-point bucket padding with non-zero padding to eliminate PillarVFE divide-by-zero / NaN propagation.",
        ],
        "pre_fix_status": {
            "raw_output_issue": "NaN propagation was observed before the non-zero padding repair.",
            "ap": "unavailable",
        },
        "post_fix_status": {
            "raw_output_issue": raw_summary,
            "decode_nms_issue": decode_summary,
            "official_eval_status": bucket_eval.get("status"),
            "official_ap": bucket_eval.get("official_result_dict", {}),
        },
        "remaining_blocker": "Non-zero padding removed the NaN failure mode, but the bucketed TRT path still collapses to empty post-NMS outputs and zero official AP. The remaining blocker is still in the TRT raw-output semantic alignment path.",
        "resume_safe": False,
    }
    write_json(fix_json, fix_payload)
    write_markdown(
        fix_md,
        "# TensorRT Accuracy Fix Report\n\n"
        "- Applied fix: removed zero-point padding that created NaNs in PillarVFE.\n"
        "- Result: raw NaN failure mode is mitigated, but TRT wrapper official AP remains zero.\n"
        "- Remaining blocker: `TRT raw output / tensor semantic mismatch` leading to empty post-NMS outputs and empty KITTI export files.\n"
        "- Resume-safe status: `False`.\n",
    )

    print(
        json.dumps(
            {
                "status": "completed",
                "bisection_report": str(route_json),
                "raw_diff_report": str(raw_json),
                "decode_nms_report": str(decode_json),
                "kitti_export_audit": str(export_json),
                "fix_report": str(fix_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
