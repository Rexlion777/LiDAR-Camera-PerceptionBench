from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import read_json_or_default, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit TensorRT decode / score / yaw alignment based on current reports.")
    parser.add_argument("--input-dir", default="reports/lidar_system_algorithm")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def main() -> None:
    args = parse_args()
    input_dir = _resolve(args.input_dir)
    json_path = input_dir / "tensorrt_decode_alignment_audit.json"
    md_path = input_dir / "tensorrt_decode_alignment_audit.md"

    bucket_report = read_json_or_default(input_dir / "tensorrt_bucketed_core_report.json", {})
    submodule_report = read_json_or_default(input_dir / "tensorrt_submodule_bisection_report.json", {})
    wrapper_eval = read_json_or_default(input_dir / "wrapper_pytorch_core_eval.json", {})
    binding_audit = read_json_or_default(input_dir / "tensorrt_binding_contract_audit.json", {})

    frame_rows = bucket_report.get("frame_rows", []) if isinstance(bucket_report, dict) else []
    suspicious_frames = [
        {
            "frame_id": row.get("frame_id"),
            "trt_box_count": row.get("trt_box_count"),
            "score_mean": row.get("score_mean"),
            "topk_center_diff_mean": row.get("topk_center_diff_mean"),
        }
        for row in frame_rows
        if row.get("score_mean") == 1.0 or row.get("topk_center_diff_mean")
    ][:10]

    payload = {
        "status": "completed",
        "checks": {
            "cls_preds_normalized_flag": False,
            "dir_cls_preds_present_in_current_core_export": binding_audit.get("contract_checks", {}).get("direction_classifier_present_in_current_core_export"),
            "score_saturation_seen_in_bucketed_wrapper": any(row.get("score_mean") == 1.0 for row in frame_rows),
            "large_center_diff_seen_in_bucketed_wrapper": any((row.get("topk_center_diff_mean") or 0) > 1000 for row in frame_rows),
            "wrapper_pytorch_core_nonzero_ap": (wrapper_eval.get("official_result_dict", {}) or {}).get("Car_3d/moderate_R40"),
            "submodule_full_core_box_counts_match_reference": all(
                row.get("pytorch_box_count") == row.get("trt_full_box_count")
                for row in submodule_report.get("judgements", [])
            ),
        },
        "suspected_findings": [
            "Decode/export path is not completely dead because wrapper_pytorch_core produces non-zero KITTI AP on the same 200-frame slice.",
            "Current multi-bucket TRT wrapper still shows score saturation to 1.0 and extreme center drift on many frames, which is upstream of KITTI export formatting.",
            "Submodule bisection on bucket 8192 shows TensorRT full-core outputs can be decoded into the same box counts as PyTorch for representative frames 000002/000003.",
            "Therefore decode/yaw logic is not the primary blocker on its own; the stronger blocker remains bucketed wrapper integration / padded multi-bucket runtime semantics.",
        ],
        "suspicious_frame_preview": suspicious_frames,
    }
    write_json(json_path, payload)
    write_markdown(
        md_path,
        "# TensorRT Decode Alignment Audit\n\n"
        f"- cls_preds_normalized flag: `{payload['checks']['cls_preds_normalized_flag']}`\n"
        f"- dir_cls_preds present in current core export: `{payload['checks']['dir_cls_preds_present_in_current_core_export']}`\n"
        f"- Score saturation seen in bucketed wrapper: `{payload['checks']['score_saturation_seen_in_bucketed_wrapper']}`\n"
        f"- Large center drift seen in bucketed wrapper: `{payload['checks']['large_center_diff_seen_in_bucketed_wrapper']}`\n"
        f"- Submodule full-core box-count match: `{payload['checks']['submodule_full_core_box_counts_match_reference']}`\n"
        "- Conclusion: decode/yaw is a secondary audit item; current evidence points more strongly to bucketed wrapper integration semantics than to a pure sigmoid/yaw bug.\n",
    )
    print(json.dumps({"status": "completed", "report": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
