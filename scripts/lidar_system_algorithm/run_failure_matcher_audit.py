from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import write_json, write_markdown
from runtime.lidar_system_algorithm.tensorrt_accuracy_debug import audit_failure_matcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit KITTI failure matcher assumptions against current prediction directory.")
    parser.add_argument("--kitti-root", default="data/kitti_object_raw/extracted", help="KITTI root directory.")
    parser.add_argument("--pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt", help="Prediction directory to audit.")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm", help="Output directory.")
    parser.add_argument("--score-thresholds", default="0.0,0.1,0.3,0.5,0.7", help="Comma-separated score thresholds.")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def main() -> None:
    args = parse_args()
    kitti_root = _resolve(args.kitti_root)
    pred_dir = _resolve(args.pred_dir)
    output_dir = _resolve(args.output_dir)
    thresholds = [float(value.strip()) for value in args.score_thresholds.split(",") if value.strip()]

    payload = audit_failure_matcher(
        label_dir=kitti_root / "training" / "label_2",
        pred_dir=pred_dir,
        thresholds=thresholds,
    )
    json_path = output_dir / "failure_matcher_audit.json"
    md_path = output_dir / "failure_matcher_audit.md"
    write_json(json_path, payload)
    write_markdown(
        md_path,
        "# Failure Matcher Audit\n\n"
        f"- Status: `{payload['status']}`\n"
        f"- Frame count: `{payload['frame_count']}`\n"
        f"- Duplicate-like FP rate: `{payload['duplicate_prediction_rate']}`\n"
        f"- DontCare lines observed in GT: `{payload['unsupported_gt_filtering']['dontcare_line_count']}`\n"
        "- The matcher remains an analysis tool and does not claim equivalence to the KITTI official evaluator.\n",
    )
    print(json.dumps({"status": "completed", "failure_matcher_audit": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
