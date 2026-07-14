from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import write_csv, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scatter/padding strategy experiment for bucketed TensorRT PointPillars.")
    parser.add_argument("--bucket-size", type=int, default=8192)
    parser.add_argument("--frame-ids", default="000002,000003")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _run_subprocess(args_list: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(args_list, capture_output=True, text=True, cwd=str(PROJECT_ROOT), check=False)
    return completed.returncode, completed.stdout, completed.stderr


def main() -> None:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tensorrt_scatter_padding_experiment.json"
    md_path = output_dir / "tensorrt_scatter_padding_experiment.md"
    csv_path = output_dir / "tensorrt_scatter_padding_experiment.csv"

    experiments = [
        {"name": "repeat_first_valid", "padding_strategy": "repeat_first_valid", "bucket_size": args.bucket_size, "frame_ids": args.frame_ids},
        {"name": "duplicate_zero_coord_padding", "padding_strategy": "duplicate_zero_coord_padding", "bucket_size": args.bucket_size, "frame_ids": args.frame_ids},
        {"name": "unique_dummy_coord_padding", "padding_strategy": "unique_dummy_coord_padding", "bucket_size": args.bucket_size, "frame_ids": args.frame_ids},
    ]

    results = []
    script_path = PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "debug_tensorrt_submodule_bisection.py"
    for experiment in experiments:
        prefix = f"padding_exp_{experiment['name']}"
        cmd = [
            sys.executable,
            str(script_path),
            "--bucket-size",
            str(experiment["bucket_size"]),
            "--frame-ids",
            experiment["frame_ids"],
            "--padding-strategy",
            experiment["padding_strategy"],
            "--output-dir",
            str(output_dir),
            "--report-prefix",
            prefix,
        ]
        code, stdout, stderr = _run_subprocess(cmd)
        report_path = output_dir / f"{prefix}_report.json"
        payload = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        judgement_rows = payload.get("judgements", [])
        results.append(
            {
                "strategy": experiment["name"],
                "status": payload.get("status", "failed" if code else "completed"),
                "bucket_size": experiment["bucket_size"],
                "frame_count": len(judgement_rows),
                "real_pillar_count_mean": sum(row.get("real_pillar_count", 0) for row in judgement_rows) / len(judgement_rows) if judgement_rows else None,
                "padded_pillar_count": sum(row.get("padded_pillar_count", 0) for row in judgement_rows) / len(judgement_rows) if judgement_rows else None,
                "duplicate_coord_count": None,
                "dummy_coord_collision_count": None,
                "vfe_diff_mean": payload.get("stage_mean_abs_diff", {}).get("trt_vfe_only"),
                "scatter_diff_mean": payload.get("stage_mean_abs_diff", {}).get("trt_vfe_scatter"),
                "backbone_head_cls_diff_mean": payload.get("stage_mean_abs_diff", {}).get("trt_backbone_head_from_pytorch_scatter_cls"),
                "full_core_cls_diff_mean": payload.get("stage_mean_abs_diff", {}).get("trt_full_core_cls"),
                "decoded_box_count_mean": sum(row.get("trt_full_box_count", 0) for row in judgement_rows) / len(judgement_rows) if judgement_rows else None,
                "stderr": stderr.strip()[:500],
            }
        )

    results.append(
        {
            "strategy": "valid_masked_padding",
            "status": "skipped",
            "bucket_size": args.bucket_size,
            "frame_count": 0,
            "real_pillar_count_mean": None,
            "padded_pillar_count": None,
            "duplicate_coord_count": None,
            "dummy_coord_collision_count": None,
            "vfe_diff_mean": None,
            "scatter_diff_mean": None,
            "backbone_head_cls_diff_mean": None,
            "full_core_cls_diff_mean": None,
            "decoded_box_count_mean": None,
            "stderr": "Current exported PointPillars graph does not expose an explicit valid-mask contract before scatter.",
        }
    )

    results.append(
        {
            "strategy": "exact_small_frame_no_padding",
            "status": "skipped",
            "bucket_size": None,
            "frame_count": 0,
            "real_pillar_count_mean": None,
            "padded_pillar_count": None,
            "duplicate_coord_count": None,
            "dummy_coord_collision_count": None,
            "vfe_diff_mean": None,
            "scatter_diff_mean": None,
            "backbone_head_cls_diff_mean": None,
            "full_core_cls_diff_mean": None,
            "decoded_box_count_mean": None,
            "stderr": "Exact-shape no-padding engine not built in this round; focus stayed on bucketed padding semantics.",
        }
    )

    write_csv(csv_path, results, list(results[0].keys()) if results else ["strategy"])
    payload = {"status": "completed", "bucket_size": args.bucket_size, "frame_ids": args.frame_ids, "results": results}
    write_json(json_path, payload)
    write_markdown(
        md_path,
        "# TensorRT Scatter Padding Experiment\n\n"
        + "\n".join(
            f"- `{row['strategy']}`: status=`{row['status']}`, scatter_diff_mean=`{row['scatter_diff_mean']}`, full_core_cls_diff_mean=`{row['full_core_cls_diff_mean']}`, decoded_box_count_mean=`{row['decoded_box_count_mean']}`"
            for row in results
        )
        + "\n",
    )
    print(json.dumps({"status": "completed", "report": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
