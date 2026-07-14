from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import ensure_dir, read_json_or_default, write_csv, write_json, write_markdown


REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"
SPLIT_ROOT = REPORT_ROOT / "expanded_splits"
PRETRAINED_CKPT = Path(r"checkpoints/pointpillar_kitti.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed holdout eval comparison for pretrained/current/expanded PointPillars checkpoints.")
    parser.add_argument("--holdout-split", default="holdout_eval_500")
    parser.add_argument("--also-run-holdout-1000", action="store_true")
    return parser.parse_args()


def latest_exp(prefix: str) -> Path | None:
    candidates = sorted(REPORT_ROOT.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def count_split(name: str) -> int:
    path = SPLIT_ROOT / f"{name}.txt"
    if not path.exists():
        return 0
    return len([line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def run_eval(model_name: str, cfg_file: Path, ckpt: Path, split_name: str, output_dir: Path, pred_dir: Path) -> dict[str, Any]:
    split_file = SPLIT_ROOT / f"{split_name}.txt"
    command = [
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
        str(split_file),
        "--output-dir",
        str(output_dir),
        "--pred-dir",
        str(pred_dir),
        "--max-frames",
        "0",
    ]
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=14400, check=False)
    payload = read_json_or_default(output_dir / "kitti_official_eval.json", {})
    status = "completed" if completed.returncode == 0 and isinstance(payload, dict) and payload.get("status") == "completed" else "failed"
    return {
        "model_name": model_name,
        "status": status,
        "command": " ".join(command),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "payload": payload,
    }


def metric(payload: dict[str, Any], key: str) -> float | None:
    result = payload.get("official_result_dict", {}) if isinstance(payload, dict) else {}
    value = result.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def build_row(model_name: str, checkpoint_path: str, train_samples: int, val_samples: int, holdout_samples: int, epochs: int, payload: dict[str, Any], notes: str) -> dict[str, Any]:
    vals = [metric(payload, "Car_3d/moderate_R40"), metric(payload, "Pedestrian_3d/moderate_R40"), metric(payload, "Cyclist_3d/moderate_R40")]
    numeric = [v for v in vals if isinstance(v, (int, float))]
    return {
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "holdout_samples": holdout_samples,
        "epochs": epochs,
        "car_ap_3d_easy": metric(payload, "Car_3d/easy_R40"),
        "car_ap_3d_moderate": metric(payload, "Car_3d/moderate_R40"),
        "car_ap_3d_hard": metric(payload, "Car_3d/hard_R40"),
        "ped_ap_3d_easy": metric(payload, "Pedestrian_3d/easy_R40"),
        "ped_ap_3d_moderate": metric(payload, "Pedestrian_3d/moderate_R40"),
        "ped_ap_3d_hard": metric(payload, "Pedestrian_3d/hard_R40"),
        "cyc_ap_3d_easy": metric(payload, "Cyclist_3d/easy_R40"),
        "cyc_ap_3d_moderate": metric(payload, "Cyclist_3d/moderate_R40"),
        "cyc_ap_3d_hard": metric(payload, "Cyclist_3d/hard_R40"),
        "mean_ap_3d_moderate": sum(numeric) / len(numeric) if numeric else None,
        "eval_scope": f"{holdout_samples}-frame-holdout",
        "full_val": False,
        "notes": notes,
    }


def main() -> None:
    holdout_split = parse_args().holdout_split
    holdout_samples = count_split(holdout_split)
    output_root = ensure_dir(REPORT_ROOT / f"{holdout_split}_comparison")

    subset_dir = latest_exp("subset_finetune")
    expanded_dir = latest_exp("expanded_finetune_1000")
    if subset_dir is None or expanded_dir is None:
        raise SystemExit("Missing subset_finetune or expanded_finetune_1000 experiment.")

    subset_status = json.loads((subset_dir / "train_status.json").read_text(encoding="utf-8"))
    expanded_status = json.loads((expanded_dir / "train_status.json").read_text(encoding="utf-8"))

    models = [
        {
            "model_name": "pretrained_baseline",
            "cfg_file": expanded_dir / "run_config.yaml",
            "ckpt": PRETRAINED_CKPT,
            "train_samples": 0,
            "val_samples": 0,
            "epochs": 0,
            "notes": "Public pretrained PointPillars checkpoint on fixed holdout split.",
        },
        {
            "model_name": "subset_finetune_200_50_epoch3",
            "cfg_file": subset_dir / "run_config.yaml",
            "ckpt": Path(subset_status["trained_checkpoint"]),
            "train_samples": int(subset_status["train_sample_count"]),
            "val_samples": int(subset_status["val_sample_count"]),
            "epochs": int(subset_status["epochs"]),
            "notes": "Current 200/50 subset_finetune checkpoint evaluated on common holdout split.",
        },
        {
            "model_name": "expanded_finetune_1000_200_epoch3",
            "cfg_file": expanded_dir / "run_config.yaml",
            "ckpt": Path(expanded_status["trained_checkpoint"]),
            "train_samples": int(expanded_status["train_sample_count"]),
            "val_samples": int(expanded_status["val_sample_count"]),
            "epochs": int(expanded_status["epochs"]),
            "notes": "Expanded 1000/200 fine-tuned checkpoint evaluated on common holdout split.",
        },
    ]

    attempts = []
    rows = []
    for model in models:
        eval_dir = ensure_dir(output_root / model["model_name"])
        result = run_eval(
            model_name=model["model_name"],
            cfg_file=model["cfg_file"],
            ckpt=model["ckpt"],
            split_name=holdout_split,
            output_dir=eval_dir,
            pred_dir=PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "results" / f"{model['model_name']}_{holdout_split}_txt",
        )
        attempts.append(result)
        if result["status"] == "completed":
            rows.append(
                build_row(
                    model_name=model["model_name"],
                    checkpoint_path=str(model["ckpt"]),
                    train_samples=model["train_samples"],
                    val_samples=model["val_samples"],
                    holdout_samples=holdout_samples,
                    epochs=model["epochs"],
                    payload=result["payload"],
                    notes=model["notes"],
                )
            )

    csv_path = REPORT_ROOT / "expanded_holdout_eval_comparison.csv"
    json_path = REPORT_ROOT / "expanded_holdout_eval_comparison.json"
    md_path = REPORT_ROOT / "expanded_holdout_eval_comparison.md"
    fields = [
        "model_name",
        "checkpoint_path",
        "train_samples",
        "val_samples",
        "holdout_samples",
        "epochs",
        "car_ap_3d_easy",
        "car_ap_3d_moderate",
        "car_ap_3d_hard",
        "ped_ap_3d_easy",
        "ped_ap_3d_moderate",
        "ped_ap_3d_hard",
        "cyc_ap_3d_easy",
        "cyc_ap_3d_moderate",
        "cyc_ap_3d_hard",
        "mean_ap_3d_moderate",
        "eval_scope",
        "full_val",
        "notes",
    ]
    if rows:
        write_csv(csv_path, rows, fields)
    payload = {
        "status": "completed" if len(rows) == len(models) else "partial",
        "holdout_split": holdout_split,
        "holdout_samples": holdout_samples,
        "holdout_sample_count": holdout_samples,
        "frame_count": holdout_samples,
        "full_val": False,
        "attempts": attempts,
        "rows": rows,
        "mean_ap_3d_moderate": {row["model_name"]: row["mean_ap_3d_moderate"] for row in rows},
        "skipped_holdout_eval_1000_reason": "Not run in this turn to keep runtime bounded after expanded_finetune_1000 training and common holdout_500 comparison.",
    }
    write_json(json_path, payload)

    lines = [
        "# Expanded Holdout Eval Comparison",
        "",
        f"- Holdout split: `{holdout_split}`",
        f"- Holdout samples: `{holdout_samples}`",
        "- full_val: `false`",
        "- 50-frame subset-val AP is not mixed into this table; this file is holdout-only.",
        "",
        "## Moderate 3D AP_R40",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['model_name']}",
                "",
                f"- Car: `{row['car_ap_3d_moderate']:.4f}`",
                f"- Pedestrian: `{row['ped_ap_3d_moderate']:.4f}`",
                f"- Cyclist: `{row['cyc_ap_3d_moderate']:.4f}`",
                f"- Mean: `{row['mean_ap_3d_moderate']:.4f}`",
                f"- Notes: {row['notes']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "- This comparison uses a fixed holdout split, not full KITTI val.",
            "- The pretrained / current subset_finetune / expanded_finetune checkpoints all use the same holdout split.",
            "- holdout_eval_1000 split was generated but not evaluated in this turn to keep runtime bounded.",
        ]
    )
    write_markdown(md_path, "\n".join(lines))
    print(json.dumps({"status": payload["status"], "csv": str(csv_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
