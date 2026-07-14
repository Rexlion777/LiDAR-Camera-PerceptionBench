from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import ensure_dir, write_csv, write_json, write_markdown


KITTI_ROOT = PROJECT_ROOT / "data" / "kitti_object_raw" / "extracted" / "training"
IMAGESETS = PROJECT_ROOT / "external" / "OpenPCDet" / "data" / "kitti" / "ImageSets"
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune" / "expanded_splits"
SUPPORTED = ("Car", "Pedestrian", "Cyclist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build expanded KITTI train/val/holdout splits with explicit distribution/overlap reports.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_point_count(sample_id: str) -> int:
    velodyne = KITTI_ROOT / "velodyne" / f"{sample_id}.bin"
    if not velodyne.exists():
        return 0
    return int(velodyne.stat().st_size // 16)


def distance_bin(distance_m: float) -> str:
    if distance_m < 20.0:
        return "near"
    if distance_m < 40.0:
        return "mid"
    return "far"


def gt_count_bin(gt_count: int) -> str:
    if gt_count <= 1:
        return "1"
    if gt_count <= 3:
        return "2-3"
    if gt_count <= 6:
        return "4-6"
    return "7+"


def parse_label_stats(sample_id: str) -> dict[str, Any]:
    label_path = KITTI_ROOT / "label_2" / f"{sample_id}.txt"
    class_counts = {name: 0 for name in SUPPORTED}
    range_counts = {"near": 0, "mid": 0, "far": 0}
    gt_count = 0
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 15:
                continue
            class_name = parts[0]
            if class_name not in SUPPORTED:
                continue
            try:
                x = float(parts[11])
                z = float(parts[13])
            except Exception:
                continue
            dist = math.sqrt(x * x + z * z)
            gt_count += 1
            class_counts[class_name] += 1
            range_counts[distance_bin(dist)] += 1
    point_count = safe_point_count(sample_id)
    class_presence = tuple(int(class_counts[name] > 0) for name in SUPPORTED)
    range_presence = tuple(int(range_counts[name] > 0) for name in ("near", "mid", "far"))
    dominant_class = max(SUPPORTED, key=lambda name: (class_counts[name], name))
    dominant_range = max(("near", "mid", "far"), key=lambda name: (range_counts[name], name))
    return {
        "sample_id": sample_id,
        "gt_count": gt_count,
        "car_gt_count": class_counts["Car"],
        "ped_gt_count": class_counts["Pedestrian"],
        "cyc_gt_count": class_counts["Cyclist"],
        "near_gt_count": range_counts["near"],
        "mid_gt_count": range_counts["mid"],
        "far_gt_count": range_counts["far"],
        "point_count": point_count,
        "class_presence": class_presence,
        "range_presence": range_presence,
        "dominant_class": dominant_class,
        "dominant_range": dominant_range,
        "gt_count_bin": gt_count_bin(gt_count),
    }


def load_frame_stats(sample_ids: list[str]) -> dict[str, dict[str, Any]]:
    return {sample_id: parse_label_stats(sample_id) for sample_id in sample_ids}


def bucket_key(stats: dict[str, Any]) -> tuple[Any, ...]:
    return (
        stats["class_presence"],
        stats["range_presence"],
        stats["dominant_class"],
        stats["dominant_range"],
        stats["gt_count_bin"],
    )


def proportional_group_quotas(group_sizes: dict[tuple[Any, ...], int], target_count: int) -> dict[tuple[Any, ...], int]:
    total = sum(group_sizes.values())
    if total <= 0:
        return {key: 0 for key in group_sizes}
    quotas = {}
    fractional = []
    used = 0
    for key, size in group_sizes.items():
        raw = target_count * (size / total)
        quota = min(size, int(math.floor(raw)))
        quotas[key] = quota
        fractional.append((raw - quota, key))
        used += quota
    remaining = target_count - used
    for _, key in sorted(fractional, key=lambda item: item[0], reverse=True):
        if remaining <= 0:
            break
        if quotas[key] < group_sizes[key]:
            quotas[key] += 1
            remaining -= 1
    return quotas


def stratified_pick(sample_ids: list[str], frame_stats: dict[str, dict[str, Any]], target_count: int, seed: int) -> list[str]:
    if target_count >= len(sample_ids):
        return sorted(sample_ids)
    groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        groups[bucket_key(frame_stats[sample_id])].append(sample_id)
    group_sizes = {key: len(value) for key, value in groups.items()}
    quotas = proportional_group_quotas(group_sizes, target_count)
    rng = random.Random(seed)
    selected: list[str] = []
    leftovers: list[str] = []
    for key, members in groups.items():
        members = list(members)
        rng.shuffle(members)
        quota = quotas.get(key, 0)
        selected.extend(members[:quota])
        leftovers.extend(members[quota:])
    if len(selected) < target_count:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target_count - len(selected)])
    return sorted(selected[:target_count])


def summarize_split(name: str, sample_ids: list[str], frame_stats: dict[str, dict[str, Any]], overlaps: dict[str, int], seed: int) -> dict[str, Any]:
    rows = [frame_stats[sample_id] for sample_id in sample_ids]
    sample_count = len(rows)
    def _sum(key: str) -> int:
        return int(sum(int(row[key]) for row in rows))
    return {
        "split_name": name,
        "sample_count": sample_count,
        "car_gt_count": _sum("car_gt_count"),
        "ped_gt_count": _sum("ped_gt_count"),
        "cyc_gt_count": _sum("cyc_gt_count"),
        "near_gt_count": _sum("near_gt_count"),
        "mid_gt_count": _sum("mid_gt_count"),
        "far_gt_count": _sum("far_gt_count"),
        "avg_points_per_frame": float(np.mean([row["point_count"] for row in rows])) if rows else 0.0,
        "avg_gt_per_frame": float(np.mean([row["gt_count"] for row in rows])) if rows else 0.0,
        "overlap_with_train": overlaps.get("train", 0),
        "overlap_with_val": overlaps.get("val", 0),
        "overlap_with_holdout": overlaps.get("holdout", 0),
        "seed": seed,
    }


def write_split_file(path: Path, sample_ids: list[str]) -> None:
    path.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")


def current_status_snapshot() -> dict[str, Any]:
    report_root = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"
    smoke = json.loads((report_root / "smoke_train_20260630_112530" / "train_status.json").read_text(encoding="utf-8"))
    finetune = json.loads((report_root / "subset_finetune_20260630_112826" / "train_status.json").read_text(encoding="utf-8"))
    shell_retry = json.loads((report_root / "openpcdet_tools_eval_shell_retry_status.json").read_text(encoding="utf-8"))
    deployment = json.loads((report_root / "finetune_deployment_diagnostics.json").read_text(encoding="utf-8"))
    payload = {
        "smoke_train_status": smoke.get("experiment_status", {}),
        "subset_finetune_status": finetune.get("experiment_status", {}),
        "native_tools_test_eval_status": shell_retry.get("tools_eval_status"),
        "external_wrapper_eval_status": "completed",
        "deployment_diagnostics_status": "completed" if deployment else "not_run",
        "safe_claims": [
            "smoke_train 和 subset_finetune 的训练、checkpoint、external official eval、native tools/test.py eval 均已有真实产物。",
            "当前可真实声称的是 subset training / fine-tuning 闭环，不是 full KITTI training。",
            "当前 native tools/test.py eval 已通过 shell-level CUDA/NVVM wrapper 修复。",
        ],
        "limitations": [
            "当前主结果仍是 100/25 smoke 和 200/50 subset_finetune，小样本范围不足以作为主要 holdout 结论。",
            "当前 50-frame subset-val AP 只能视为初步信号，不应直接替代更大 holdout 评估。",
        ],
        "forbidden_claims": [
            "不能声称 full KITTI training。",
            "不能声称 SOTA。",
            "不能把 smoke_train 写成模型收敛。",
            "不能把 subset-val 或 holdout 写成 full KITTI val。",
        ],
    }
    write_json(report_root / "current_status_snapshot.json", payload)
    lines = [
        "# Current Training / Eval Status Snapshot",
        "",
        f"- smoke_train training status: `{payload['smoke_train_status'].get('training_status')}`",
        f"- smoke_train native tools/test.py eval: `{payload['smoke_train_status'].get('opencpdet_tools_eval_status')}`",
        f"- subset_finetune training status: `{payload['subset_finetune_status'].get('training_status')}`",
        f"- subset_finetune native tools/test.py eval: `{payload['subset_finetune_status'].get('opencpdet_tools_eval_status')}`",
        f"- native tools/test.py shell retry overall: `{payload['native_tools_test_eval_status']}`",
        f"- external wrapper eval status: `{payload['external_wrapper_eval_status']}`",
        f"- deployment diagnostics status: `{payload['deployment_diagnostics_status']}`",
        "",
        "## Safe Claims",
        "",
        *[f"- {item}" for item in payload["safe_claims"]],
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in payload["limitations"]],
        "",
        "## Forbidden Claims",
        "",
        *[f"- {item}" for item in payload["forbidden_claims"]],
    ]
    write_markdown(report_root / "current_status_snapshot.md", "\n".join(lines))
    return payload


def main() -> None:
    args = parse_args()
    ensure_dir(OUTPUT_ROOT)
    current_status_snapshot()

    official_train = load_ids(IMAGESETS / "train.txt")
    official_val = load_ids(IMAGESETS / "val.txt")
    all_ids = sorted(set(official_train) | set(official_val))
    stats = load_frame_stats(all_ids)

    train2000 = stratified_pick(official_train, stats, min(2000, len(official_train)), args.seed + 2000)
    remaining_after_train2000 = sorted(set(official_train) - set(train2000))
    val500 = stratified_pick(remaining_after_train2000, stats, min(500, len(remaining_after_train2000)), args.seed + 500)

    train1000 = stratified_pick(train2000, stats, min(1000, len(train2000)), args.seed + 1000)
    val200 = stratified_pick(val500, stats, min(200, len(val500)), args.seed + 200)

    holdout1000 = stratified_pick(official_val, stats, min(1000, len(official_val)), args.seed + 10000)
    holdout500 = stratified_pick(holdout1000, stats, min(500, len(holdout1000)), args.seed + 5000)

    split_defs = {
        "train_1000": train1000,
        "val_200": val200,
        "holdout_eval_500": holdout500,
        "train_2000": train2000,
        "val_500": val500,
        "holdout_eval_1000": holdout1000,
    }

    for split_name, sample_ids in split_defs.items():
        write_split_file(OUTPUT_ROOT / f"{split_name}.txt", sample_ids)

    summaries = []
    for split_name, sample_ids in split_defs.items():
        overlaps = {
            "train": len(set(sample_ids) & set(train1000 if split_name != "train_1000" else train2000)),
            "val": len(set(sample_ids) & set(val200 if split_name != "val_200" else val500)),
            "holdout": len(set(sample_ids) & set(holdout500 if split_name != "holdout_eval_500" else holdout1000)),
        }
        if split_name.startswith("train"):
            overlaps["train"] = len(set(sample_ids) & set(train2000 if split_name == "train_1000" else train1000))
        if split_name.startswith("val"):
            overlaps["val"] = len(set(sample_ids) & set(val500 if split_name == "val_200" else val200))
        if split_name.startswith("holdout"):
            overlaps["holdout"] = len(set(sample_ids) & set(holdout1000 if split_name == "holdout_eval_500" else holdout500))
        summaries.append(summarize_split(split_name, sample_ids, stats, overlaps, args.seed))

    summary_fields = [
        "split_name",
        "sample_count",
        "car_gt_count",
        "ped_gt_count",
        "cyc_gt_count",
        "near_gt_count",
        "mid_gt_count",
        "far_gt_count",
        "avg_points_per_frame",
        "avg_gt_per_frame",
        "overlap_with_train",
        "overlap_with_val",
        "overlap_with_holdout",
        "seed",
    ]
    write_csv(OUTPUT_ROOT / "split_distribution_summary.csv", summaries, summary_fields)

    manifest = {
        "status": "completed",
        "seed": args.seed,
        "strategy": {
            "sampling": "bucketed proportional frame sampling on official train / official val pools",
            "notes": [
                "不直接取前 N 帧；先按 class_presence / range_presence / dominant_class / dominant_range / gt_count_bin 分组，再按组比例采样。",
                "holdout 从官方 val 池采样，train/val 从官方 train 池采样，因此 train-vs-holdout 无跨池 overlap。",
                "train_1000 是 train_2000 的子集，val_200 是 val_500 的子集，holdout_eval_500 是 holdout_eval_1000 的子集。",
            ],
        },
        "source_pools": {
            "official_train_count": len(official_train),
            "official_val_count": len(official_val),
        },
        "splits": {name: {"path": str(OUTPUT_ROOT / f"{name}.txt"), "sample_count": len(ids)} for name, ids in split_defs.items()},
        "explicit_overlaps": {
            f"{lhs}__{rhs}": len(set(split_defs[lhs]) & set(split_defs[rhs]))
            for lhs in split_defs
            for rhs in split_defs
            if lhs < rhs
        },
    }
    write_json(OUTPUT_ROOT / "split_manifest.json", manifest)

    report_lines = [
        "# Expanded Split Distribution Report",
        "",
        f"- Seed: `{args.seed}`",
        f"- Official train pool: `{len(official_train)}` frames",
        f"- Official val pool: `{len(official_val)}` frames",
        "",
        "## Sampling Strategy",
        "",
        "- 先解析每帧 GT 类别、near/mid/far 距离段、每帧 GT 数量、每帧点云点数。",
        "- 再按 `class_presence + range_presence + dominant_class + dominant_range + gt_count_bin` 分桶做比例采样。",
        "- `holdout` 来自官方 `val.txt`，`train/val` 来自官方 `train.txt`，显式避免跨池 overlap。",
        "- `train_1000 / val_200 / holdout_eval_500` 用于本轮主实验；`train_2000 / val_500 / holdout_eval_1000` 作为扩展预留。",
        "",
        "## Split Summary",
        "",
    ]
    for row in summaries:
        report_lines.extend(
            [
                f"### {row['split_name']}",
                "",
                f"- sample_count: `{row['sample_count']}`",
                f"- class GT: `Car={row['car_gt_count']}, Pedestrian={row['ped_gt_count']}, Cyclist={row['cyc_gt_count']}`",
                f"- range GT: `near={row['near_gt_count']}, mid={row['mid_gt_count']}, far={row['far_gt_count']}`",
                f"- avg_points_per_frame: `{row['avg_points_per_frame']:.1f}`",
                f"- avg_gt_per_frame: `{row['avg_gt_per_frame']:.2f}`",
                f"- overlap_with_train: `{row['overlap_with_train']}`",
                f"- overlap_with_val: `{row['overlap_with_val']}`",
                f"- overlap_with_holdout: `{row['overlap_with_holdout']}`",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Explicit Overlap",
            "",
            *[f"- `{key}` = `{value}`" for key, value in manifest["explicit_overlaps"].items()],
        ]
    )
    write_markdown(OUTPUT_ROOT / "split_distribution_report.md", "\n".join(report_lines))

    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(OUTPUT_ROOT),
                "generated_splits": {key: len(value) for key, value in split_defs.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
