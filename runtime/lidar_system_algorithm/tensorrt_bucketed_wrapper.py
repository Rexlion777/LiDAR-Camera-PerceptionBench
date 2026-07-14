from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


def parse_bucket_sizes(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        buckets = [int(part) for part in parts]
    else:
        buckets = [int(item) for item in value]
    buckets = sorted({bucket for bucket in buckets if bucket > 0})
    if not buckets:
        raise ValueError("At least one positive bucket size is required.")
    return buckets


def select_bucket_size(pillar_count: int, bucket_sizes: Iterable[int]) -> int:
    buckets = parse_bucket_sizes(bucket_sizes)
    for bucket in buckets:
        if pillar_count <= bucket:
            return bucket
    return buckets[-1]


@dataclass
class TruncationStats:
    pillar_count: int
    bucket_size: int
    truncated: bool
    truncated_pillars: int
    padding_pillars: int

    @property
    def kept_pillars(self) -> int:
        return min(self.pillar_count, self.bucket_size)


def compute_truncation_stats(pillar_count: int, bucket_size: int) -> TruncationStats:
    pillar_count = int(pillar_count)
    bucket_size = int(bucket_size)
    truncated_pillars = max(0, pillar_count - bucket_size)
    padding_pillars = max(0, bucket_size - pillar_count)
    return TruncationStats(
        pillar_count=pillar_count,
        bucket_size=bucket_size,
        truncated=truncated_pillars > 0,
        truncated_pillars=truncated_pillars,
        padding_pillars=padding_pillars,
    )


def summarize_values(values: Iterable[float]) -> dict[str, float | int | None]:
    seq = sorted(float(value) for value in values)
    if not seq:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}

    def percentile(pct: float) -> float:
        if len(seq) == 1:
            return seq[0]
        position = (len(seq) - 1) * pct / 100.0
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return seq[lower]
        blend = position - lower
        return seq[lower] * (1.0 - blend) + seq[upper] * blend

    return {
        "count": len(seq),
        "mean": sum(seq) / len(seq),
        "p50": percentile(50.0),
        "p95": percentile(95.0),
        "min": seq[0],
        "max": seq[-1],
    }


def summarize_bucket_run_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    buckets = sorted({int(row["selected_bucket_size"]) for row in rows})
    summary_rows: list[dict] = []
    for bucket in buckets:
        bucket_rows = [row for row in rows if int(row["selected_bucket_size"]) == bucket]
        pillar_stats = summarize_values(float(row["full_pillar_count"]) for row in bucket_rows)
        trunc_values = [float(row["truncated_pillars"]) for row in bucket_rows]
        core_stats = summarize_values(float(row["trt_core_ms"]) for row in bucket_rows if row.get("trt_core_ms") not in (None, ""))
        wrapper_stats = summarize_values(float(row["online_total_ms"]) for row in bucket_rows if row.get("online_total_ms") not in (None, ""))
        nms_stats = summarize_values(float(row["nms_postprocess_ms"]) for row in bucket_rows if row.get("nms_postprocess_ms") not in (None, ""))
        score_values = [float(row["score_mean"]) for row in bucket_rows if row.get("score_mean") not in (None, "")]
        summary_rows.append(
            {
                "bucket_size": bucket,
                "frame_count": len(bucket_rows),
                "bucket_hit_count": len(bucket_rows),
                "truncation_frame_count": sum(1 for row in bucket_rows if bool(row.get("pillar_truncation_applied"))),
                "truncation_rate": sum(1 for row in bucket_rows if bool(row.get("pillar_truncation_applied"))) / len(bucket_rows),
                "mean_truncated_pillars": (sum(trunc_values) / len(trunc_values)) if trunc_values else 0.0,
                "pillar_count_mean": pillar_stats["mean"],
                "pillar_count_p50": pillar_stats["p50"],
                "pillar_count_p95": pillar_stats["p95"],
                "pillar_count_max": pillar_stats["max"],
                "core_latency_mean": core_stats["mean"],
                "core_latency_p50": core_stats["p50"],
                "core_latency_p95": core_stats["p95"],
                "online_wrapper_latency_mean": wrapper_stats["mean"],
                "online_wrapper_latency_p50": wrapper_stats["p50"],
                "online_wrapper_latency_p95": wrapper_stats["p95"],
                "nms_postprocess_latency_mean": nms_stats["mean"],
                "output_box_count_mean": (sum(float(row["trt_box_count"]) for row in bucket_rows) / len(bucket_rows)) if bucket_rows else 0.0,
                "score_mean": (sum(score_values) / len(score_values)) if score_values else None,
            }
        )
    return summary_rows


def bucket_hit_distribution(rows: list[dict]) -> dict[int, int]:
    distribution: dict[int, int] = {}
    for row in rows:
        bucket = int(row["selected_bucket_size"])
        distribution[bucket] = distribution.get(bucket, 0) + 1
    return distribution
