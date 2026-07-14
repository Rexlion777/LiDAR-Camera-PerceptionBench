from __future__ import annotations

import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch


def cuda_synchronize_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed_stage(stage_name: str, stage_record: dict[str, float | None]) -> Iterator[None]:
    cuda_synchronize_if_needed()
    start = time.perf_counter()
    try:
        yield
    finally:
        cuda_synchronize_if_needed()
        stage_record[stage_name] = (time.perf_counter() - start) * 1000.0


@dataclass
class StageProfiler:
    runs: list[dict[str, float | None]] = field(default_factory=list)

    def new_run(self) -> dict[str, float | None]:
        record: dict[str, float | None] = {}
        self.runs.append(record)
        return record


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_stage(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(sorted_values, 0.50),
        "p95_ms": _percentile(sorted_values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def aggregate_stage_records(records: list[dict[str, float | None]], stage_names: list[str]) -> list[dict]:
    rows: list[dict] = []
    for stage_name in stage_names:
        values = [float(record[stage_name]) for record in records if record.get(stage_name) is not None]
        summary = summarize_stage(values)
        rows.append({"stage": stage_name, **summary})
    return rows
