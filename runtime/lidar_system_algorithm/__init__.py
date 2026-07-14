"""Runtime utilities for the LiDAR-camera benchmark.

Public symbols are loaded lazily so geometry and tracking utilities remain
usable without importing optional GPU or visualization dependencies.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Calibration": ("calibration", "Calibration"),
    "parse_calibration_file": ("calibration", "parse_calibration_file"),
    "DetectionBox": ("dbscan_baseline", "DetectionBox"),
    "build_dbscan_baseline": ("dbscan_baseline", "build_dbscan_baseline"),
    "list_frame_ids": ("kitti_io", "list_frame_ids"),
    "locate_default_kitti_root": ("kitti_io", "locate_default_kitti_root"),
    "resolve_frame_assets": ("kitti_io", "resolve_frame_assets"),
    "StageProfiler": ("profiling", "StageProfiler"),
    "aggregate_stage_records": ("profiling", "aggregate_stage_records"),
    "MultiObjectTracker": ("tracking", "MultiObjectTracker"),
    "TrackState": ("tracking", "TrackState"),
    "pillarize_points": ("voxelization_ablation", "pillarize_points"),
}


def __getattr__(name: str) -> Any:
    """Resolve a public symbol only when the caller first requests it."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value

__all__ = sorted(_EXPORTS)
