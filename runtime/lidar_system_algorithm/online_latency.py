from __future__ import annotations

from .profiling import aggregate_stage_records


ONLINE_STAGE_NAMES = [
    "data_load_ms",
    "calibration_parse_ms",
    "point_preprocess_ms",
    "voxelization_or_pillarization_ms",
    "model_forward_ms",
    "nms_ms",
    "postprocess_ms",
    "tracking_ms",
    "online_total_ms",
]

DEBUG_STAGE_NAMES = [
    "visualization_ms",
    "report_write_ms",
    "image_save_ms",
    "debug_total_ms",
]


def compute_online_debug_records(
    run_records: list[dict],
    tracking_by_frame: dict[str, float] | None = None,
    report_write_ms: float | None = None,
    image_save_ms: float | None = None,
) -> list[dict]:
    tracking_by_frame = tracking_by_frame or {}
    output: list[dict] = []
    for record in run_records:
        frame_id = str(record.get("frame_id", ""))
        tracking_ms = float(tracking_by_frame.get(frame_id, record.get("tracking_ms", 0.0) or 0.0))
        online_total_ms = (
            float(record.get("data_load_ms", 0.0) or 0.0)
            + float(record.get("calibration_parse_ms", 0.0) or 0.0)
            + float(record.get("point_preprocess_ms", 0.0) or 0.0)
            + float(record.get("voxelization_or_pillarization_ms", 0.0) or 0.0)
            + float(record.get("model_forward_ms", 0.0) or 0.0)
            + float(record.get("nms_ms", 0.0) or 0.0)
            + float(record.get("postprocess_ms", 0.0) or 0.0)
            + tracking_ms
        )
        debug_total_ms = (
            float(record.get("visualization_ms", 0.0) or 0.0)
            + float(report_write_ms or record.get("report_write_ms", 0.0) or 0.0)
            + float(image_save_ms or record.get("image_save_ms", 0.0) or 0.0)
        )
        merged = {
            "frame_id": frame_id,
            "data_load_ms": record.get("data_load_ms"),
            "calibration_parse_ms": record.get("calibration_parse_ms"),
            "point_preprocess_ms": record.get("point_preprocess_ms"),
            "voxelization_or_pillarization_ms": record.get("voxelization_or_pillarization_ms"),
            "model_forward_ms": record.get("model_forward_ms"),
            "nms_ms": record.get("nms_ms"),
            "postprocess_ms": record.get("postprocess_ms"),
            "tracking_ms": tracking_ms,
            "online_total_ms": online_total_ms,
            "visualization_ms": record.get("visualization_ms"),
            "report_write_ms": report_write_ms if report_write_ms is not None else record.get("report_write_ms", 0.0),
            "image_save_ms": image_save_ms if image_save_ms is not None else record.get("image_save_ms", 0.0),
            "debug_total_ms": debug_total_ms,
            "legacy_total_ms": record.get("total_ms"),
            "pillar_count": record.get("pillar_count"),
            "detected_box_count": record.get("detected_box_count"),
        }
        output.append(merged)
    return output


def summarize_online_latency(records: list[dict]) -> dict:
    online_summary = aggregate_stage_records(records, ONLINE_STAGE_NAMES)
    debug_summary = aggregate_stage_records(records, DEBUG_STAGE_NAMES)
    return {
        "schema_version": "online_vs_debug_v1",
        "online_stage_names": ONLINE_STAGE_NAMES,
        "debug_stage_names": DEBUG_STAGE_NAMES,
        "online_summary": online_summary,
        "debug_summary": debug_summary,
        "records": records,
        "notes": [
            "online_total_ms excludes visualization, report writing, and image saving.",
            "debug_total_ms is reported separately so offline visualization does not pollute perception runtime.",
        ],
    }
