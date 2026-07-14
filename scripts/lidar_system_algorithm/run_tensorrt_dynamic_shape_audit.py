from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import write_json, write_markdown


def main() -> None:
    output_dir = PROJECT_ROOT / "reports" / "lidar_system_algorithm"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tensorrt_dynamic_shape_audit.json"
    md_path = output_dir / "tensorrt_dynamic_shape_audit.md"
    payload = {
        "status": "completed_with_blocker",
        "scope": "Audit dynamic-shape / plugin routes for PointPillars TensorRT core deployment.",
        "findings": [
            "PointPillarScatter export remains fixed-shape in the current project path.",
            "TensorRT bucket engines build successfully for multiple capacities, but current raw tensor alignment is not stable across buckets.",
            "Zero-point padding was one blocker and has been fixed; raw tensor mismatch still remains on several buckets.",
        ],
        "dynamic_shape_blockers": [
            "Current ONNX export path uses fixed dummy/bucket shapes and does not propagate true dynamic pillar counts.",
            "PointPillarScatter includes data-dependent behavior that previously blocked torch.export/dynamo export in this project.",
            "A production-ready dynamic route likely needs either a custom scatter plugin or a reworked export graph.",
        ],
        "plugin_route_candidates": [
            "Scatter plugin: replace PointPillarScatter with a TensorRT plugin or a custom CUDA plugin stage.",
            "Preprocess plugin: keep voxelization external but pass compact valid-pillar count and plugin-aware masked tensors.",
            "NMS plugin: keep core in TRT and move class-agnostic NMS to a verified plugin only after raw tensor alignment is solved.",
        ],
        "recommendation": "Treat bucketed fixed-shape TensorRT as a capacity/latency study and raw-tensor debugging aid. Do not present it as dynamic-shape deployment until scatter/output alignment is solved.",
    }
    write_json(json_path, payload)
    write_markdown(
        md_path,
        "# TensorRT Dynamic-Shape Audit\n\n"
        f"- Status: `{payload['status']}`\n"
        "- Current export path remains fixed-shape / bucketed.\n"
        "- The next practical route is a scatter/plugin strategy after raw tensor alignment is fixed.\n",
    )
    print(json.dumps({"output": str(json_path), "status": payload["status"]}, indent=2))


if __name__ == "__main__":
    main()
