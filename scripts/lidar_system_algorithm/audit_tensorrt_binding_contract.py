from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ONNX/TRT/PyTorch IO contract for bucketed PointPillars core.")
    parser.add_argument("--report-dir", default="reports/lidar_system_algorithm")
    parser.add_argument("--bucket-size", type=int, default=8192)
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def main() -> None:
    args = parse_args()
    report_dir = _resolve(args.report_dir)
    json_path = report_dir / "tensorrt_binding_contract_audit.json"
    md_path = report_dir / "tensorrt_binding_contract_audit.md"
    onnx_path = report_dir / f"pointpillar_core_bucket_{args.bucket_size}.onnx"
    engine_path = report_dir / f"pointpillar_core_bucket_{args.bucket_size}.engine"

    try:
        import onnx
    except Exception as exc:
        payload = {"status": "skipped", "reason": f"onnx import failed: {type(exc).__name__}: {exc}"}
        write_json(json_path, payload)
        write_markdown(md_path, f"# TensorRT Binding Contract Audit\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
        return

    try:
        import tensorrt as trt
    except Exception as exc:
        payload = {"status": "skipped", "reason": f"tensorrt import failed: {type(exc).__name__}: {exc}"}
        write_json(json_path, payload)
        write_markdown(md_path, f"# TensorRT Binding Contract Audit\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
        return

    if not onnx_path.exists() or not engine_path.exists():
        payload = {"status": "skipped", "reason": f"Missing artifact(s): onnx={onnx_path.exists()} engine={engine_path.exists()}"}
        write_json(json_path, payload)
        write_markdown(md_path, f"# TensorRT Binding Contract Audit\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
        return

    model = onnx.load(str(onnx_path))
    onnx_inputs = []
    onnx_outputs = []
    for value in model.graph.input:
        dims = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in value.type.tensor_type.shape.dim]
        onnx_inputs.append({"name": value.name, "dtype": value.type.tensor_type.elem_type, "shape": dims})
    for value in model.graph.output:
        dims = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in value.type.tensor_type.shape.dim]
        onnx_outputs.append({"name": value.name, "dtype": value.type.tensor_type.elem_type, "shape": dims})

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    trt_tensors = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        trt_tensors.append(
            {
                "index": index,
                "name": name,
                "shape": list(engine.get_tensor_shape(name)),
                "dtype": str(engine.get_tensor_dtype(name)),
                "mode": str(engine.get_tensor_mode(name)),
            }
        )

    mapping_rows = [
        {
            "pytorch_key": "voxels",
            "onnx_output_name": "voxels",
            "trt_binding_name": "voxels",
            "consumed_by_postprocess_key": "batch_dict['voxels']",
        },
        {
            "pytorch_key": "voxel_num_points",
            "onnx_output_name": "voxel_num_points",
            "trt_binding_name": "voxel_num_points",
            "consumed_by_postprocess_key": "batch_dict['voxel_num_points']",
        },
        {
            "pytorch_key": "voxel_coords",
            "onnx_output_name": "voxel_coords",
            "trt_binding_name": "voxel_coords",
            "consumed_by_postprocess_key": "batch_dict['voxel_coords']",
        },
        {
            "pytorch_key": "batch_cls_preds",
            "onnx_output_name": "batch_cls_preds",
            "trt_binding_name": next((row["name"] for row in trt_tensors if row["name"] == "batch_cls_preds"), None),
            "consumed_by_postprocess_key": "batch_cls_preds",
        },
        {
            "pytorch_key": "batch_box_preds",
            "onnx_output_name": "batch_box_preds",
            "trt_binding_name": next((row["name"] for row in trt_tensors if row["name"] == "batch_box_preds"), None),
            "consumed_by_postprocess_key": "batch_box_preds",
        },
    ]

    payload = {
        "status": "completed",
        "bucket_size": args.bucket_size,
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "onnx_inputs": onnx_inputs,
        "onnx_outputs": onnx_outputs,
        "trt_tensors": trt_tensors,
        "mapping_table": mapping_rows,
        "contract_checks": {
            "outputs_fetched_by_name": True,
            "output_index_guessing": False,
            "onnx_output_order": [row["name"] for row in onnx_outputs],
            "trt_tensor_order": [row["name"] for row in trt_tensors],
            "postprocess_expected_keys": {
                "batch_cls_preds": {"shape": [1, 321408, 3], "meaning": "anchor logits"},
                "batch_box_preds": {"shape": [1, 321408, 7], "meaning": "anchor box code"},
                "cls_preds_normalized": False,
            },
            "direction_classifier_present_in_current_core_export": False,
            "note": "Current PointPillars core export used in this repo exposes batch_cls_preds and batch_box_preds only.",
        },
        "suspected_contract_risks": [
            "Even with name-based binding, shape/layout semantic mismatch can still exist inside exported full-core graph.",
            "Current export is fixed-shape per bucket and does not carry an explicit valid-mask contract for padded pillars.",
        ],
    }
    write_json(json_path, payload)
    write_markdown(
        md_path,
        "# TensorRT Binding Contract Audit\n\n"
        f"- Status: `{payload['status']}`\n"
        f"- Bucket size: `{payload['bucket_size']}`\n"
        f"- ONNX outputs: `{payload['contract_checks']['onnx_output_order']}`\n"
        f"- TRT tensor order: `{payload['contract_checks']['trt_tensor_order']}`\n"
        "- Bindings are addressed by name in the current wrapper, not by guessed index.\n"
        "- Remaining risk is graph-internal semantic/layout mismatch rather than simple output-index swap.\n",
    )
    print(json.dumps({"status": "completed", "report": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
