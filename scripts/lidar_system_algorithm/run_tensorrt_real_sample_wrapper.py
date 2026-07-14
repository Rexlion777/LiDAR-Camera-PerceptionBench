from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.kitti_io import list_frame_ids, locate_default_kitti_root, resolve_frame_assets
from runtime.lidar_system_algorithm.report_schema import write_csv, write_json, write_markdown
from runtime.lidar_system_algorithm.visualization import draw_bar_chart, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real KITTI sample through OpenPCDet preprocessing and a TensorRT PointPillars core engine.")
    parser.add_argument("--kitti-root", default="", help="KITTI root directory.")
    parser.add_argument("--openpcdet-root", default="external/OpenPCDet", help="OpenPCDet root path.")
    parser.add_argument("--cfg-file", default="external/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml", help="PointPillars config path.")
    parser.add_argument("--ckpt", default="checkpoints/pointpillar_kitti.pth", help="PointPillars checkpoint path.")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames.")
    parser.add_argument("--engine", default="reports/lidar_system_algorithm/pointpillar_core_attempt.engine", help="TensorRT core engine path.")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm", help="Output directory.")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    if len(path_str) >= 3 and path_str[1] == ":" and path_str[2] in {"\\", "/"}:
        drive = path_str[0].lower()
        tail = path_str[2:].replace("\\", "/")
        return Path(f"/mnt/{drive}{tail}")
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _ensure_wsl_runtime_paths() -> None:
    ld_parts = [
        "/usr/local/cuda/lib64",
        "/usr/lib/wsl/lib",
        "/opt/tensorrt/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/nvvm/lib64",
    ]
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts + [os.environ.get("LD_LIBRARY_PATH", "")])
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")


def main() -> None:
    args = parse_args()
    _ensure_wsl_runtime_paths()
    output_dir = _resolve(args.output_dir)
    figures_dir = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tensorrt_real_sample_wrapper.csv"
    json_path = output_dir / "tensorrt_real_sample_wrapper.json"
    md_path = output_dir / "tensorrt_real_sample_wrapper.md"
    fig_path = figures_dir / "tensorrt_real_sample_latency.png"

    try:
        import torch
        import tensorrt as trt
        from functools import partial
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils
    except Exception as exc:
        payload = {"status": "skipped", "reason": f"runtime import failed: {type(exc).__name__}: {exc}"}
        write_json(json_path, payload)
        write_markdown(md_path, f"# TensorRT Real-Sample Wrapper\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
        print(json.dumps(payload, indent=2))
        return

    kitti_root = _resolve(args.kitti_root) if args.kitti_root else locate_default_kitti_root()
    openpcdet_root = _resolve(args.openpcdet_root)
    cfg_file = _resolve(args.cfg_file)
    ckpt_path = _resolve(args.ckpt)
    engine_path = _resolve(args.engine)
    if not engine_path.exists():
        payload = {"status": "skipped", "reason": f"TensorRT engine not found: {engine_path}"}
        write_json(json_path, payload)
        write_markdown(md_path, f"# TensorRT Real-Sample Wrapper\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
        print(json.dumps(payload, indent=2))
        return

    sys.path.insert(0, str(openpcdet_root))
    os.chdir(openpcdet_root / "tools")
    cfg_from_yaml_file(str(cfg_file), cfg)

    def cuda_sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def timed(fn, *fn_args, **fn_kwargs):
        cuda_sync()
        start = time.perf_counter()
        result = fn(*fn_args, **fn_kwargs)
        cuda_sync()
        return result, (time.perf_counter() - start) * 1000.0

    def processor_name(processor) -> str:
        func = processor.func if isinstance(processor, partial) else processor
        return getattr(func, "__name__", str(func))

    class RuntimeDataset(DatasetTemplate):
        def __init__(self):
            super().__init__(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                training=False,
                root_path=openpcdet_root,
                logger=common_utils.create_logger(),
            )

        def __len__(self):
            return 0

        def __getitem__(self, index):
            raise IndexError(index)

        def prepare_timed(self, points: np.ndarray, frame_id: str) -> tuple[dict, dict]:
            data_dict = {"points": points, "frame_id": frame_id}
            data_dict = self.set_lidar_aug_matrix(data_dict)
            data_dict, point_preprocess_ms = timed(self.point_feature_encoder.forward, data_dict)
            voxel_ms = 0.0
            other_ms = 0.0
            for processor in self.data_processor.data_processor_queue:
                data_dict, elapsed = timed(processor, data_dict=data_dict)
                if processor_name(processor) == "transform_points_to_voxels":
                    voxel_ms += elapsed
                else:
                    other_ms += elapsed
            data_dict.pop("gt_names", None)
            return data_dict, {"point_preprocess_ms": point_preprocess_ms + other_ms, "voxelization_or_pillarization_ms": voxel_ms}

    dataset = RuntimeDataset()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_path), logger=common_utils.create_logger(), to_cpu=True)
    model.cuda().eval()

    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(trt_logger)
    engine = trt_runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()
    engine_shapes = {engine.get_tensor_name(i): tuple(engine.get_tensor_shape(engine.get_tensor_name(i))) for i in range(engine.num_io_tensors)}
    fixed_voxel_count = int(engine_shapes.get("voxels", (64, 32, 4))[0])

    rows: list[dict] = []
    detections: list[dict] = []
    frame_ids = list_frame_ids(kitti_root, limit=args.frames)
    with torch.no_grad():
        for frame_id in frame_ids:
            assets = resolve_frame_assets(kitti_root, frame_id)
            raw_points, data_load_ms = timed(np.fromfile, str(assets.lidar_path), np.float32)
            points = raw_points.reshape(-1, 4)
            prepared, prep = dataset.prepare_timed(points, frame_id)
            full_pillar_count = int(prepared["voxels"].shape[0])
            original_prepared = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in prepared.items()}

            capped = dict(prepared)
            for key in ["voxels", "voxel_num_points", "voxel_coords"]:
                value = capped[key]
                if value.shape[0] >= fixed_voxel_count:
                    capped[key] = value[:fixed_voxel_count].copy()
                else:
                    pad_shape = (fixed_voxel_count - value.shape[0],) + value.shape[1:]
                    capped[key] = np.concatenate([value, np.zeros(pad_shape, dtype=value.dtype)], axis=0)
            batch_dict, collate_ms = timed(dataset.collate_batch, [capped])
            _, gpu_transfer_ms = timed(load_data_to_gpu, batch_dict)

            py_batch = {key: value for key, value in batch_dict.items()}
            for module in model.module_list:
                py_batch, _ = timed(module, py_batch)
            pytorch_core_ms = 0.0
            py_batch = {key: value for key, value in batch_dict.items()}
            for module in model.module_list:
                py_batch, elapsed = timed(module, py_batch)
                pytorch_core_ms += elapsed
            (py_pred, _), py_post_ms = timed(model.post_processing, py_batch)

            buffers = {
                "voxels": batch_dict["voxels"].contiguous(),
                "voxel_num_points": batch_dict["voxel_num_points"].contiguous(),
                "voxel_coords": batch_dict["voxel_coords"].contiguous(),
                "batch_cls_preds": torch.empty((1, 321408, len(cfg.CLASS_NAMES)), device="cuda", dtype=torch.float32),
                "batch_box_preds": torch.empty((1, 321408, 7), device="cuda", dtype=torch.float32),
            }
            for name, tensor in buffers.items():
                if name in engine_shapes:
                    context.set_tensor_address(name, int(tensor.data_ptr()))
            stream = torch.cuda.Stream()
            for _ in range(3):
                context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            start = time.perf_counter()
            ok = context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            trt_core_ms = (time.perf_counter() - start) * 1000.0
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned false")

            trt_batch = dict(batch_dict)
            trt_batch["batch_cls_preds"] = buffers["batch_cls_preds"]
            trt_batch["batch_box_preds"] = buffers["batch_box_preds"]
            trt_batch["cls_preds_normalized"] = False
            (trt_pred, _), trt_post_ms = timed(model.post_processing, trt_batch)

            point_preprocess_ms = float(prep["point_preprocess_ms"] + collate_ms + gpu_transfer_ms)
            online_total_ms = data_load_ms + point_preprocess_ms + float(prep["voxelization_or_pillarization_ms"]) + trt_core_ms + trt_post_ms
            pytorch_online_total_ms = data_load_ms + point_preprocess_ms + float(prep["voxelization_or_pillarization_ms"]) + pytorch_core_ms + py_post_ms
            py_boxes = py_pred[0]["pred_boxes"].detach().cpu().numpy()
            trt_boxes = trt_pred[0]["pred_boxes"].detach().cpu().numpy()
            topk = min(10, py_boxes.shape[0], trt_boxes.shape[0])
            center_diff = float(np.linalg.norm(py_boxes[:topk, :3] - trt_boxes[:topk, :3], axis=1).mean()) if topk else None
            row = {
                "frame_id": frame_id,
                "data_load_ms": data_load_ms,
                "point_preprocess_ms": point_preprocess_ms,
                "voxelization_or_pillarization_ms": prep["voxelization_or_pillarization_ms"],
                "trt_core_ms": trt_core_ms,
                "nms_postprocess_ms": trt_post_ms,
                "online_total_ms": online_total_ms,
                "pytorch_core_ms": pytorch_core_ms,
                "pytorch_nms_postprocess_ms": py_post_ms,
                "pytorch_online_total_ms": pytorch_online_total_ms,
                "speedup_core_only": pytorch_core_ms / trt_core_ms if trt_core_ms > 0 else None,
                "speedup_online_total_if_any": pytorch_online_total_ms / online_total_ms if online_total_ms > 0 else None,
                "full_pillar_count": full_pillar_count,
                "engine_pillar_count": fixed_voxel_count,
                "pillar_truncation_applied": full_pillar_count > fixed_voxel_count,
                "pytorch_box_count": int(py_boxes.shape[0]),
                "trt_box_count": int(trt_boxes.shape[0]),
                "topk_center_diff_mean": center_diff,
            }
            rows.append(row)
            for idx, box in enumerate(trt_boxes[:50]):
                detections.append({"frame_id": frame_id, "object_id": idx, "box_3d_lidar": box.tolist(), "source": "tensorrt_core_wrapper"})

    fieldnames = [
        "frame_id",
        "data_load_ms",
        "point_preprocess_ms",
        "voxelization_or_pillarization_ms",
        "trt_core_ms",
        "nms_postprocess_ms",
        "online_total_ms",
        "pytorch_core_ms",
        "pytorch_nms_postprocess_ms",
        "pytorch_online_total_ms",
        "speedup_core_only",
        "speedup_online_total_if_any",
        "full_pillar_count",
        "engine_pillar_count",
        "pillar_truncation_applied",
        "pytorch_box_count",
        "trt_box_count",
        "topk_center_diff_mean",
    ]
    write_csv(csv_path, rows, fieldnames)
    summary = {
        "status": "completed" if rows else "skipped",
        "scope": "real KITTI point cloud -> OpenPCDet preprocessing -> fixed-shape TensorRT PointPillars core -> OpenPCDet postprocess/NMS",
        "limitations": [
            "TensorRT engine covers core network only; voxelization and NMS remain outside TensorRT.",
            f"Engine expects {fixed_voxel_count} pillars, so real samples are truncated/padded to that fixed shape.",
            "This is not a full dynamic TensorRT detector and no FP16 acceleration is claimed.",
        ],
        "engine_path": str(engine_path),
        "engine_shapes": {key: list(value) for key, value in engine_shapes.items()},
        "frame_count": len(rows),
        "mean_trt_core_ms": statistics.fmean(row["trt_core_ms"] for row in rows) if rows else None,
        "mean_pytorch_core_ms": statistics.fmean(row["pytorch_core_ms"] for row in rows) if rows else None,
        "mean_online_total_ms": statistics.fmean(row["online_total_ms"] for row in rows) if rows else None,
        "mean_pytorch_online_total_ms": statistics.fmean(row["pytorch_online_total_ms"] for row in rows) if rows else None,
        "mean_speedup_core_only": statistics.fmean(row["speedup_core_only"] for row in rows if row["speedup_core_only"] is not None) if rows else None,
        "mean_speedup_online_total_if_any": statistics.fmean(row["speedup_online_total_if_any"] for row in rows if row["speedup_online_total_if_any"] is not None) if rows else None,
        "truncated_frame_count": sum(1 for row in rows if row["pillar_truncation_applied"]),
        "rows": rows,
        "detections_preview": detections[:100],
    }
    write_json(json_path, summary)
    write_markdown(
        md_path,
        f"""# TensorRT Real-Sample Wrapper

- Status: `{summary["status"]}`
- Scope: `{summary["scope"]}`
- Mean TRT core ms: `{summary["mean_trt_core_ms"]}`
- Mean PyTorch core ms: `{summary["mean_pytorch_core_ms"]}`
- Mean online total ms: `{summary["mean_online_total_ms"]}`
- Mean PyTorch online total ms: `{summary["mean_pytorch_online_total_ms"]}`
- Mean core-only speedup: `{summary["mean_speedup_core_only"]}`
- Engine pillar count: `{fixed_voxel_count}`
- Truncated frames: `{summary["truncated_frame_count"]}`

## Limitations

- TensorRT covers the fixed-shape PointPillars core only.
- Voxelization/pillarization and NMS/postprocess remain outside the TensorRT engine.
- Real KITTI samples are truncated or padded to the engine's fixed pillar shape, so this is a deployment wrapper/prototype rather than a full dynamic TensorRT detector.
""",
    )
    save_image(
        fig_path,
        draw_bar_chart(
            [
                {"stage": "trt_core_ms", "mean_ms": summary["mean_trt_core_ms"]},
                {"stage": "pytorch_core_ms", "mean_ms": summary["mean_pytorch_core_ms"]},
                {"stage": "trt_wrapper_online_ms", "mean_ms": summary["mean_online_total_ms"]},
                {"stage": "pytorch_online_ms", "mean_ms": summary["mean_pytorch_online_total_ms"]},
            ],
            title="TensorRT Real-Sample Wrapper Latency",
        ),
    )
    print(f"Saved TensorRT real-sample wrapper: {json_path}")


if __name__ == "__main__":
    main()
