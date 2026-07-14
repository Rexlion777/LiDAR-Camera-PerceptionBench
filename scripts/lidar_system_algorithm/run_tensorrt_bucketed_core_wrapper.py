from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.kitti_io import list_frame_ids, locate_default_kitti_root, resolve_frame_assets
from runtime.lidar_system_algorithm.report_schema import write_csv, write_json, write_markdown
from runtime.lidar_system_algorithm.tensorrt_bucketed_wrapper import (
    bucket_hit_distribution,
    compute_truncation_stats,
    parse_bucket_sizes,
    select_bucket_size,
    summarize_bucket_run_rows,
    summarize_values,
)
from runtime.lidar_system_algorithm.visualization import draw_bar_chart, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realistic/bucketed TensorRT PointPillars core wrapper on KITTI.")
    parser.add_argument("--kitti-root", default="data/kitti_object_raw/extracted", help="KITTI root directory.")
    parser.add_argument("--openpcdet-root", default="external/OpenPCDet", help="OpenPCDet root path.")
    parser.add_argument("--cfg-file", default="external/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml", help="PointPillars config path.")
    parser.add_argument("--ckpt", default="checkpoints/pointpillar_kitti.pth", help="PointPillars checkpoint path.")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames for the capacity/latency report. 0 means all available frames.")
    parser.add_argument("--bucket-sizes", default="4096,8192,12000,16000,20000", help="Comma-separated fixed-shape bucket sizes.")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm", help="Output directory.")
    parser.add_argument("--pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt_trt_bucketed", help="KITTI prediction txt output directory for the wrapper path.")
    parser.add_argument("--split-file", default="external/OpenPCDet/data/kitti/ImageSets/val.txt", help="KITTI split file for optional official eval.")
    parser.add_argument("--eval-max-frames", type=int, default=0, help="Optional frame cap for bucketed wrapper official eval. 0 means full split.")
    parser.add_argument("--run-official-eval", action="store_true", help="Export KITTI txt and run local official eval for the bucketed wrapper.")
    parser.add_argument("--build-only", action="store_true", help="Build engines and write the capacity report without running frames.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild ONNX/engine artifacts even if they already exist.")
    return parser.parse_args()


def read_image_shape(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return np.array(image.shape[:2], dtype=np.int32)


def load_split_ids(split_file: Path, max_frames: int) -> list[str]:
    frame_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return frame_ids[:max_frames] if max_frames > 0 else frame_ids


def pad_bucket_inputs(prepared: dict, bucket_size: int) -> dict:
    capped = dict(prepared)
    current_count = int(capped["voxels"].shape[0])
    if current_count >= bucket_size:
        for key in ["voxels", "voxel_num_points", "voxel_coords"]:
            capped[key] = capped[key][:bucket_size].copy()
        return capped

    if current_count <= 0:
        capped["voxels"] = np.zeros((bucket_size,) + capped["voxels"].shape[1:], dtype=capped["voxels"].dtype)
        capped["voxel_num_points"] = np.ones((bucket_size,), dtype=capped["voxel_num_points"].dtype)
        capped["voxel_coords"] = np.zeros((bucket_size,) + capped["voxel_coords"].shape[1:], dtype=capped["voxel_coords"].dtype)
        return capped

    pad_count = bucket_size - current_count
    # Repeat the first valid pillar instead of padding zero-point pillars, because
    # PillarVFE divides by voxel_num_points and zero counts would create NaNs.
    capped["voxels"] = np.concatenate([capped["voxels"], np.repeat(capped["voxels"][:1], pad_count, axis=0)], axis=0)
    capped["voxel_num_points"] = np.concatenate(
        [capped["voxel_num_points"], np.repeat(capped["voxel_num_points"][:1], pad_count, axis=0)],
        axis=0,
    )
    capped["voxel_coords"] = np.concatenate([capped["voxel_coords"], np.repeat(capped["voxel_coords"][:1], pad_count, axis=0)], axis=0)
    return capped


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    figures_dir = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "tensorrt_bucketed_core_report.json"
    report_md = output_dir / "tensorrt_bucketed_core_report.md"
    latency_csv = output_dir / "tensorrt_bucketed_latency.csv"
    truncation_csv = output_dir / "tensorrt_bucketed_truncation.csv"
    eval_json_path = output_dir / "tensorrt_bucketed_kitti_eval.json"
    eval_md_path = output_dir / "tensorrt_bucketed_kitti_eval.md"
    latency_fig = figures_dir / "tensorrt_bucket_latency_vs_capacity.png"
    bucket_hit_fig = figures_dir / "tensorrt_bucket_hit_distribution.png"

    try:
        import cv2  # noqa: F401
        import tensorrt as trt
        import torch
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.datasets.kitti.kitti_dataset import KittiDataset
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import calibration_kitti, common_utils
    except Exception as exc:
        payload = {
            "status": "skipped",
            "reason": f"runtime import failed: {type(exc).__name__}: {exc}",
            "limitations": ["TensorRT bucketed wrapper requires the WSL OpenPCDet/TensorRT runtime."],
        }
        write_json(report_json, payload)
        write_markdown(report_md, f"# TensorRT Bucketed Core Report\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
        print(json.dumps(payload, indent=2))
        return

    kitti_root = (PROJECT_ROOT / args.kitti_root).resolve() if not Path(args.kitti_root).is_absolute() else Path(args.kitti_root)
    openpcdet_root = (PROJECT_ROOT / args.openpcdet_root).resolve() if not Path(args.openpcdet_root).is_absolute() else Path(args.openpcdet_root)
    cfg_file = (PROJECT_ROOT / args.cfg_file).resolve() if not Path(args.cfg_file).is_absolute() else Path(args.cfg_file)
    ckpt_path = Path(args.ckpt).expanduser()
    pred_dir = (PROJECT_ROOT / args.pred_dir).resolve() if not Path(args.pred_dir).is_absolute() else Path(args.pred_dir)
    split_file = (PROJECT_ROOT / args.split_file).resolve() if not Path(args.split_file).is_absolute() else Path(args.split_file)
    bucket_sizes = parse_bucket_sizes(args.bucket_sizes)

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
                name = getattr(getattr(processor, "func", processor), "__name__", str(processor))
                if name == "transform_points_to_voxels":
                    voxel_ms += elapsed
                else:
                    other_ms += elapsed
            data_dict.pop("gt_names", None)
            return data_dict, {"point_preprocess_ms": point_preprocess_ms + other_ms, "voxelization_or_pillarization_ms": voxel_ms}

    class PointPillarCore(torch.nn.Module):
        def __init__(self, pointpillar_model):
            super().__init__()
            self.modules_for_export = torch.nn.ModuleList(list(pointpillar_model.module_list))

        def forward(self, voxels, voxel_num_points, voxel_coords):
            batch_dict = {
                "voxels": voxels,
                "voxel_num_points": voxel_num_points,
                "voxel_coords": voxel_coords,
                "batch_size": 1,
            }
            for module in self.modules_for_export:
                batch_dict = module(batch_dict)
            return batch_dict["batch_cls_preds"], batch_dict["batch_box_preds"]

    dataset = RuntimeDataset()
    logger = common_utils.create_logger()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=True)
    model.cuda().eval()
    core = PointPillarCore(model).cuda().eval()

    trt_logger = trt.Logger(trt.Logger.WARNING)
    build_rows: list[dict] = []
    engine_paths: dict[int, Path] = {}
    engine_shapes: dict[int, dict[str, list[int]]] = {}

    def build_engine_for_bucket(bucket_size: int) -> dict:
        onnx_path = output_dir / f"pointpillar_core_bucket_{bucket_size}.onnx"
        engine_path = output_dir / f"pointpillar_core_bucket_{bucket_size}.engine"
        build_row = {
            "bucket_size": bucket_size,
            "build_success": False,
            "onnx_path": str(onnx_path),
            "engine_path": str(engine_path),
            "engine_size_mb": None,
            "build_ms": None,
            "parser_errors": "",
            "memory_usage_mb": None,
            "status": "not_started",
        }
        try:
            if args.force_rebuild or not onnx_path.exists():
                voxels = torch.randn(bucket_size, 32, 4, device="cuda", dtype=torch.float32)
                voxel_num_points = torch.randint(1, 32, (bucket_size,), device="cuda", dtype=torch.int32)
                voxel_coords = torch.zeros((bucket_size, 4), device="cuda", dtype=torch.int32)
                voxel_coords[:, 2] = torch.arange(bucket_size, device="cuda", dtype=torch.int32) % 496
                voxel_coords[:, 3] = torch.arange(bucket_size, device="cuda", dtype=torch.int32) % 432
                torch.onnx.export(
                    core,
                    (voxels, voxel_num_points, voxel_coords),
                    str(onnx_path),
                    opset_version=17,
                    dynamo=False,
                    input_names=["voxels", "voxel_num_points", "voxel_coords"],
                    output_names=["batch_cls_preds", "batch_box_preds"],
                )
            if args.force_rebuild or not engine_path.exists():
                builder = trt.Builder(trt_logger)
                network = builder.create_network(0)
                parser = trt.OnnxParser(network, trt_logger)
                parse_ok = parser.parse(onnx_path.read_bytes())
                parser_errors = [parser.get_error(index).desc() for index in range(parser.num_errors)]
                if not parse_ok:
                    build_row["status"] = "parse_failed"
                    build_row["parser_errors"] = " | ".join(parser_errors)
                    return build_row
                config = builder.create_builder_config()
                if hasattr(config, "set_memory_pool_limit"):
                    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
                elif hasattr(config, "max_workspace_size"):
                    config.max_workspace_size = 2 << 30
                start = time.perf_counter()
                serialized = builder.build_serialized_network(network, config) if hasattr(builder, "build_serialized_network") else None
                build_ms = (time.perf_counter() - start) * 1000.0
                if serialized is None:
                    build_row["status"] = "build_failed"
                    build_row["build_ms"] = build_ms
                    build_row["parser_errors"] = "builder returned no serialized network"
                    return build_row
                engine_path.write_bytes(bytes(serialized))
                build_row["build_ms"] = build_ms
            build_row["build_success"] = True
            build_row["status"] = "completed"
            build_row["engine_size_mb"] = engine_path.stat().st_size / (1024.0 * 1024.0)
            engine_paths[bucket_size] = engine_path
            runtime = trt.Runtime(trt_logger)
            engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
            engine_shapes[bucket_size] = {
                engine.get_tensor_name(index): list(engine.get_tensor_shape(engine.get_tensor_name(index)))
                for index in range(engine.num_io_tensors)
            }
            return build_row
        except Exception as exc:
            build_row["status"] = f"failed:{type(exc).__name__}"
            build_row["parser_errors"] = str(exc)
            return build_row

    for bucket_size in bucket_sizes:
        build_rows.append(build_engine_for_bucket(bucket_size))

    if args.build_only:
        payload = {
            "status": "completed" if any(row["build_success"] for row in build_rows) else "skipped",
            "scope": "Build-only realistic/bucketed TensorRT PointPillars core engines. Voxelization and NMS remain outside TensorRT.",
            "bucket_sizes": bucket_sizes,
            "engine_build_rows": build_rows,
            "dynamic_shape_audit": {
                "status": "skipped",
                "reason": "Dynamic-shape export/audit was not implemented here because PointPillarScatter export is currently fixed-shape in this project.",
            },
            "limitations": [
                "TensorRT covers the PointPillars core only.",
                "This build report does not imply a full TensorRT detector.",
            ],
        }
        write_json(report_json, payload)
        write_markdown(
            report_md,
            "# TensorRT Bucketed Core Report\n\n"
            f"- Status: `{payload['status']}`\n"
            f"- Bucket sizes: `{bucket_sizes}`\n"
            f"- Successful builds: `{sum(1 for row in build_rows if row['build_success'])}` / `{len(build_rows)}`\n",
        )
        return

    successful_buckets = [row["bucket_size"] for row in build_rows if row["build_success"]]
    if not successful_buckets:
        payload = {
            "status": "skipped",
            "reason": "No TensorRT bucket engine was built successfully.",
            "bucket_sizes": bucket_sizes,
            "engine_build_rows": build_rows,
        }
        write_json(report_json, payload)
        write_markdown(report_md, f"# TensorRT Bucketed Core Report\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
        print(json.dumps(payload, indent=2))
        return

    engines: dict[int, tuple[object, object]] = {}
    for bucket_size in successful_buckets:
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(engine_paths[bucket_size].read_bytes())
        engines[bucket_size] = (engine, engine.create_execution_context())

    frame_rows: list[dict] = []
    frame_ids = list_frame_ids(kitti_root, limit=args.frames if args.frames > 0 else None)
    pred_dir.mkdir(parents=True, exist_ok=True)
    per_frame_predictions: list[dict] = []

    with torch.no_grad():
        for frame_id in frame_ids:
            assets = resolve_frame_assets(kitti_root, frame_id)
            raw_points, data_load_ms = timed(np.fromfile, str(assets.lidar_path), np.float32)
            points = raw_points.reshape(-1, 4)
            prepared, prep = dataset.prepare_timed(points, frame_id)
            full_pillar_count = int(prepared["voxels"].shape[0])
            bucket_size = select_bucket_size(full_pillar_count, successful_buckets)
            trunc_stats = compute_truncation_stats(full_pillar_count, bucket_size)
            capped = pad_bucket_inputs(prepared, bucket_size)
            batch_dict, collate_ms = timed(dataset.collate_batch, [capped])
            _, gpu_transfer_ms = timed(load_data_to_gpu, batch_dict)

            pytorch_outputs, pytorch_core_ms = timed(core, batch_dict["voxels"], batch_dict["voxel_num_points"], batch_dict["voxel_coords"])
            py_batch = dict(batch_dict)
            py_batch["batch_cls_preds"] = pytorch_outputs[0]
            py_batch["batch_box_preds"] = pytorch_outputs[1]
            py_batch["cls_preds_normalized"] = False
            (py_pred, _), py_post_ms = timed(model.post_processing, py_batch)

            engine, context = engines[bucket_size]
            shapes = engine_shapes[bucket_size]
            buffers = {
                "voxels": batch_dict["voxels"].contiguous(),
                "voxel_num_points": batch_dict["voxel_num_points"].contiguous(),
                "voxel_coords": batch_dict["voxel_coords"].contiguous(),
                "batch_cls_preds": torch.empty(tuple(shapes["batch_cls_preds"]), device="cuda", dtype=torch.float32),
                "batch_box_preds": torch.empty(tuple(shapes["batch_box_preds"]), device="cuda", dtype=torch.float32),
            }
            for name, tensor in buffers.items():
                if name in shapes:
                    context.set_tensor_address(name, int(tensor.data_ptr()))
            stream = torch.cuda.Stream()
            for _ in range(3):
                context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            ok = context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            trt_core_ms = (time.perf_counter() - start) * 1000.0
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned false")
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

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
            pred_scores = trt_pred[0]["pred_scores"].detach().cpu().numpy()
            row = {
                "frame_id": frame_id,
                "selected_bucket_size": bucket_size,
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
                "engine_pillar_count": bucket_size,
                "pillar_truncation_applied": trunc_stats.truncated,
                "truncated_pillars": trunc_stats.truncated_pillars,
                "padding_pillars": trunc_stats.padding_pillars,
                "pytorch_box_count": int(py_boxes.shape[0]),
                "trt_box_count": int(trt_boxes.shape[0]),
                "topk_center_diff_mean": center_diff,
                "score_mean": float(np.mean(pred_scores)) if pred_scores.size else None,
                "score_max": float(np.max(pred_scores)) if pred_scores.size else None,
                "peak_cuda_memory_mb": peak_memory_mb,
            }
            frame_rows.append(row)

            if args.run_official_eval:
                calib = calibration_kitti.Calibration(assets.calib_path)
                image_shape = read_image_shape(assets.image_path)
                prediction_batch = {
                    "frame_id": [frame_id],
                    "calib": [calib],
                    "image_shape": torch.from_numpy(np.expand_dims(image_shape, axis=0)),
                }
                annos = KittiDataset.generate_prediction_dicts(
                    batch_dict=prediction_batch,
                    pred_dicts=trt_pred,
                    class_names=cfg.CLASS_NAMES,
                    output_path=pred_dir,
                )
                if annos:
                    per_frame_predictions.append(
                        {
                            "frame_id": frame_id,
                            "detection_count": int(len(annos[0]["name"])),
                            "score_mean": float(np.mean(annos[0]["score"])) if len(annos[0]["score"]) > 0 else None,
                        }
                    )

    bucket_summary_rows = summarize_bucket_run_rows(frame_rows)
    build_summary_rows = []
    for build_row in build_rows:
        matching = next((row for row in bucket_summary_rows if row["bucket_size"] == build_row["bucket_size"]), None)
        build_summary_rows.append({**build_row, **(matching or {})})

    latency_fieldnames = [
        "bucket_size",
        "build_success",
        "status",
        "engine_size_mb",
        "build_ms",
        "memory_usage_mb",
        "frame_count",
        "bucket_hit_count",
        "core_latency_mean",
        "core_latency_p50",
        "core_latency_p95",
        "online_wrapper_latency_mean",
        "online_wrapper_latency_p50",
        "online_wrapper_latency_p95",
        "nms_postprocess_latency_mean",
        "output_box_count_mean",
        "score_mean",
    ]
    truncation_fieldnames = [
        "bucket_size",
        "frame_count",
        "bucket_hit_count",
        "truncation_frame_count",
        "truncation_rate",
        "mean_truncated_pillars",
        "pillar_count_mean",
        "pillar_count_p50",
        "pillar_count_p95",
        "pillar_count_max",
    ]

    write_csv(
        latency_csv,
        [{key: row.get(key) for key in latency_fieldnames} for row in build_summary_rows],
        latency_fieldnames,
    )
    write_csv(
        truncation_csv,
        [{key: row.get(key) for key in truncation_fieldnames} for row in build_summary_rows],
        truncation_fieldnames,
    )

    save_image(
        latency_fig,
        draw_bar_chart(
            [{"stage": f"{row['bucket_size']} pillars", "mean_ms": row.get("online_wrapper_latency_mean")} for row in build_summary_rows],
            title="TensorRT Bucketed Wrapper Latency vs Capacity",
        ),
    )
    hit_distribution = bucket_hit_distribution(frame_rows)
    save_image(
        bucket_hit_fig,
        draw_bar_chart(
            [{"stage": f"{bucket}", "mean_ms": hit_distribution.get(bucket, 0)} for bucket in successful_buckets],
            title="TensorRT Bucket Hit Distribution",
        ),
    )

    official_eval_payload: dict | None = None
    if args.run_official_eval:
        full_eval_frame_ids = load_split_ids(split_file, args.eval_max_frames)
        if set(full_eval_frame_ids) - {row["frame_id"] for row in frame_rows}:
            # Re-run remaining frames for prediction export only.
            with torch.no_grad():
                for frame_id in full_eval_frame_ids:
                    pred_txt = pred_dir / f"{frame_id}.txt"
                    if pred_txt.exists():
                        continue
                    assets = resolve_frame_assets(kitti_root, frame_id)
                    points = np.fromfile(str(assets.lidar_path), dtype=np.float32).reshape(-1, 4)
                    prepared, _ = dataset.prepare_timed(points, frame_id)
                    full_pillar_count = int(prepared["voxels"].shape[0])
                    bucket_size = select_bucket_size(full_pillar_count, successful_buckets)
                    capped = pad_bucket_inputs(prepared, bucket_size)
                    batch_dict = dataset.collate_batch([capped])
                    load_data_to_gpu(batch_dict)
                    engine, context = engines[bucket_size]
                    shapes = engine_shapes[bucket_size]
                    buffers = {
                        "voxels": batch_dict["voxels"].contiguous(),
                        "voxel_num_points": batch_dict["voxel_num_points"].contiguous(),
                        "voxel_coords": batch_dict["voxel_coords"].contiguous(),
                        "batch_cls_preds": torch.empty(tuple(shapes["batch_cls_preds"]), device="cuda", dtype=torch.float32),
                        "batch_box_preds": torch.empty(tuple(shapes["batch_box_preds"]), device="cuda", dtype=torch.float32),
                    }
                    for name, tensor in buffers.items():
                        if name in shapes:
                            context.set_tensor_address(name, int(tensor.data_ptr()))
                    stream = torch.cuda.Stream()
                    ok = context.execute_async_v3(stream.cuda_stream)
                    stream.synchronize()
                    if not ok:
                        raise RuntimeError("TensorRT execute_async_v3 returned false during eval export")
                    trt_batch = dict(batch_dict)
                    trt_batch["batch_cls_preds"] = buffers["batch_cls_preds"]
                    trt_batch["batch_box_preds"] = buffers["batch_box_preds"]
                    trt_batch["cls_preds_normalized"] = False
                    trt_pred, _ = model.post_processing(trt_batch)
                    calib = calibration_kitti.Calibration(assets.calib_path)
                    image_shape = read_image_shape(assets.image_path)
                    prediction_batch = {
                        "frame_id": [frame_id],
                        "calib": [calib],
                        "image_shape": torch.from_numpy(np.expand_dims(image_shape, axis=0)),
                    }
                    KittiDataset.generate_prediction_dicts(
                        batch_dict=prediction_batch,
                        pred_dicts=trt_pred,
                        class_names=cfg.CLASS_NAMES,
                        output_path=pred_dir,
                    )
        runner = PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "wsl_kitti_eval_runner.py"
        raw_eval_json = output_dir / "tensorrt_bucketed_kitti_eval_raw.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(runner),
                "--openpcdet-root",
                str(openpcdet_root),
                "--label-dir",
                str((kitti_root / "training" / "label_2").resolve()),
                "--pred-dir",
                str(pred_dir),
                "--split-file",
                str(split_file),
                "--output-json",
                str(raw_eval_json),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(PROJECT_ROOT),
        )
        if completed.returncode == 0 and raw_eval_json.exists():
            raw_eval = json.loads(raw_eval_json.read_text(encoding="utf-8"))
            baseline = json.loads((output_dir / "kitti_official_eval.json").read_text(encoding="utf-8-sig")) if (output_dir / "kitti_official_eval.json").exists() else {}
            base_dict = baseline.get("official_result_dict", {}) if isinstance(baseline, dict) else {}
            trt_dict = raw_eval.get("official_result_dict", {})
            zero_ap = all(float(trt_dict.get(key, 0.0) or 0.0) == 0.0 for key in ["Car_3d/moderate_R40", "Pedestrian_3d/moderate_R40", "Cyclist_3d/moderate_R40"])
            official_eval_payload = {
                "status": "completed_with_blocker" if zero_ap else "completed",
                "frame_count": len(full_eval_frame_ids),
                "prediction_dir": str(pred_dir),
                "evaluation_backend": raw_eval.get("evaluation_backend"),
                "official_result_text": raw_eval.get("official_result_text", ""),
                "official_result_dict": trt_dict,
                "ap_delta": {
                    key: (trt_dict.get(key) - base_dict.get(key)) if (trt_dict.get(key) is not None and base_dict.get(key) is not None) else None
                    for key in ["Car_3d/moderate_R40", "Pedestrian_3d/moderate_R40", "Cyclist_3d/moderate_R40"]
                },
                "latency_reference": {
                    "bucketed_wrapper_online_mean_ms": summarize_values(row["online_total_ms"] for row in frame_rows)["mean"],
                    "pytorch_wrapper_online_mean_ms": summarize_values(row["pytorch_online_total_ms"] for row in frame_rows)["mean"],
                },
                "truncation_reference": {
                    "frames_profiled": len(frame_rows),
                    "truncation_frame_count": sum(1 for row in frame_rows if row["pillar_truncation_applied"]),
                },
                "blocker": (
                    "Bucketed TensorRT wrapper exported KITTI predictions but produced zero AP on the 200-frame eval slice. "
                    "This suggests a remaining postprocess / tensor semantic mismatch, so the bucketed eval should not be treated as deployment-ready accuracy."
                    if zero_ap
                    else None
                ),
            }
        else:
            official_eval_payload = {
                "status": "skipped",
                "reason": "Bucketed TensorRT wrapper official eval failed.",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        write_json(eval_json_path, official_eval_payload)
        write_markdown(
            eval_md_path,
            "# TensorRT Bucketed KITTI Eval\n\n"
            f"- Status: `{official_eval_payload.get('status')}`\n"
            f"- Car moderate 3D AP_R40: `{official_eval_payload.get('official_result_dict', {}).get('Car_3d/moderate_R40')}`\n"
            f"- Pedestrian moderate 3D AP_R40: `{official_eval_payload.get('official_result_dict', {}).get('Pedestrian_3d/moderate_R40')}`\n"
            f"- Cyclist moderate 3D AP_R40: `{official_eval_payload.get('official_result_dict', {}).get('Cyclist_3d/moderate_R40')}`\n",
        )

    report_payload = {
        "status": "completed",
        "scope": "real KITTI point cloud -> OpenPCDet preprocessing -> bucketed fixed-shape TensorRT PointPillars core -> OpenPCDet postprocess/NMS",
        "bucket_sizes": bucket_sizes,
        "successful_bucket_sizes": successful_buckets,
        "engine_build_rows": build_rows,
        "bucket_summary_rows": build_summary_rows,
        "bucket_hit_distribution": {str(key): value for key, value in hit_distribution.items()},
        "frame_count": len(frame_rows),
        "frame_rows": frame_rows,
        "overall_pillar_count": summarize_values(row["full_pillar_count"] for row in frame_rows),
        "overall_truncation_frame_count": sum(1 for row in frame_rows if row["pillar_truncation_applied"]),
        "overall_truncation_rate": sum(1 for row in frame_rows if row["pillar_truncation_applied"]) / len(frame_rows) if frame_rows else None,
        "overall_mean_truncated_pillars": statistics.fmean(row["truncated_pillars"] for row in frame_rows) if frame_rows else None,
        "mean_trt_core_ms": statistics.fmean(row["trt_core_ms"] for row in frame_rows) if frame_rows else None,
        "mean_pytorch_core_ms": statistics.fmean(row["pytorch_core_ms"] for row in frame_rows) if frame_rows else None,
        "mean_online_total_ms": statistics.fmean(row["online_total_ms"] for row in frame_rows) if frame_rows else None,
        "mean_pytorch_online_total_ms": statistics.fmean(row["pytorch_online_total_ms"] for row in frame_rows) if frame_rows else None,
        "mean_speedup_core_only": statistics.fmean(row["speedup_core_only"] for row in frame_rows if row["speedup_core_only"] is not None) if frame_rows else None,
        "mean_speedup_online_total_if_any": statistics.fmean(row["speedup_online_total_if_any"] for row in frame_rows if row["speedup_online_total_if_any"] is not None) if frame_rows else None,
        "dynamic_shape_audit": {
            "status": "skipped",
            "reason": "This round focused on realistic fixed-shape/bucketed core capacity. Dynamic-shape TensorRT export remains blocked by fixed-shape PointPillarScatter export in the current project.",
        },
        "official_eval": official_eval_payload,
        "limitations": [
            "TensorRT engine covers the PointPillars core only; voxelization and NMS remain outside TensorRT.",
            "Bucketed fixed-shape engines reduce truncation risk but do not make this a full dynamic TensorRT detector.",
            "No FP16 acceleration is claimed here.",
        ],
    }
    write_json(report_json, report_payload)
    write_markdown(
        report_md,
        "# TensorRT Bucketed Core Report\n\n"
        f"- Status: `{report_payload['status']}`\n"
        f"- Bucket sizes: `{bucket_sizes}`\n"
        f"- Successful buckets: `{successful_buckets}`\n"
        f"- Overall truncation rate: `{report_payload['overall_truncation_rate']}`\n"
        f"- Mean TRT core ms: `{report_payload['mean_trt_core_ms']}`\n"
        f"- Mean PyTorch core ms: `{report_payload['mean_pytorch_core_ms']}`\n"
        f"- Mean wrapper online ms: `{report_payload['mean_online_total_ms']}`\n"
        f"- Mean PyTorch wrapper online ms: `{report_payload['mean_pytorch_online_total_ms']}`\n"
        "\n## Limitations\n\n"
        "- TensorRT still covers core only.\n"
        "- Voxelization and NMS remain outside the engine.\n"
        "- Bucketed fixed-shape capacity analysis must not be described as a full dynamic TensorRT detector.\n",
    )
    print(f"Saved TensorRT bucketed core report: {report_json}")


if __name__ == "__main__":
    main()
