from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.kitti_io import resolve_frame_assets
from runtime.lidar_system_algorithm.pointpillars_wrapper_runtime import run_pointpillars_modules
from runtime.lidar_system_algorithm.report_schema import write_csv, write_json, write_markdown
from runtime.lidar_system_algorithm.tensorrt_debug_utils import prediction_dir_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TensorRT backbone/head-only PointPillars path.")
    parser.add_argument("--kitti-root", default="data/kitti_object_raw/extracted")
    parser.add_argument("--openpcdet-root", default="external/OpenPCDet")
    parser.add_argument("--cfg-file", default="external/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml")
    parser.add_argument("--ckpt", default="checkpoints/pointpillar_kitti.pth")
    parser.add_argument("--eval-max-frames", type=int, default=200)
    parser.add_argument("--split-file", default="external/OpenPCDet/data/kitti/ImageSets/val.txt")
    parser.add_argument("--pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt_trt_backbone_head_only")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--report-suffix", default="", help="Optional suffix for report filenames, for example _1000.")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _read_split_ids(path: Path, max_frames: int) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids[:max_frames] if max_frames > 0 else ids


def _read_image_shape(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return np.array(image.shape[:2], dtype=np.int32)


def _numeric_summary(rows: list[dict], key: str) -> dict[str, float | int | None]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.report_suffix
    report_json = output_dir / f"tensorrt_backbone_head_only_eval{suffix}.json"
    report_md = output_dir / f"tensorrt_backbone_head_only_eval{suffix}.md"
    latency_csv = output_dir / f"tensorrt_backbone_head_only_latency{suffix}.csv"
    diff_csv = output_dir / f"tensorrt_backbone_head_only_diff{suffix}.csv"
    pred_dir = _resolve(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    try:
        import tensorrt as trt
        import torch
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.datasets.kitti.kitti_dataset import KittiDataset
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import calibration_kitti, common_utils
    except Exception as exc:
        payload = {"status": "skipped", "reason": f"runtime import failed: {type(exc).__name__}: {exc}"}
        write_json(report_json, payload)
        write_markdown(report_md, f"# TensorRT Backbone/Head-only Eval\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
        return

    openpcdet_root = _resolve(args.openpcdet_root)
    os.chdir(openpcdet_root / "tools")
    cfg_from_yaml_file(str(_resolve(args.cfg_file)), cfg)

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

        def prepare_runtime(self, points: np.ndarray, frame_id: str) -> dict:
            data_dict = {"points": points, "frame_id": frame_id}
            data_dict = self.set_lidar_aug_matrix(data_dict)
            data_dict = self.point_feature_encoder.forward(data_dict)
            for processor in self.data_processor.data_processor_queue:
                data_dict = processor(data_dict=data_dict)
            data_dict.pop("gt_names", None)
            return data_dict

    class BackboneHeadOnly(torch.nn.Module):
        def __init__(self, backbone, dense_head):
            super().__init__()
            self.backbone = backbone
            self.dense_head = dense_head

        def forward(self, spatial_features):
            batch_dict = {"spatial_features": spatial_features, "batch_size": 1}
            batch_dict = self.backbone(batch_dict)
            batch_dict = self.dense_head(batch_dict)
            return batch_dict["batch_cls_preds"], batch_dict["batch_box_preds"]

    dataset = RuntimeDataset()
    logger = common_utils.create_logger()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(Path(args.ckpt).expanduser()), logger=logger, to_cpu=True)
    model.cuda().eval()
    modules = torch.nn.ModuleList(list(model.module_list)).cuda().eval()
    backbone_head = BackboneHeadOnly(modules[2], modules[3]).cuda().eval()

    artifact_dir = output_dir / "backbone_head_only"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = artifact_dir / "pointpillar_backbone_head_only.onnx"
    engine_path = artifact_dir / "pointpillar_backbone_head_only.engine"

    split_ids = _read_split_ids(_resolve(args.split_file), args.eval_max_frames)
    sample_id = split_ids[0]
    assets = resolve_frame_assets(_resolve(args.kitti_root), sample_id)
    sample_points = np.fromfile(str(assets.lidar_path), dtype=np.float32).reshape(-1, 4)
    sample_prepared = dataset.prepare_runtime(sample_points, sample_id)
    sample_batch = dataset.collate_batch([sample_prepared])
    load_data_to_gpu(sample_batch)
    with torch.no_grad():
        sample_forward, sample_stages = run_pointpillars_modules(modules[:2], sample_batch, capture_stages=True)
        sample_scatter = sample_stages["after_scatter"]["spatial_features"].contiguous()
    if args.force_rebuild or not onnx_path.exists():
        torch.onnx.export(
            backbone_head,
            (sample_scatter,),
            str(onnx_path),
            opset_version=17,
            dynamo=False,
            input_names=["spatial_features"],
            output_names=["batch_cls_preds", "batch_box_preds"],
        )
    if args.force_rebuild or not engine_path.exists():
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, trt_logger)
        if not parser.parse(onnx_path.read_bytes()):
            errors = " | ".join(parser.get_error(i).desc() for i in range(parser.num_errors))
            payload = {"status": "skipped", "reason": f"TensorRT parse failed: {errors}"}
            write_json(report_json, payload)
            write_markdown(report_md, f"# TensorRT Backbone/Head-only Eval\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
            return
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            payload = {"status": "skipped", "reason": "TensorRT build_serialized_network returned None"}
            write_json(report_json, payload)
            write_markdown(report_md, f"# TensorRT Backbone/Head-only Eval\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
            return
        engine_path.write_bytes(bytes(serialized))

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()

    temp_split = output_dir / f"trt_backbone_head_only_split_{len(split_ids)}{suffix}.txt"
    temp_split.write_text("\n".join(split_ids) + "\n", encoding="utf-8")
    frame_rows = []
    with torch.no_grad():
        for frame_id in split_ids:
            assets = resolve_frame_assets(_resolve(args.kitti_root), frame_id)
            points = np.fromfile(str(assets.lidar_path), dtype=np.float32).reshape(-1, 4)
            prepared = dataset.prepare_runtime(points, frame_id)
            batch_dict = dataset.collate_batch([prepared])
            load_data_to_gpu(batch_dict)

            stage0 = time.perf_counter()
            pre_batch, stages = run_pointpillars_modules(modules[:2], batch_dict, capture_stages=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            preprocess_ms = (time.perf_counter() - stage0) * 1000.0
            spatial_features = stages["after_scatter"]["spatial_features"].contiguous()

            trt_inputs = {"spatial_features": spatial_features}
            trt_outputs = {
                "batch_cls_preds": torch.empty((1, 321408, len(cfg.CLASS_NAMES)), device="cuda", dtype=torch.float32),
                "batch_box_preds": torch.empty((1, 321408, 7), device="cuda", dtype=torch.float32),
            }
            for name, tensor in {**trt_inputs, **trt_outputs}.items():
                context.set_tensor_address(name, int(tensor.data_ptr()))
            stream = torch.cuda.Stream()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            ok = context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            trt_core_ms = (time.perf_counter() - start) * 1000.0
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned false")

            py_stage0 = time.perf_counter()
            py_forward, _ = run_pointpillars_modules(modules[2:], dict(stages["after_scatter"]), capture_stages=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pytorch_core_ms = (time.perf_counter() - py_stage0) * 1000.0

            trt_batch = dict(pre_batch)
            trt_batch["spatial_features"] = spatial_features
            trt_batch["batch_cls_preds"] = trt_outputs["batch_cls_preds"]
            trt_batch["batch_box_preds"] = trt_outputs["batch_box_preds"]
            trt_batch["cls_preds_normalized"] = False

            post_start = time.perf_counter()
            trt_pred, _ = model.post_processing(trt_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            post_ms = (time.perf_counter() - post_start) * 1000.0

            calib = calibration_kitti.Calibration(assets.calib_path)
            image_shape = _read_image_shape(assets.image_path)
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

            py_pred, _ = model.post_processing(py_forward)
            py_scores = py_pred[0]["pred_scores"].detach().cpu().numpy()
            trt_scores = trt_pred[0]["pred_scores"].detach().cpu().numpy()
            frame_rows.append(
                {
                    "frame_id": frame_id,
                    "pytorch_box_count": int(py_pred[0]["pred_boxes"].shape[0]),
                    "trt_box_count": int(trt_pred[0]["pred_boxes"].shape[0]),
                    "empty_prediction_file": int(len(annos[0]["name"]) == 0),
                    "point_preprocess_ms": preprocess_ms,
                    "trt_core_ms": trt_core_ms,
                    "pytorch_core_ms": pytorch_core_ms,
                    "nms_postprocess_ms": post_ms,
                    "trt_score_mean": float(np.mean(trt_scores)) if trt_scores.size else None,
                    "pytorch_score_mean": float(np.mean(py_scores)) if py_scores.size else None,
                }
            )

    runner = PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "wsl_kitti_eval_runner.py"
    raw_eval_json = output_dir / f"tensorrt_backbone_head_only_eval_raw{suffix}.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--openpcdet-root",
            str(openpcdet_root),
            "--label-dir",
            str((_resolve(args.kitti_root) / "training" / "label_2").resolve()),
            "--pred-dir",
            str(pred_dir),
            "--split-file",
            str(temp_split),
            "--output-json",
            str(raw_eval_json),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(PROJECT_ROOT),
    )
    pred_stats = prediction_dir_stats(pred_dir)
    if completed.returncode == 0 and raw_eval_json.exists():
        raw_eval = json.loads(raw_eval_json.read_text(encoding="utf-8"))
        result_dict = raw_eval.get("official_result_dict", {})
        payload = {
            "status": "completed",
            "frame_count": len(split_ids),
            "prediction_dir": str(pred_dir),
            "engine_path": str(engine_path),
            "official_result_dict": result_dict,
            "official_eval_status": raw_eval.get("status"),
            "evaluation_backend": raw_eval.get("evaluation_backend"),
            "latency_summary": {
                "trt_core_ms": _numeric_summary(frame_rows, "trt_core_ms"),
                "pytorch_core_ms": _numeric_summary(frame_rows, "pytorch_core_ms"),
                "preprocess_ms": _numeric_summary(frame_rows, "point_preprocess_ms"),
                "nms_postprocess_ms": _numeric_summary(frame_rows, "nms_postprocess_ms"),
            },
            **pred_stats,
            "frame_rows_preview": frame_rows[:20],
            "blocker": None,
        }
    else:
        payload = {
            "status": "skipped",
            "frame_count": len(split_ids),
            "prediction_dir": str(pred_dir),
            **pred_stats,
            "blocker": f"backbone/head-only official eval failed: {completed.stderr.strip() or completed.stdout.strip()}",
        }

    write_json(report_json, payload)
    write_csv(latency_csv, frame_rows, fieldnames=list(frame_rows[0].keys()) if frame_rows else ["frame_id"])
    write_csv(diff_csv, frame_rows, fieldnames=list(frame_rows[0].keys()) if frame_rows else ["frame_id"])
    write_markdown(
        report_md,
        "# TensorRT Backbone/Head-only Eval\n\n"
        f"- Status: `{payload['status']}`\n"
        f"- Empty prediction files: `{payload.get('empty_prediction_file_count')}`\n"
        f"- Invalid geometry count: `{payload.get('invalid_geometry_count')}`\n"
        f"- TRT core mean/p50/p95 ms: `{payload.get('latency_summary', {}).get('trt_core_ms', {}).get('mean')}` / `{payload.get('latency_summary', {}).get('trt_core_ms', {}).get('p50')}` / `{payload.get('latency_summary', {}).get('trt_core_ms', {}).get('p95')}`\n"
        f"- PyTorch core mean/p50/p95 ms: `{payload.get('latency_summary', {}).get('pytorch_core_ms', {}).get('mean')}` / `{payload.get('latency_summary', {}).get('pytorch_core_ms', {}).get('p50')}` / `{payload.get('latency_summary', {}).get('pytorch_core_ms', {}).get('p95')}`\n"
        f"- Car/Ped/Cyc moderate 3D AP_R40: `{payload.get('official_result_dict', {}).get('Car_3d/moderate_R40')}` / `{payload.get('official_result_dict', {}).get('Pedestrian_3d/moderate_R40')}` / `{payload.get('official_result_dict', {}).get('Cyclist_3d/moderate_R40')}`\n"
        f"- Blocker: `{payload.get('blocker')}`\n",
    )
    print(json.dumps({"status": payload["status"], "report": str(report_json)}, indent=2))


if __name__ == "__main__":
    main()
