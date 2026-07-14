from __future__ import annotations

import argparse
import json
import os
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
from runtime.lidar_system_algorithm.visualization import draw_bar_chart, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile PyTorch vs backbone/head-only TRT online pipeline.")
    parser.add_argument("--kitti-root", default="data/kitti_object_raw/extracted")
    parser.add_argument("--openpcdet-root", default="external/OpenPCDet")
    parser.add_argument("--cfg-file", default="external/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml")
    parser.add_argument("--ckpt", default="checkpoints/pointpillar_kitti.pth")
    parser.add_argument("--split-file", default="external/OpenPCDet/data/kitti/ImageSets/val.txt")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _read_split_ids(path: Path, max_frames: int) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids[:max_frames] if max_frames > 0 else ids


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tensorrt_backbone_head_online_latency.json"
    md_path = output_dir / "tensorrt_backbone_head_online_latency.md"
    csv_path = output_dir / "tensorrt_backbone_head_online_latency.csv"
    fig_path = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures" / "trt_backbone_head_online_latency_breakdown.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import tensorrt as trt
        import torch
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils
    except Exception as exc:
        payload = {"status": "skipped", "reason": f"runtime import failed: {type(exc).__name__}: {exc}"}
        write_json(json_path, payload)
        write_markdown(md_path, f"# Online Latency Profile\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
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
    engine_path = artifact_dir / "pointpillar_backbone_head_only.engine"
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    split_ids = _read_split_ids(_resolve(args.split_file), args.frames)
    rows = []
    for frame_id in split_ids:
        assets = resolve_frame_assets(_resolve(args.kitti_root), frame_id)
        load_t0 = time.perf_counter()
        points = np.fromfile(str(assets.lidar_path), dtype=np.float32).reshape(-1, 4)
        load_ms = (time.perf_counter() - load_t0) * 1000.0

        prep_t0 = time.perf_counter()
        prepared = dataset.prepare_runtime(points, frame_id)
        batch_dict = dataset.collate_batch([prepared])
        load_data_to_gpu(batch_dict)
        sync()
        preprocess_ms = (time.perf_counter() - prep_t0) * 1000.0

        with torch.no_grad():
            vfe_t0 = time.perf_counter()
            vfe_batch = modules[0](dict(batch_dict))
            sync()
            vfe_ms = (time.perf_counter() - vfe_t0) * 1000.0

            scatter_t0 = time.perf_counter()
            scatter_batch = modules[1](vfe_batch)
            sync()
            scatter_ms = (time.perf_counter() - scatter_t0) * 1000.0
            spatial_features = scatter_batch["spatial_features"].contiguous()

            py_core_t0 = time.perf_counter()
            py_forward, _ = run_pointpillars_modules(modules[2:], dict(scatter_batch), capture_stages=False)
            sync()
            py_backbone_head_ms = (time.perf_counter() - py_core_t0) * 1000.0

            py_post_t0 = time.perf_counter()
            model.post_processing(py_forward)
            sync()
            py_post_ms = (time.perf_counter() - py_post_t0) * 1000.0

            trt_outputs = {
                "batch_cls_preds": torch.empty((1, 321408, len(cfg.CLASS_NAMES)), device="cuda", dtype=torch.float32),
                "batch_box_preds": torch.empty((1, 321408, 7), device="cuda", dtype=torch.float32),
            }
            for name, tensor in {"spatial_features": spatial_features, **trt_outputs}.items():
                context.set_tensor_address(name, int(tensor.data_ptr()))
            stream = torch.cuda.Stream()
            sync()
            trt_core_t0 = time.perf_counter()
            ok = context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            sync()
            trt_backbone_head_ms = (time.perf_counter() - trt_core_t0) * 1000.0
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned false")

            trt_batch = dict(scatter_batch)
            trt_batch["batch_cls_preds"] = trt_outputs["batch_cls_preds"]
            trt_batch["batch_box_preds"] = trt_outputs["batch_box_preds"]
            trt_batch["cls_preds_normalized"] = False
            trt_post_t0 = time.perf_counter()
            model.post_processing(trt_batch)
            sync()
            trt_post_ms = (time.perf_counter() - trt_post_t0) * 1000.0

        py_total = load_ms + preprocess_ms + vfe_ms + scatter_ms + py_backbone_head_ms + py_post_ms
        trt_total = load_ms + preprocess_ms + vfe_ms + scatter_ms + trt_backbone_head_ms + trt_post_ms
        rows.append(
            {
                "frame_id": frame_id,
                "load_ms": load_ms,
                "preprocess_voxelization_ms": preprocess_ms,
                "vfe_ms": vfe_ms,
                "scatter_ms": scatter_ms,
                "pytorch_backbone_head_ms": py_backbone_head_ms,
                "trt_backbone_head_ms": trt_backbone_head_ms,
                "pytorch_postprocess_nms_ms": py_post_ms,
                "trt_postprocess_nms_ms": trt_post_ms,
                "kitti_export_ms": 0.0,
                "tracking_ms": 0.0,
                "pytorch_online_total_ms": py_total,
                "trt_online_total_ms": trt_total,
                "visualization_ms": 0.0,
            }
        )

    payload = {
        "status": "completed",
        "frame_count": len(rows),
        "visualization_excluded": True,
        "pytorch_summary": {
            "online_total_ms": _summary([row["pytorch_online_total_ms"] for row in rows]),
            "backbone_head_ms": _summary([row["pytorch_backbone_head_ms"] for row in rows]),
            "postprocess_nms_ms": _summary([row["pytorch_postprocess_nms_ms"] for row in rows]),
        },
        "trt_summary": {
            "online_total_ms": _summary([row["trt_online_total_ms"] for row in rows]),
            "backbone_head_ms": _summary([row["trt_backbone_head_ms"] for row in rows]),
            "postprocess_nms_ms": _summary([row["trt_postprocess_nms_ms"] for row in rows]),
        },
        "shared_stage_summary": {
            "load_ms": _summary([row["load_ms"] for row in rows]),
            "preprocess_voxelization_ms": _summary([row["preprocess_voxelization_ms"] for row in rows]),
            "vfe_ms": _summary([row["vfe_ms"] for row in rows]),
            "scatter_ms": _summary([row["scatter_ms"] for row in rows]),
        },
        "speedup": {
            "core_only": (
                _summary([row["pytorch_backbone_head_ms"] for row in rows])["mean"] / _summary([row["trt_backbone_head_ms"] for row in rows])["mean"]
            ),
            "online_total": (
                _summary([row["pytorch_online_total_ms"] for row in rows])["mean"] / _summary([row["trt_online_total_ms"] for row in rows])["mean"]
            ),
        },
        "rows_preview": rows[:20],
    }

    chart_rows = [
        {"stage": "PyTorch online_total", "mean_ms": payload["pytorch_summary"]["online_total_ms"]["mean"]},
        {"stage": "TRT online_total", "mean_ms": payload["trt_summary"]["online_total_ms"]["mean"]},
        {"stage": "PyTorch backbone_head", "mean_ms": payload["pytorch_summary"]["backbone_head_ms"]["mean"]},
        {"stage": "TRT backbone_head", "mean_ms": payload["trt_summary"]["backbone_head_ms"]["mean"]},
        {"stage": "Shared VFE", "mean_ms": payload["shared_stage_summary"]["vfe_ms"]["mean"]},
        {"stage": "Shared scatter", "mean_ms": payload["shared_stage_summary"]["scatter_ms"]["mean"]},
        {"stage": "TRT postprocess", "mean_ms": payload["trt_summary"]["postprocess_nms_ms"]["mean"]},
    ]
    save_image(fig_path, draw_bar_chart(chart_rows, title="Backbone/Head-only TRT Online Latency Breakdown"))

    write_json(json_path, payload)
    write_csv(csv_path, rows, fieldnames=list(rows[0].keys()) if rows else ["frame_id"])
    write_markdown(
        md_path,
        "# Backbone/Head-only TRT Online Latency\n\n"
        f"- Visualization excluded: `{payload['visualization_excluded']}`\n"
        f"- PyTorch online_total mean/p50/p95: `{payload['pytorch_summary']['online_total_ms']['mean']}` / `{payload['pytorch_summary']['online_total_ms']['p50']}` / `{payload['pytorch_summary']['online_total_ms']['p95']}`\n"
        f"- TRT online_total mean/p50/p95: `{payload['trt_summary']['online_total_ms']['mean']}` / `{payload['trt_summary']['online_total_ms']['p50']}` / `{payload['trt_summary']['online_total_ms']['p95']}`\n"
        f"- PyTorch backbone/head mean: `{payload['pytorch_summary']['backbone_head_ms']['mean']}`\n"
        f"- TRT backbone/head mean: `{payload['trt_summary']['backbone_head_ms']['mean']}`\n"
        f"- Core-only speedup: `{payload['speedup']['core_only']}`\n"
        f"- Online speedup: `{payload['speedup']['online_total']}`\n",
    )
    print(json.dumps({"status": "completed", "report": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
