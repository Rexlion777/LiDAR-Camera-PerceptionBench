from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.kitti_io import resolve_frame_assets
from runtime.lidar_system_algorithm.report_schema import write_csv, write_json, write_markdown
from runtime.lidar_system_algorithm.tensorrt_debug_utils import pad_prepared_inputs, summarize_diff, summarize_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TensorRT submodule bisection on PointPillars bucketed core.")
    parser.add_argument("--kitti-root", default="data/kitti_object_raw/extracted")
    parser.add_argument("--openpcdet-root", default="external/OpenPCDet")
    parser.add_argument("--cfg-file", default="external/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml")
    parser.add_argument("--ckpt", default="checkpoints/pointpillar_kitti.pth")
    parser.add_argument("--bucket-size", type=int, default=8192)
    parser.add_argument("--frame-ids", default="000002,000003")
    parser.add_argument("--padding-strategy", default="repeat_first_valid", choices=["repeat_first_valid", "duplicate_zero_coord_padding", "unique_dummy_coord_padding"])
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm")
    parser.add_argument("--report-prefix", default="tensorrt_submodule_bisection")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def main() -> None:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.report_prefix}_report.json"
    md_path = output_dir / f"{args.report_prefix}_report.md"
    csv_path = output_dir / f"{args.report_prefix}_tensor_diff.csv"
    artifact_dir = output_dir / "submodule_engines"
    artifact_dir.mkdir(parents=True, exist_ok=True)

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
        write_markdown(md_path, f"# TensorRT Submodule Bisection\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
        return

    kitti_root = _resolve(args.kitti_root)
    openpcdet_root = _resolve(args.openpcdet_root)
    cfg_file = _resolve(args.cfg_file)
    ckpt_path = Path(args.ckpt).expanduser()
    frame_ids = [item.strip() for item in args.frame_ids.split(",") if item.strip()]

    os.chdir(openpcdet_root / "tools")
    cfg_from_yaml_file(str(cfg_file), cfg)

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

    class VFEOnly(torch.nn.Module):
        def __init__(self, vfe):
            super().__init__()
            self.vfe = vfe

        def forward(self, voxels, voxel_num_points, voxel_coords):
            batch_dict = {"voxels": voxels, "voxel_num_points": voxel_num_points, "voxel_coords": voxel_coords, "batch_size": 1}
            batch_dict = self.vfe(batch_dict)
            return batch_dict["pillar_features"]

    class VFEScatter(torch.nn.Module):
        def __init__(self, vfe, scatter):
            super().__init__()
            self.vfe = vfe
            self.scatter = scatter

        def forward(self, voxels, voxel_num_points, voxel_coords):
            batch_dict = {"voxels": voxels, "voxel_num_points": voxel_num_points, "voxel_coords": voxel_coords, "batch_size": 1}
            batch_dict = self.vfe(batch_dict)
            batch_dict = self.scatter(batch_dict)
            return batch_dict["spatial_features"]

    class BackboneHeadFromSpatial(torch.nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, spatial_features):
            batch_dict = {"spatial_features": spatial_features, "batch_size": 1}
            batch_dict = self.backbone(batch_dict)
            batch_dict = self.head(batch_dict)
            return batch_dict["batch_cls_preds"], batch_dict["batch_box_preds"]

    class FullCore(torch.nn.Module):
        def __init__(self, modules):
            super().__init__()
            self.modules_for_export = torch.nn.ModuleList(list(modules))

        def forward(self, voxels, voxel_num_points, voxel_coords):
            batch_dict = {"voxels": voxels, "voxel_num_points": voxel_num_points, "voxel_coords": voxel_coords, "batch_size": 1}
            for module in self.modules_for_export:
                batch_dict = module(batch_dict)
            return batch_dict["batch_cls_preds"], batch_dict["batch_box_preds"]

    def build_engine(module, sample_inputs, input_names, output_names, stem: str):
        onnx_path = artifact_dir / f"{stem}.onnx"
        engine_path = artifact_dir / f"{stem}.engine"
        if not onnx_path.exists():
            torch.onnx.export(
                module,
                sample_inputs,
                str(onnx_path),
                opset_version=17,
                dynamo=False,
                input_names=input_names,
                output_names=output_names,
            )
        if not engine_path.exists():
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, logger)
            if not parser.parse(onnx_path.read_bytes()):
                errors = " | ".join(parser.get_error(i).desc() for i in range(parser.num_errors))
                raise RuntimeError(f"TRT parse failed for {stem}: {errors}")
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError(f"TRT build failed for {stem}")
            engine_path.write_bytes(bytes(serialized))
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        return engine, engine.create_execution_context(), onnx_path, engine_path

    def trt_forward(engine, context, tensor_map: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        output = {}
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            shape = tuple(engine.get_tensor_shape(name))
            mode = str(engine.get_tensor_mode(name))
            if "INPUT" in mode:
                context.set_tensor_address(name, int(tensor_map[name].contiguous().data_ptr()))
            else:
                output_tensor = torch.empty(shape, device="cuda", dtype=torch.float32)
                tensor_map[name] = output_tensor
                context.set_tensor_address(name, int(output_tensor.data_ptr()))
        stream = torch.cuda.Stream()
        ok = context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if "OUTPUT" in str(engine.get_tensor_mode(name)):
                output[name] = tensor_map[name].detach().cpu().numpy()
        return output

    dataset = RuntimeDataset()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_path), logger=common_utils.create_logger(), to_cpu=True)
    model.cuda().eval()
    vfe_mod, scatter_mod, backbone_mod, head_mod = model.module_list
    vfe_only = VFEOnly(vfe_mod).cuda().eval()
    vfe_scatter = VFEScatter(vfe_mod, scatter_mod).cuda().eval()
    backbone_head = BackboneHeadFromSpatial(backbone_mod, head_mod).cuda().eval()
    full_core = FullCore(model.module_list).cuda().eval()

    prepared_rows = []
    for frame_id in frame_ids:
        assets = resolve_frame_assets(kitti_root, frame_id)
        points = np.fromfile(str(assets.lidar_path), dtype=np.float32).reshape(-1, 4)
        prepared = dataset.prepare_runtime(points, frame_id)
        padded, pad_stats = pad_prepared_inputs(prepared, args.bucket_size, args.padding_strategy)
        batch_dict = dataset.collate_batch([padded])
        load_data_to_gpu(batch_dict)
        prepared_rows.append((frame_id, batch_dict, pad_stats))

    sample_batch = prepared_rows[0][1]
    sample_inputs = (
        sample_batch["voxels"].contiguous(),
        sample_batch["voxel_num_points"].contiguous(),
        sample_batch["voxel_coords"].contiguous(),
    )
    with torch.no_grad():
        sample_vfe = vfe_only(*sample_inputs)
        sample_scatter = vfe_scatter(*sample_inputs)

    engines = {
        "trt_vfe_only": build_engine(vfe_only, sample_inputs, ["voxels", "voxel_num_points", "voxel_coords"], ["pillar_features"], f"vfe_only_bucket_{args.bucket_size}"),
        "trt_vfe_scatter": build_engine(vfe_scatter, sample_inputs, ["voxels", "voxel_num_points", "voxel_coords"], ["spatial_features"], f"vfe_scatter_bucket_{args.bucket_size}"),
        "trt_backbone_head_from_pytorch_scatter": build_engine(backbone_head, (sample_scatter,), ["spatial_features"], ["batch_cls_preds", "batch_box_preds"], f"backbone_head_bucket_{args.bucket_size}"),
        "trt_full_core": build_engine(full_core, sample_inputs, ["voxels", "voxel_num_points", "voxel_coords"], ["batch_cls_preds", "batch_box_preds"], f"full_core_bucket_{args.bucket_size}"),
    }

    rows = []
    judgements = []
    with torch.no_grad():
        for frame_id, batch_dict, pad_stats in prepared_rows:
            voxels = batch_dict["voxels"].contiguous()
            voxel_num_points = batch_dict["voxel_num_points"].contiguous()
            voxel_coords = batch_dict["voxel_coords"].contiguous()

            # PyTorch references
            pytorch_vfe = vfe_only(voxels, voxel_num_points, voxel_coords)
            pytorch_scatter = vfe_scatter(voxels, voxel_num_points, voxel_coords)
            pytorch_bh_cls, pytorch_bh_box = backbone_head(pytorch_scatter)
            pytorch_full_cls, pytorch_full_box = full_core(voxels, voxel_num_points, voxel_coords)

            trt_vfe = trt_forward(engines["trt_vfe_only"][0], engines["trt_vfe_only"][1], {"voxels": voxels, "voxel_num_points": voxel_num_points, "voxel_coords": voxel_coords})
            trt_scatter = trt_forward(engines["trt_vfe_scatter"][0], engines["trt_vfe_scatter"][1], {"voxels": voxels, "voxel_num_points": voxel_num_points, "voxel_coords": voxel_coords})
            trt_bh = trt_forward(engines["trt_backbone_head_from_pytorch_scatter"][0], engines["trt_backbone_head_from_pytorch_scatter"][1], {"spatial_features": pytorch_scatter.contiguous()})
            trt_full = trt_forward(engines["trt_full_core"][0], engines["trt_full_core"][1], {"voxels": voxels, "voxel_num_points": voxel_num_points, "voxel_coords": voxel_coords})

            stage_specs = [
                ("trt_vfe_only", pytorch_vfe.detach().cpu().numpy(), trt_vfe["pillar_features"]),
                ("trt_vfe_scatter", pytorch_scatter.detach().cpu().numpy(), trt_scatter["spatial_features"]),
                ("trt_backbone_head_from_pytorch_scatter_cls", pytorch_bh_cls.detach().cpu().numpy(), trt_bh["batch_cls_preds"]),
                ("trt_backbone_head_from_pytorch_scatter_box", pytorch_bh_box.detach().cpu().numpy(), trt_bh["batch_box_preds"]),
                ("trt_full_core_cls", pytorch_full_cls.detach().cpu().numpy(), trt_full["batch_cls_preds"]),
                ("trt_full_core_box", pytorch_full_box.detach().cpu().numpy(), trt_full["batch_box_preds"]),
            ]
            for stage_name, pytorch_array, trt_array in stage_specs:
                diff = summarize_diff(pytorch_array, trt_array)
                rows.append(
                    {
                        "frame_id": frame_id,
                        "padding_strategy": args.padding_strategy,
                        "bucket_size": args.bucket_size,
                        "real_pillar_count": pad_stats["real_pillar_count"],
                        "padded_pillar_count": pad_stats["padded_pillar_count"],
                        "stage": stage_name,
                        "pytorch_summary": json.dumps(summarize_tensor(pytorch_array), ensure_ascii=False),
                        "trt_summary": json.dumps(summarize_tensor(trt_array), ensure_ascii=False),
                        "max_abs_diff": diff.get("max_abs_diff"),
                        "mean_abs_diff": diff.get("mean_abs_diff"),
                        "p99_abs_diff": diff.get("p99_abs_diff"),
                        "nan_count": diff.get("nan_count"),
                        "inf_count": diff.get("inf_count"),
                        "zero_ratio": diff.get("zero_ratio"),
                    }
                )

            py_batch = dict(batch_dict)
            py_batch["batch_cls_preds"] = pytorch_full_cls
            py_batch["batch_box_preds"] = pytorch_full_box
            py_batch["cls_preds_normalized"] = False
            py_pred, _ = model.post_processing(py_batch)
            trt_bh_batch = dict(batch_dict)
            trt_bh_batch["batch_cls_preds"] = torch.from_numpy(trt_bh["batch_cls_preds"]).to(device="cuda", dtype=torch.float32)
            trt_bh_batch["batch_box_preds"] = torch.from_numpy(trt_bh["batch_box_preds"]).to(device="cuda", dtype=torch.float32)
            trt_bh_batch["cls_preds_normalized"] = False
            trt_bh_pred, _ = model.post_processing(trt_bh_batch)
            trt_full_batch = dict(batch_dict)
            trt_full_batch["batch_cls_preds"] = torch.from_numpy(trt_full["batch_cls_preds"]).to(device="cuda", dtype=torch.float32)
            trt_full_batch["batch_box_preds"] = torch.from_numpy(trt_full["batch_box_preds"]).to(device="cuda", dtype=torch.float32)
            trt_full_batch["cls_preds_normalized"] = False
            trt_full_pred, _ = model.post_processing(trt_full_batch)

            judgements.append(
                {
                    "frame_id": frame_id,
                    "real_pillar_count": pad_stats["real_pillar_count"],
                    "padded_pillar_count": pad_stats["padded_pillar_count"],
                    "pytorch_box_count": int(py_pred[0]["pred_boxes"].shape[0]),
                    "trt_backbone_head_box_count": int(trt_bh_pred[0]["pred_boxes"].shape[0]),
                    "trt_full_box_count": int(trt_full_pred[0]["pred_boxes"].shape[0]),
                }
            )

    write_csv(csv_path, rows, list(rows[0].keys()) if rows else ["frame_id"])

    def _stage_mean(prefix: str) -> float | None:
        values = [row["mean_abs_diff"] for row in rows if row["stage"] == prefix and row["mean_abs_diff"] is not None]
        return float(np.mean(values)) if values else None

    summary = {
        "status": "completed",
        "bucket_size": args.bucket_size,
        "frame_ids": frame_ids,
        "padding_strategy": args.padding_strategy,
        "rows": rows,
        "judgements": judgements,
        "stage_mean_abs_diff": {
            "trt_vfe_only": _stage_mean("trt_vfe_only"),
            "trt_vfe_scatter": _stage_mean("trt_vfe_scatter"),
            "trt_backbone_head_from_pytorch_scatter_cls": _stage_mean("trt_backbone_head_from_pytorch_scatter_cls"),
            "trt_backbone_head_from_pytorch_scatter_box": _stage_mean("trt_backbone_head_from_pytorch_scatter_box"),
            "trt_full_core_cls": _stage_mean("trt_full_core_cls"),
            "trt_full_core_box": _stage_mean("trt_full_core_box"),
        },
    }
    write_json(json_path, summary)
    write_markdown(
        md_path,
        "# TensorRT Submodule Bisection\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Bucket size: `{summary['bucket_size']}`\n"
        f"- Frames: `{summary['frame_ids']}`\n"
        f"- Padding strategy: `{summary['padding_strategy']}`\n"
        f"- Mean abs diff VFE: `{summary['stage_mean_abs_diff']['trt_vfe_only']}`\n"
        f"- Mean abs diff VFE+scatter: `{summary['stage_mean_abs_diff']['trt_vfe_scatter']}`\n"
        f"- Mean abs diff backbone/head cls: `{summary['stage_mean_abs_diff']['trt_backbone_head_from_pytorch_scatter_cls']}`\n"
        f"- Mean abs diff full core cls: `{summary['stage_mean_abs_diff']['trt_full_core_cls']}`\n",
    )
    print(json.dumps({"status": "completed", "report": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
