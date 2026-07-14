from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WSL deployment probe for OpenPCDet/ONNX/TensorRT.")
    parser.add_argument("--openpcdet-root", required=True, help="OpenPCDet root path on Linux/WSL.")
    parser.add_argument("--cfg-file", required=True, help="PointPillars config path on Linux/WSL.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path on Linux/WSL.")
    parser.add_argument("--output-json", required=True, help="Output JSON path on Linux/WSL.")
    return parser.parse_args()


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def try_exec(snippet: str) -> dict:
    try:
        scope: dict = {}
        exec(snippet, scope)
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "error": repr(exc)}


def attempt_pointpillar_core_deployment(openpcdet_root: Path, cfg_file: Path, ckpt_path: Path, output_dir: Path) -> dict:
    """Export the tensor core of PointPillars, excluding voxelization and NMS."""
    result: dict = {
        "status": "not_started",
        "scope": "PointPillars core network only: VFE + scatter + BEV backbone + dense head; voxelization and NMS are excluded.",
        "onnx_export": {"status": "not_started"},
        "tensorrt_build": {"status": "not_started"},
        "tensorrt_dummy_latency": {"status": "not_started"},
    }
    try:
        import torch
        import tensorrt as trt
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network
        from pcdet.utils import common_utils

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

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=RuntimeDataset())
        model.load_params_from_file(filename=str(ckpt_path), logger=common_utils.create_logger(), to_cpu=True)
        model.cuda().eval()

        class PointPillarCore(torch.nn.Module):
            def __init__(self, pointpillar):
                super().__init__()
                self.modules_for_export = torch.nn.ModuleList(list(pointpillar.module_list))

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

        core = PointPillarCore(model).cuda().eval()
        voxels = torch.randn(64, 32, 4, device="cuda", dtype=torch.float32)
        voxel_num_points = torch.randint(1, 32, (64,), device="cuda", dtype=torch.int32)
        voxel_coords = torch.zeros((64, 4), device="cuda", dtype=torch.int32)
        voxel_coords[:, 2] = torch.arange(64, device="cuda", dtype=torch.int32) % 496
        voxel_coords[:, 3] = torch.arange(64, device="cuda", dtype=torch.int32) % 432
        with torch.no_grad():
            outputs = core(voxels, voxel_num_points, voxel_coords)

        output_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = output_dir / "pointpillar_core_attempt.onnx"
        engine_path = output_dir / "pointpillar_core_attempt.engine"
        torch.onnx.export(
            core,
            (voxels, voxel_num_points, voxel_coords),
            str(onnx_path),
            opset_version=17,
            dynamo=False,
            input_names=["voxels", "voxel_num_points", "voxel_coords"],
            output_names=["batch_cls_preds", "batch_box_preds"],
        )
        result["onnx_export"] = {
            "status": "completed",
            "path": str(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
            "output_shapes": [list(tensor.shape) for tensor in outputs],
            "warnings": [
                "Legacy TorchScript exporter was used because torch.export/dynamo export fails on PointPillarScatter data-dependent Python control flow.",
                "The exported graph is fixed to the dummy voxel shape [64, 32, 4] and is not a full dynamic end-to-end detector.",
            ],
        }

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        try:
            network = builder.create_network(0)
        except TypeError:
            network = builder.create_network()
        parser = trt.OnnxParser(network, logger)
        parse_ok = parser.parse(onnx_path.read_bytes())
        parser_errors = [parser.get_error(index).desc() for index in range(parser.num_errors)]
        result["tensorrt_build"] = {
            "status": "parse_failed",
            "tensorrt_version": trt.__version__,
            "parse_ok": bool(parse_ok),
            "parser_errors": parser_errors,
        }
        if not parse_ok:
            result["status"] = "partial"
            return result

        config = builder.create_builder_config()
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        elif hasattr(config, "max_workspace_size"):
            config.max_workspace_size = 2 << 30
        platform_has_fast_fp16 = bool(getattr(builder, "platform_has_fast_fp16", False))
        if platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        start = time.perf_counter()
        if hasattr(builder, "build_serialized_network"):
            serialized = builder.build_serialized_network(network, config)
        else:
            engine = builder.build_engine(network, config)
            serialized = engine.serialize() if engine else None
        build_ms = (time.perf_counter() - start) * 1000.0
        if serialized is None:
            result["tensorrt_build"].update({"status": "build_failed", "build_ms": build_ms})
            result["status"] = "partial"
            return result
        engine_path.write_bytes(bytes(serialized))
        result["tensorrt_build"].update(
            {
                "status": "completed",
                "engine_path": str(engine_path),
                "engine_size_bytes": engine_path.stat().st_size,
                "build_ms": build_ms,
                "platform_has_fast_fp16": platform_has_fast_fp16,
                "fp16_requested": platform_has_fast_fp16,
            }
        )

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        context = engine.create_execution_context()
        names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        buffers = {
            "voxels": voxels,
            "voxel_num_points": voxel_num_points,
            "voxel_coords": voxel_coords,
            "batch_cls_preds": torch.empty((1, 321408, 3), device="cuda", dtype=torch.float32),
            "batch_box_preds": torch.empty((1, 321408, 7), device="cuda", dtype=torch.float32),
        }
        for name, tensor in buffers.items():
            if name in names:
                context.set_tensor_address(name, int(tensor.data_ptr()))
        stream = torch.cuda.Stream()
        for _ in range(5):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        timings = []
        for _ in range(30):
            start = time.perf_counter()
            ok = context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned false")
            timings.append((time.perf_counter() - start) * 1000.0)
        timings_sorted = sorted(timings)
        result["tensorrt_dummy_latency"] = {
            "status": "completed",
            "runs": len(timings),
            "mean_ms": sum(timings) / len(timings),
            "p50_ms": timings_sorted[len(timings_sorted) // 2],
            "min_ms": min(timings),
            "max_ms": max(timings),
            "note": "Dummy fixed-shape core-network latency only; not end-to-end KITTI latency and not comparable to full PyTorch pipeline.",
        }
        result["status"] = "completed"
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def main() -> None:
    args = parse_args()
    openpcdet_root = Path(args.openpcdet_root).resolve()
    cfg_file = Path(args.cfg_file).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    output_json = Path(args.output_json).resolve()

    sys.path.insert(0, str(openpcdet_root))

    modules = {name: module_exists(name) for name in ["onnx", "onnxruntime", "tensorrt", "SharedArray", "spconv", "pcdet"]}
    torch_info = {"installed": False, "cuda_available": False, "cuda_name": None, "version": None}
    tensorrt_info = {"installed": False, "version": None}
    if modules["tensorrt"]:
        import tensorrt as trt

        tensorrt_info = {"installed": True, "version": getattr(trt, "__version__", None)}
    if module_exists("torch"):
        import torch

        torch_info = {
            "installed": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "version": torch.__version__,
        }

    import_checks = {
        "pcdet_config": try_exec("from pcdet.config import cfg, cfg_from_yaml_file"),
        "dataset_template": try_exec("from pcdet.datasets import DatasetTemplate"),
        "pcdet_models": try_exec("from pcdet.models import build_network, load_data_to_gpu"),
        "iou3d_nms_op": try_exec("from pcdet.ops.iou3d_nms import iou3d_nms_cuda"),
        "roiaware_pool3d_op": try_exec("from pcdet.ops.roiaware_pool3d import roiaware_pool3d_cuda"),
    }

    blockers = []
    if not modules["spconv"]:
        blockers.append("WSL runtime is missing `spconv`, so Linux-side OpenPCDet model construction is incomplete.")
    if import_checks["pcdet_models"]["status"] != "ok":
        blockers.append(f"Importing `pcdet.models` failed: {import_checks['pcdet_models'].get('error')}")
    if import_checks["iou3d_nms_op"]["status"] != "ok":
        blockers.append(f"Custom op `iou3d_nms_cuda` is unavailable: {import_checks['iou3d_nms_op'].get('error')}")
    if import_checks["roiaware_pool3d_op"]["status"] != "ok":
        blockers.append(f"Custom op `roiaware_pool3d_cuda` is unavailable: {import_checks['roiaware_pool3d_op'].get('error')}")
    if not modules["onnx"]:
        blockers.append("WSL runtime is missing `onnx`.")
    if not modules["onnxruntime"]:
        blockers.append("WSL runtime is missing `onnxruntime`.")
    if not modules["tensorrt"]:
        blockers.append("WSL runtime is missing `tensorrt`.")

    deployment_attempt = (
        attempt_pointpillar_core_deployment(openpcdet_root, cfg_file, ckpt_path, output_json.parent)
        if not blockers and torch_info["cuda_available"] and ckpt_path.exists()
        else {"status": "skipped", "reason": "Skipped because WSL import/runtime blockers remain or checkpoint is missing."}
    )

    payload = {
        "status": "completed",
        "platform": "wsl",
        "python_executable": sys.executable,
        "openpcdet_root": str(openpcdet_root),
        "cfg_file": str(cfg_file),
        "checkpoint_path": str(ckpt_path),
        "paths": {
            "openpcdet_root_exists": openpcdet_root.exists(),
            "cfg_exists": cfg_file.exists(),
            "checkpoint_exists": ckpt_path.exists(),
        },
        "modules": modules,
        "torch": torch_info,
        "tensorrt": tensorrt_info,
        "trtexec_found": shutil.which("trtexec") is not None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "import_checks": import_checks,
        "blockers": blockers,
        "deployment_attempt": deployment_attempt,
        "onnx_export": {
            "status": deployment_attempt.get("onnx_export", {}).get("status", "skipped") if not blockers else "skipped",
            "reason": " ; ".join(blockers) if blockers else deployment_attempt.get("onnx_export", {}).get("warnings", ["No immediate blocker detected during WSL precheck."]),
        },
        "tensorrt_build": {
            "status": deployment_attempt.get("tensorrt_build", {}).get("status", "skipped") if not blockers else "skipped",
            "reason": " ; ".join(blockers) if blockers else deployment_attempt.get("tensorrt_build", {}),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "blocker_count": len(blockers)}, indent=2))


if __name__ == "__main__":
    main()
