from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass
class OpenPCDetProbe:
    openpcdet_root: str
    config_path: str
    checkpoint_path: str
    root_exists: bool
    config_exists: bool
    checkpoint_exists: bool
    pcdet_importable: bool
    pcdet_error: str | None
    cuda_available: bool
    torch_version: str
    onnx_installed: bool
    onnxruntime_installed: bool
    tensorrt_installed: bool
    blockers: list[str]
    config_summary: dict

    @property
    def inference_available(self) -> bool:
        return self.root_exists and self.config_exists and self.checkpoint_exists and self.pcdet_importable


def _module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def build_openpcdet_env(python_executable: Path, extra_pythonpaths: list[Path] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    prefix = python_executable.resolve().parent
    if prefix.name.lower() == "scripts":
        prefix = prefix.parent

    path_parts = [prefix / "Library" / "bin", prefix / "Scripts", prefix]
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join([str(part) for part in path_parts if part.exists()] + ([existing_path] if existing_path else []))

    cuda_home = prefix / "Library"
    if (cuda_home / "bin" / "nvcc.exe").exists():
        env["CUDA_HOME"] = str(cuda_home)
        env["CUDA_PATH"] = str(cuda_home)

    if extra_pythonpaths:
        existing_pythonpath = env.get("PYTHONPATH", "")
        pythonpath_parts = [str(path) for path in extra_pythonpaths if path.exists()]
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        if pythonpath_parts:
            env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["OPENPCDET_NO_VIS"] = "1"
    return env


def probe_python_runtime(python_executable: Path, env: dict[str, str] | None = None) -> dict:
    probe_code = """
import importlib.util
import json
import platform
import sys

mods = ['onnx', 'onnxruntime', 'tensorrt', 'spconv', 'cv2', 'matplotlib', 'scipy']
result = {
    'version': platform.python_version(),
    'executable': sys.executable,
    'modules': {m: bool(importlib.util.find_spec(m)) for m in mods},
}
try:
    import torch
    result['torch'] = {
        'installed': True,
        'version': torch.__version__,
        'cuda_available': bool(torch.cuda.is_available()),
        'cuda_device_count': int(torch.cuda.device_count()),
        'cuda_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception as exc:
    result['torch'] = {
        'installed': False,
        'error': str(exc),
        'cuda_available': False,
        'cuda_device_count': 0,
        'cuda_name': None,
    }
print(json.dumps(result))
"""
    completed = subprocess.run(
        [str(python_executable), "-c", probe_code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        return {
            "version": None,
            "executable": str(python_executable),
            "modules": {},
            "torch": {
                "installed": False,
                "cuda_available": False,
                "cuda_device_count": 0,
                "cuda_name": None,
                "error": completed.stderr.strip() or completed.stdout.strip() or "probe_failed",
            },
            "probe_error": completed.stderr.strip() or completed.stdout.strip() or "probe_failed",
        }
    return json.loads(completed.stdout)


def load_config_summary(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data_cfg = payload.get("DATA_CONFIG", {})
    model_cfg = payload.get("MODEL", {})
    post_cfg = model_cfg.get("POST_PROCESSING", {})
    return {
        "class_names": payload.get("CLASS_NAMES", []),
        "point_cloud_range": data_cfg.get("POINT_CLOUD_RANGE"),
        "data_processor": data_cfg.get("DATA_PROCESSOR", []),
        "post_nms_type": post_cfg.get("NMS_CONFIG", {}).get("NMS_TYPE"),
        "model_name": model_cfg.get("NAME"),
    }


def probe_openpcdet(openpcdet_root: Path, config_path: Path, checkpoint_path: Path) -> OpenPCDetProbe:
    blockers: list[str] = []
    pcdet_error: str | None = None
    inserted_path = False
    if openpcdet_root.exists() and str(openpcdet_root) not in sys.path:
        sys.path.insert(0, str(openpcdet_root))
        inserted_path = True
    try:
        import pcdet  # type: ignore  # noqa: F401

        pcdet_importable = True
    except Exception as exc:
        pcdet_importable = False
        pcdet_error = repr(exc)
        blockers.append(f"pcdet import failed: {exc}")
    finally:
        if inserted_path and sys.path and sys.path[0] == str(openpcdet_root):
            sys.path.pop(0)

    if not openpcdet_root.exists():
        blockers.append(f"OpenPCDet root not found: {openpcdet_root}")
    if not config_path.exists():
        blockers.append(f"PointPillars config not found: {config_path}")
    if not checkpoint_path.exists():
        blockers.append(f"PointPillars checkpoint not found: {checkpoint_path}")
    if not torch.cuda.is_available():
        blockers.append("CUDA unavailable: current environment is CPU-only")
    if not _module_exists("onnx"):
        blockers.append("onnx package not installed")
    if not _module_exists("onnxruntime"):
        blockers.append("onnxruntime package not installed")
    if not _module_exists("tensorrt"):
        blockers.append("TensorRT Python package not installed")

    config_summary = load_config_summary(config_path)
    if config_summary.get("data_processor"):
        processor_names = [str(item.get("NAME")) for item in config_summary["data_processor"] if isinstance(item, dict)]
        if any("transform_points_to_voxels" in name.lower() for name in processor_names if name):
            blockers.append("Source inspection indicates voxelization/custom ops path may complicate ONNX export")

    return OpenPCDetProbe(
        openpcdet_root=str(openpcdet_root),
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        root_exists=openpcdet_root.exists(),
        config_exists=config_path.exists(),
        checkpoint_exists=checkpoint_path.exists(),
        pcdet_importable=pcdet_importable,
        pcdet_error=pcdet_error,
        cuda_available=torch.cuda.is_available(),
        torch_version=torch.__version__,
        onnx_installed=_module_exists("onnx"),
        onnxruntime_installed=_module_exists("onnxruntime"),
        tensorrt_installed=_module_exists("tensorrt"),
        blockers=blockers,
        config_summary=config_summary,
    )


def run_single_frame_inference(*args, **kwargs) -> dict:
    return {
        "status": "skipped",
        "reason": "Actual PointPillars inference was not executed in this environment. Use the adapter after providing a compatible checkpoint and runtime dependencies.",
        "args": args,
        "kwargs": kwargs,
    }
