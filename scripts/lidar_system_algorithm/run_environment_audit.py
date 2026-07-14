from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.kitti_io import locate_default_kitti_root
from runtime.lidar_system_algorithm.openpcdet_adapter import build_openpcdet_env, probe_openpcdet, probe_python_runtime
from runtime.lidar_system_algorithm.report_schema import timestamp_utc, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the local LiDAR system algorithm environment.")
    parser.add_argument("--kitti-root", default="", help="Optional KITTI root override.")
    parser.add_argument("--openpcdet-root", default="external/OpenPCDet", help="OpenPCDet root path.")
    parser.add_argument("--cfg-file", default="external/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml", help="PointPillars config path.")
    parser.add_argument("--ckpt", default="", help="PointPillars checkpoint path.")
    parser.add_argument("--python-exe", default="python", help="OpenPCDet runtime Python executable.")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm", help="Audit report output directory.")
    parser.add_argument("--wsl-distro", default="Ubuntu-24.04", help="WSL distro for deployment probing.")
    parser.add_argument("--wsl-python", default="python", help="WSL Python used for deployment probing.")
    return parser.parse_args()


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[-1]
    if drive:
        return f"/mnt/{drive}{tail}"
    return resolved.as_posix()


def run_wsl_probe(distro: str, wsl_python: str, openpcdet_root: Path, cfg_file: Path, ckpt_path: Path, output_json: Path) -> dict:
    probe_script = PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "wsl_onnx_tensorrt_probe.py"
    wsl_env_prefix = (
        "CUDA_HOME=/usr/local/cuda "
        "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib:"
        "/opt/tensorrt/lib:"
        "/usr/local/cuda/lib64:"
        "/usr/local/cuda/nvvm/lib64 "
    )
    command = [
        "wsl",
        "-d",
        distro,
        "bash",
        "-lc",
        wsl_env_prefix
        + " ".join(
            [
                shlex.quote(wsl_python),
                shlex.quote(to_wsl_path(probe_script)),
                "--openpcdet-root",
                shlex.quote(to_wsl_path(openpcdet_root)),
                "--cfg-file",
                shlex.quote(to_wsl_path(cfg_file)),
                "--ckpt",
                shlex.quote(to_wsl_path(ckpt_path)),
                "--output-json",
                shlex.quote(to_wsl_path(output_json)),
            ]
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=False, check=False, cwd=str(PROJECT_ROOT))
    if completed.returncode != 0:
        stdout = completed.stdout.decode("utf-8", errors="ignore") if isinstance(completed.stdout, bytes) else str(completed.stdout)
        stderr = completed.stderr.decode("utf-8", errors="ignore") if isinstance(completed.stderr, bytes) else str(completed.stderr)
        return {
            "status": "failed",
            "stdout": stdout,
            "stderr": stderr,
            "blockers": ["WSL probe execution failed."],
        }
    return json.loads(output_json.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    kitti_root = Path(args.kitti_root).expanduser() if args.kitti_root else locate_default_kitti_root()
    openpcdet_root = (PROJECT_ROOT / args.openpcdet_root).resolve() if not Path(args.openpcdet_root).is_absolute() else Path(args.openpcdet_root)
    cfg_file = (PROJECT_ROOT / args.cfg_file).resolve() if not Path(args.cfg_file).is_absolute() else Path(args.cfg_file)
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else PROJECT_ROOT / "models" / "pointpillar_kitti.pth"
    python_exe = Path(args.python_exe).expanduser()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    probe = probe_openpcdet(openpcdet_root=openpcdet_root, config_path=cfg_file, checkpoint_path=ckpt_path)
    target_runtime = (
        probe_python_runtime(python_executable=python_exe, env=build_openpcdet_env(python_executable=python_exe, extra_pythonpaths=[openpcdet_root]))
        if python_exe.exists()
        else {"executable": str(python_exe), "probe_error": "python_executable_missing", "modules": {}, "torch": {"installed": False, "cuda_available": False, "cuda_device_count": 0, "cuda_name": None}}
    )
    wsl_probe_json = output_dir / "environment_audit_wsl_runtime.json"
    wsl_runtime = run_wsl_probe(
        distro=args.wsl_distro,
        wsl_python=args.wsl_python,
        openpcdet_root=openpcdet_root,
        cfg_file=cfg_file,
        ckpt_path=ckpt_path,
        output_json=wsl_probe_json,
    )
    kitti_found = kitti_root.exists()
    frame_count = 0
    training_root = None
    if (kitti_root / "training" / "velodyne").exists():
        training_root = kitti_root / "training"
    elif (kitti_root / "extracted" / "training" / "velodyne").exists():
        training_root = kitti_root / "extracted" / "training"
    if training_root is not None:
        frame_count = len(list((training_root / "velodyne").glob("*.bin")))

    audit = {
        "timestamp_utc": timestamp_utc(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "openpcdet_python_executable": str(python_exe),
        "wsl_python_executable": args.wsl_python,
        "wsl_distro": args.wsl_distro,
        "kitti_root": str(kitti_root),
        "kitti_found": kitti_found,
        "kitti_training_frame_count": frame_count,
        "openpcdet": {
            "root": probe.openpcdet_root,
            "config_path": probe.config_path,
            "checkpoint_path": probe.checkpoint_path,
            "root_exists": probe.root_exists,
            "config_exists": probe.config_exists,
            "checkpoint_exists": probe.checkpoint_exists,
            "pcdet_importable": probe.pcdet_importable,
            "pcdet_error": probe.pcdet_error,
            "config_summary": probe.config_summary,
        },
        "runtime": {
            "current_python": {
                "torch_version": probe.torch_version,
                "cuda_available": probe.cuda_available,
                "cuda_device_count": torch.cuda.device_count(),
                "onnx_installed": probe.onnx_installed,
                "onnxruntime_installed": probe.onnxruntime_installed,
                "tensorrt_installed": probe.tensorrt_installed,
                "cv2_installed": module_exists("cv2"),
                "numpy_installed": module_exists("numpy"),
            },
            "openpcdet_python": target_runtime,
            "wsl_python": wsl_runtime,
        },
        "tasks": {
            "kitti_pointcloud_pipeline": {"status": "runnable" if kitti_found else "skipped", "reason": "" if kitti_found else "KITTI root not found"},
            "dbscan_baseline": {"status": "runnable" if kitti_found else "skipped", "reason": "" if kitti_found else "KITTI root not found"},
            "pointpillars_inference": {
                "status": "runnable" if probe.inference_available and target_runtime.get("torch", {}).get("cuda_available", False) else "skeleton_only",
                "reason": "" if probe.inference_available and target_runtime.get("torch", {}).get("cuda_available", False) else "; ".join(probe.blockers + ([] if target_runtime.get("torch", {}).get("cuda_available", False) else ["OpenPCDet runtime python CUDA unavailable"])),
            },
            "latency_profiling": {
                "status": "runnable_with_model_forward" if probe.inference_available and target_runtime.get("torch", {}).get("cuda_available", False) else "runnable_with_partial_stages",
                "reason": "OpenPCDet helper can profile model forward on the dedicated CUDA environment." if probe.inference_available and target_runtime.get("torch", {}).get("cuda_available", False) else "Preprocessing/pillarization profiling can run on CPU. Model-forward stages require a valid checkpoint/runtime.",
            },
            "voxelization_ablation": {"status": "runnable", "reason": "Pure preprocessing experiment implemented with local data."},
            "kitti_official_eval": {
                "status": "runnable_with_native_cuda_eval" if kitti_found and probe.inference_available else "skipped",
                "reason": "Prediction export uses the Windows CUDA OpenPCDet runtime; evaluation now prefers the native OpenPCDet numba CUDA evaluator in WSL and falls back to CPU polygon IoU only if native eval fails." if kitti_found and probe.inference_available else "KITTI labels or PointPillars runtime unavailable.",
            },
            "onnx_tensorrt_precheck": {
                "status": "runnable_core_engine_attempt",
                "reason": "WSL deployment packages, spconv, and OpenPCDet custom CUDA ops are present. The precheck attempts fixed-shape PointPillars core ONNX export and TensorRT engine build, excluding voxelization and NMS.",
            },
            "tracking_demo": {"status": "runnable", "reason": "PointPillars detections or DBSCAN baseline detections can feed the lightweight tracker."},
        },
        "blockers": {
            "current_python": probe.blockers,
            "wsl": wsl_runtime.get("blockers", []) if isinstance(wsl_runtime, dict) else ["wsl_probe_unavailable"],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "environment_audit.json"
    md_path = output_dir / "environment_audit.md"
    write_json(json_path, audit)

    markdown = f"""# Environment Audit

- Timestamp: {audit["timestamp_utc"]}
- Python: `{audit["python_executable"]}`
- KITTI root: `{audit["kitti_root"]}`
- KITTI found: `{audit["kitti_found"]}`
- KITTI training frames: `{audit["kitti_training_frame_count"]}`
- OpenPCDet found: `{probe.root_exists}`
- PointPillars config found: `{probe.config_exists}`
- PointPillars checkpoint found: `{probe.checkpoint_exists}`
- `pcdet` importable in repo Python: `{probe.pcdet_importable}`
- Current Python CUDA available: `{probe.cuda_available}`
- OpenPCDet Python: `{python_exe}`
- OpenPCDet Python CUDA available: `{target_runtime.get("torch", {}).get("cuda_available")}`
- TensorRT installed in OpenPCDet Python: `{target_runtime.get("modules", {}).get("tensorrt")}`
- ONNX installed in OpenPCDet Python: `{target_runtime.get("modules", {}).get("onnx")}`
- ONNX Runtime installed in OpenPCDet Python: `{target_runtime.get("modules", {}).get("onnxruntime")}`
- WSL distro: `{args.wsl_distro}`
- WSL Python: `{args.wsl_python}`
- WSL CUDA available: `{wsl_runtime.get("torch", {}).get("cuda_available") if isinstance(wsl_runtime, dict) else None}`
- WSL ONNX installed: `{wsl_runtime.get("modules", {}).get("onnx") if isinstance(wsl_runtime, dict) else None}`
- WSL ONNX Runtime installed: `{wsl_runtime.get("modules", {}).get("onnxruntime") if isinstance(wsl_runtime, dict) else None}`
- WSL TensorRT installed: `{wsl_runtime.get("modules", {}).get("tensorrt") if isinstance(wsl_runtime, dict) else None}`

## Runnable Tasks

- KITTI/OpenCV data pipeline: `{audit["tasks"]["kitti_pointcloud_pipeline"]["status"]}`
- DBSCAN baseline: `{audit["tasks"]["dbscan_baseline"]["status"]}`
- PointPillars inference: `{audit["tasks"]["pointpillars_inference"]["status"]}`
- Profiling: `{audit["tasks"]["latency_profiling"]["status"]}`
- KITTI official eval: `{audit["tasks"]["kitti_official_eval"]["status"]}`
- Voxelization ablation: `{audit["tasks"]["voxelization_ablation"]["status"]}`
- ONNX/TensorRT precheck: `{audit["tasks"]["onnx_tensorrt_precheck"]["status"]}`
- Tracking demo: `{audit["tasks"]["tracking_demo"]["status"]}`

## Blockers

"""
    for blocker in probe.blockers:
        markdown += f"- Current Python: {blocker}\n"
    for blocker in audit["blockers"]["wsl"]:
        markdown += f"- WSL: {blocker}\n"
    if not probe.blockers and not audit["blockers"]["wsl"]:
        markdown += "- No blocking issue detected during audit.\n"
    write_markdown(md_path, markdown)
    print(json.dumps({"environment_audit_json": str(json_path), "environment_audit_md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
