from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.openpcdet_adapter import build_openpcdet_env, probe_openpcdet, probe_python_runtime
from runtime.lidar_system_algorithm.report_schema import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX/TensorRT deployment precheck.")
    parser.add_argument("--openpcdet-root", default="external/OpenPCDet", help="OpenPCDet root path.")
    parser.add_argument("--cfg-file", default="external/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml", help="PointPillars config path.")
    parser.add_argument("--ckpt", default="", help="PointPillars checkpoint path.")
    parser.add_argument("--python-exe", default="python", help="OpenPCDet runtime Python executable.")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm", help="Output directory.")
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


def run_wsl_probe(
    distro: str,
    wsl_python: str,
    openpcdet_root: Path,
    cfg_file: Path,
    ckpt_path: Path,
    output_json: Path,
) -> dict:
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
    stdout = completed.stdout.decode("utf-8", errors="ignore") if isinstance(completed.stdout, bytes) else str(completed.stdout)
    stderr = completed.stderr.decode("utf-8", errors="ignore") if isinstance(completed.stderr, bytes) else str(completed.stderr)
    if completed.returncode != 0:
        return {
            "status": "failed",
            "stdout": stdout,
            "stderr": stderr,
            "blockers": ["WSL deployment probe execution failed."],
        }
    return json.loads(output_json.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    openpcdet_root = (PROJECT_ROOT / args.openpcdet_root).resolve() if not Path(args.openpcdet_root).is_absolute() else Path(args.openpcdet_root)
    cfg_file = (PROJECT_ROOT / args.cfg_file).resolve() if not Path(args.cfg_file).is_absolute() else Path(args.cfg_file)
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else PROJECT_ROOT / "models" / "pointpillar_kitti.pth"
    python_exe = Path(args.python_exe).expanduser()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe = probe_openpcdet(openpcdet_root=openpcdet_root, config_path=cfg_file, checkpoint_path=ckpt_path)
    target_runtime = (
        probe_python_runtime(python_executable=python_exe, env=build_openpcdet_env(python_executable=python_exe, extra_pythonpaths=[openpcdet_root]))
        if python_exe.exists()
        else {"executable": str(python_exe), "modules": {}, "torch": {"installed": False, "cuda_available": False, "cuda_device_count": 0, "cuda_name": None}, "probe_error": "python_executable_missing"}
    )
    custom_ops_dir = openpcdet_root / "pcdet" / "ops"
    source_blockers = []
    if custom_ops_dir.exists():
        source_blockers.append("Source inspection: OpenPCDet ships custom ops under pcdet/ops, which often complicate ONNX/TensorRT export.")
    if probe.config_summary.get("post_nms_type"):
        source_blockers.append(f"Source inspection: post-processing uses NMS type `{probe.config_summary['post_nms_type']}` and may need custom export handling.")
    if not target_runtime.get("modules", {}).get("onnx", False):
        source_blockers.append("Windows OpenPCDet runtime is missing `onnx`, so export cannot be validated there.")
    if not probe.checkpoint_exists:
        source_blockers.append("ONNX export skipped because the PointPillars checkpoint was not found.")
    if not target_runtime.get("torch", {}).get("cuda_available", False):
        source_blockers.append("TensorRT precheck detected CUDA-unavailable runtime in the dedicated OpenPCDet environment; engine build was skipped.")
    if not target_runtime.get("modules", {}).get("tensorrt", False):
        source_blockers.append("Windows OpenPCDet runtime is missing `tensorrt`.")

    wsl_probe_json = output_dir / "deployment_precheck_wsl.json"
    wsl_probe = run_wsl_probe(
        distro=args.wsl_distro,
        wsl_python=args.wsl_python,
        openpcdet_root=openpcdet_root,
        cfg_file=cfg_file,
        ckpt_path=ckpt_path,
        output_json=wsl_probe_json,
    )
    wsl_blockers = wsl_probe.get("blockers", []) if isinstance(wsl_probe, dict) else []
    deployment_attempt = wsl_probe.get("deployment_attempt", {}) if isinstance(wsl_probe, dict) else {}
    combined_reason = "; ".join(source_blockers + wsl_blockers) if (source_blockers or wsl_blockers) else "WSL core-network deployment attempt completed; see deployment_attempt for scope and limitations."

    payload = {
        "openpcdet_probe": probe.__dict__,
        "openpcdet_python_runtime": target_runtime,
        "wsl_probe": wsl_probe,
        "onnx_export": {
            "status": deployment_attempt.get("onnx_export", {}).get("status", "skipped"),
            "reason": combined_reason,
            "scope": deployment_attempt.get("scope"),
            "artifact": deployment_attempt.get("onnx_export", {}),
        },
        "tensorrt": {
            "status": deployment_attempt.get("tensorrt_build", {}).get("status", "skipped"),
            "reason": combined_reason,
            "scope": deployment_attempt.get("scope"),
            "artifact": deployment_attempt.get("tensorrt_build", {}),
            "dummy_latency": deployment_attempt.get("tensorrt_dummy_latency", {}),
        },
        "deployment_attempt": deployment_attempt,
        "source_blockers": source_blockers,
        "wsl_blockers": wsl_blockers,
    }
    json_path = output_dir / "deployment_precheck.json"
    md_path = output_dir / "deployment_precheck.md"
    write_json(json_path, payload)
    latency = payload["tensorrt"].get("dummy_latency", {})
    if isinstance(latency, dict) and latency.get("status") == "completed":
        latency_csv = output_dir / "deployment_latency.csv"
        latency_csv.write_text(
            "runtime,scope,mean_ms,p50_ms,min_ms,max_ms,runs,note\n"
            f"TensorRT,pointpillar_core_fixed_dummy,{latency.get('mean_ms')},{latency.get('p50_ms')},{latency.get('min_ms')},{latency.get('max_ms')},{latency.get('runs')},\"{latency.get('note')}\"\n",
            encoding="utf-8",
        )

    markdown = f"""# Deployment Precheck

## Environment

- OpenPCDet root exists: `{probe.root_exists}`
- PointPillars config exists: `{probe.config_exists}`
- PointPillars checkpoint exists: `{probe.checkpoint_exists}`
- `pcdet` importable in current repo Python: `{probe.pcdet_importable}`
- Windows/OpenPCDet python: `{python_exe}`
- CUDA available in Windows/OpenPCDet python: `{target_runtime.get("torch", {}).get("cuda_available")}`
- ONNX installed in Windows/OpenPCDet python: `{target_runtime.get("modules", {}).get("onnx")}`
- ONNX Runtime installed in Windows/OpenPCDet python: `{target_runtime.get("modules", {}).get("onnxruntime")}`
- TensorRT installed in Windows/OpenPCDet python: `{target_runtime.get("modules", {}).get("tensorrt")}`
- WSL distro: `{args.wsl_distro}`
- WSL python: `{args.wsl_python}`
- CUDA available in WSL python: `{wsl_probe.get("torch", {}).get("cuda_available") if isinstance(wsl_probe, dict) else None}`
- ONNX installed in WSL python: `{wsl_probe.get("modules", {}).get("onnx") if isinstance(wsl_probe, dict) else None}`
- ONNX Runtime installed in WSL python: `{wsl_probe.get("modules", {}).get("onnxruntime") if isinstance(wsl_probe, dict) else None}`
- TensorRT installed in WSL python: `{wsl_probe.get("modules", {}).get("tensorrt") if isinstance(wsl_probe, dict) else None}`
- `pcdet.models` importable in WSL: `{wsl_probe.get("import_checks", {}).get("pcdet_models", {}).get("status") if isinstance(wsl_probe, dict) else None}`

## Result

- ONNX export: `{payload["onnx_export"]["status"]}`
- TensorRT build: `{payload["tensorrt"]["status"]}`
- TensorRT dummy core latency mean ms: `{payload["tensorrt"].get("dummy_latency", {}).get("mean_ms")}`
- TensorRT FP16 requested: `{payload["tensorrt"].get("artifact", {}).get("fp16_requested")}`

## Blockers

"""
    for blocker in source_blockers:
        markdown += f"- {blocker}\n"
    for blocker in wsl_blockers:
        markdown += f"- {blocker}\n"
    if not source_blockers and not wsl_blockers:
        markdown += "- No blocker detected during source/WLS precheck.\n"
    markdown += """
## Interpretation

- Real PointPillars inference is already running in the Windows CUDA environment.
- The isolated WSL deployment environment now imports `pcdet.models`, `spconv`, and the compiled OpenPCDet CUDA ops.
- A fixed-shape PointPillars core network was exported to ONNX and built into a TensorRT engine. This is not a full end-to-end detector because voxelization and NMS are excluded, and the dummy latency must not be compared with full KITTI pipeline latency.
- FP16 was not claimed unless TensorRT reports `platform_has_fast_fp16=True`.
"""
    write_markdown(md_path, markdown)
    print(f"Saved deployment precheck: {json_path}")


if __name__ == "__main__":
    main()
