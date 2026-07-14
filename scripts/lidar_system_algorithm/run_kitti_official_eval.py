from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.kitti_io import locate_default_kitti_root
from runtime.lidar_system_algorithm.openpcdet_adapter import build_openpcdet_env, probe_openpcdet, probe_python_runtime
from runtime.lidar_system_algorithm.report_schema import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KITTI official eval using the dedicated OpenPCDet runtime.")
    parser.add_argument("--kitti-root", default="", help="KITTI root directory.")
    parser.add_argument("--openpcdet-root", default="external/OpenPCDet", help="OpenPCDet root path.")
    parser.add_argument("--cfg-file", default="external/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml", help="PointPillars config path.")
    parser.add_argument("--ckpt", default="checkpoints/pointpillar_kitti.pth", help="PointPillars checkpoint path.")
    parser.add_argument("--python-exe", default="python", help="Dedicated OpenPCDet runtime Python executable.")
    parser.add_argument("--split-file", default="external/OpenPCDet/data/kitti/ImageSets/val.txt", help="KITTI split file.")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm", help="Output directory.")
    parser.add_argument("--pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt", help="KITTI prediction txt output directory.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap for smoke tests. 0 means full split.")
    parser.add_argument("--wsl-distro", default="Ubuntu-24.04", help="WSL distro used for official KITTI evaluator.")
    parser.add_argument("--wsl-python", default="python", help="Python path inside WSL used for official KITTI evaluator.")
    return parser.parse_args()


def to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[-1]
    if drive:
        return f"/mnt/{drive}{tail}"
    return resolved.as_posix()


def call_wsl_evaluator(
    distro: str,
    wsl_python: str,
    openpcdet_root: Path,
    label_dir: Path,
    pred_dir: Path,
    split_file: Path,
    output_json: Path,
) -> dict:
    runner = PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "wsl_kitti_eval_runner.py"
    wsl_env_prefix = (
        "CUDA_HOME=/usr/local/cuda "
        "NUMBA_FORCE_CUDA_CC=8.9 "
        "LD_LIBRARY_PATH=/usr/local/cuda/nvvm/lib64:"
        "/usr/local/cuda/lib64:/usr/lib/wsl/lib "
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
                shlex.quote(to_wsl_path(runner)),
                "--openpcdet-root",
                shlex.quote(to_wsl_path(openpcdet_root)),
                "--label-dir",
                shlex.quote(to_wsl_path(label_dir)),
                "--pred-dir",
                shlex.quote(to_wsl_path(pred_dir)),
                "--split-file",
                shlex.quote(to_wsl_path(split_file)),
                "--output-json",
                shlex.quote(to_wsl_path(output_json)),
            ]
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=False, check=False, cwd=str(PROJECT_ROOT))
    stdout = completed.stdout
    stderr = completed.stderr
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="ignore")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="ignore")
    if completed.returncode != 0:
        raise RuntimeError(f"WSL KITTI evaluator failed.\nstdout:\n{stdout}\nstderr:\n{stderr}")
    return json.loads(output_json.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    kitti_root = (
        ((PROJECT_ROOT / args.kitti_root).resolve() if not Path(args.kitti_root).is_absolute() else Path(args.kitti_root).expanduser())
        if args.kitti_root
        else locate_default_kitti_root().resolve()
    )
    openpcdet_root = (PROJECT_ROOT / args.openpcdet_root).resolve() if not Path(args.openpcdet_root).is_absolute() else Path(args.openpcdet_root)
    cfg_file = (PROJECT_ROOT / args.cfg_file).resolve() if not Path(args.cfg_file).is_absolute() else Path(args.cfg_file)
    split_file = (PROJECT_ROOT / args.split_file).resolve() if not Path(args.split_file).is_absolute() else Path(args.split_file)
    ckpt_path = Path(args.ckpt).expanduser()
    python_exe = Path(args.python_exe).expanduser()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    pred_dir = (PROJECT_ROOT / args.pred_dir).resolve() if not Path(args.pred_dir).is_absolute() else Path(args.pred_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    probe = probe_openpcdet(openpcdet_root=openpcdet_root, config_path=cfg_file, checkpoint_path=ckpt_path)
    runtime_probe = (
        probe_python_runtime(python_executable=python_exe, env=build_openpcdet_env(python_executable=python_exe, extra_pythonpaths=[openpcdet_root]))
        if python_exe.exists()
        else {"executable": str(python_exe), "probe_error": "python_executable_missing", "modules": {}, "torch": {"cuda_available": False}}
    )
    if not probe.inference_available or not runtime_probe.get("torch", {}).get("cuda_available", False):
        reason = "OpenPCDet root/checkpoint/runtime unavailable for official eval."
        payload = {
            "status": "skipped",
            "reason": reason,
            "probe": probe.__dict__,
            "runtime_probe": runtime_probe,
        }
        write_json(output_dir / "kitti_official_eval.json", payload)
        write_markdown(output_dir / "kitti_official_eval.md", f"# KITTI Official Eval\n\n- Status: `skipped`\n- Reason: `{reason}`\n")
        raise SystemExit(reason)

    helper_script = PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "openpcdet_kitti_eval_helper.py"
    helper_output = output_dir / "kitti_official_eval_raw.json"
    wsl_output = output_dir / "kitti_official_eval_wsl.json"
    windows_compat_dir = PROJECT_ROOT / "day09_openpcdet" / "windows_compat"
    env = build_openpcdet_env(python_executable=python_exe, extra_pythonpaths=[windows_compat_dir, openpcdet_root])
    command = [
        str(python_exe),
        str(helper_script),
        "--openpcdet-root",
        str(openpcdet_root),
        "--cfg-file",
        str(cfg_file),
        "--ckpt",
        str(ckpt_path),
        "--kitti-root",
        str(kitti_root),
        "--split-file",
        str(split_file),
        "--pred-dir",
        str(pred_dir),
        "--output-json",
        str(helper_output),
        "--max-frames",
        str(args.max_frames),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env, cwd=str(PROJECT_ROOT))
    if completed.returncode != 0:
        error_payload = {
            "status": "failed",
            "reason": "OpenPCDet helper failed during KITTI official eval.",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "probe": probe.__dict__,
            "runtime_probe": runtime_probe,
        }
        write_json(output_dir / "kitti_official_eval.json", error_payload)
        write_markdown(
            output_dir / "kitti_official_eval.md",
            "# KITTI Official Eval\n\n- Status: `failed`\n- Reason: `OpenPCDet helper failed during KITTI official eval.`\n",
        )
        raise RuntimeError(f"KITTI official eval failed.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")

    helper_payload = json.loads(helper_output.read_text(encoding="utf-8"))
    wsl_eval = call_wsl_evaluator(
        distro=args.wsl_distro,
        wsl_python=args.wsl_python,
        openpcdet_root=openpcdet_root,
        label_dir=Path(helper_payload["label_dir"]),
        pred_dir=pred_dir,
        split_file=split_file,
        output_json=wsl_output,
    )
    result_payload = {
        "status": helper_payload.get("status", "completed"),
        "probe": probe.__dict__,
        "runtime_probe": runtime_probe,
        "frame_count": helper_payload.get("frame_count"),
        "prediction_dir": helper_payload.get("prediction_dir"),
        "mean_inference_ms": helper_payload.get("mean_inference_ms"),
        "p50_inference_ms": helper_payload.get("p50_inference_ms"),
        "p95_inference_ms": helper_payload.get("p95_inference_ms"),
        "total_elapsed_ms": helper_payload.get("total_elapsed_ms"),
        "total_detections": helper_payload.get("total_detections"),
        "official_result_text": wsl_eval.get("official_result_text", ""),
        "official_result_dict": wsl_eval.get("official_result_dict", {}),
        "per_frame_preview": helper_payload.get("per_frame", [])[:10],
        "raw_output_json": str(helper_output),
        "wsl_eval_json": str(wsl_output),
        "evaluator_runtime": {
            "mode": wsl_eval.get("evaluation_backend", "wsl_cpu_eval"),
            "distro": args.wsl_distro,
            "python": args.wsl_python,
        },
    }
    write_json(output_dir / "kitti_official_eval.json", result_payload)

    markdown = f"""# KITTI Official Eval

## Status

- Status: `{result_payload["status"]}`
- Frame count: `{result_payload["frame_count"]}`
- Prediction txt dir: `{result_payload["prediction_dir"]}`
- Mean inference ms: `{result_payload["mean_inference_ms"]}`
- P50 inference ms: `{result_payload["p50_inference_ms"]}`
- P95 inference ms: `{result_payload["p95_inference_ms"]}`
- Total detections: `{result_payload["total_detections"]}`

## Official Result

```text
{result_payload["official_result_text"].strip()}
```

## Notes

- Prediction export uses the local PointPillars checkpoint together with OpenPCDet's KITTI-format prediction export path.
- The evaluation stage reuses OpenPCDet's KITTI evaluation logic in WSL. It prefers the native numba CUDA rotate-IoU evaluator and records a CPU polygon fallback only if native eval fails.
- The reported AP numbers come directly from this local evaluation run and are not training claims or leaderboard claims.
"""
    write_markdown(output_dir / "kitti_official_eval.md", markdown)
    print(f"Saved KITTI official eval: {output_dir / 'kitti_official_eval.json'}")


if __name__ == "__main__":
    main()
