from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"
WSL_DISTRO = "Ubuntu-24.04"
WSL_PYTHON = "python"


def run_wsl(command: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["wsl", "-d", WSL_DISTRO, "bash", "-lc", command],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def clean_wsl_output(text: str) -> str:
    if not text:
        return text
    return "\n".join(line for line in text.splitlines() if not line.startswith("w\x00s\x00l\x00:"))


def run_json_probe(script: str) -> dict:
    result = run_wsl(f"{WSL_PYTHON} - <<'PY'\n{script}\nPY", timeout=180)
    stdout = clean_wsl_output(result.stdout).strip()
    return json.loads(stdout or "{}")


def find_existing_paths(candidates: list[str]) -> list[str]:
    found: list[str] = []
    for candidate in candidates:
        if "*" in candidate:
            command = f"compgen -G '{candidate}' || true"
        else:
            command = f"if [ -e '{candidate}' ]; then printf '%s\\n' '{candidate}'; fi"
        result = run_wsl(command)
        for line in clean_wsl_output(result.stdout).splitlines():
            line = line.strip()
            if line and line not in found:
                found.append(line)
    return found


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    python_info = run_json_probe(
        """
import importlib.util
import json
import os
import sys
payload = {
    "python_executable": sys.executable,
    "python_version": sys.version,
    "cuda_home": os.environ.get("CUDA_HOME"),
    "cuda_path": os.environ.get("CUDA_PATH"),
    "conda_prefix": os.environ.get("CONDA_PREFIX"),
    "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    "path": os.environ.get("PATH"),
}
try:
    import torch
    payload["torch_version"] = torch.__version__
    payload["torch_cuda_available"] = bool(torch.cuda.is_available())
    payload["torch_cuda_version"] = torch.version.cuda
except Exception as exc:
    payload["torch_import_error"] = f"{type(exc).__name__}: {exc}"
payload["numba_installed"] = importlib.util.find_spec("numba") is not None
print(json.dumps(payload, ensure_ascii=False))
"""
    )

    numba_probe = run_json_probe(
        """
import json
payload = {"ran": True}
try:
    from numba import cuda
    payload["cuda_is_available"] = bool(cuda.is_available())
    try:
        arr = cuda.to_device([1.0, 2.0, 3.0, 4.0])
        payload["device_roundtrip"] = arr.copy_to_host().tolist()
    except Exception as exc:
        payload["device_roundtrip_error"] = f"{type(exc).__name__}: {exc}"
except Exception as exc:
    payload["import_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(payload, ensure_ascii=False))
"""
    )

    libdevice_candidates = find_existing_paths(
        [
            "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
            "/usr/local/cuda-*/nvvm/libdevice/libdevice.10.bc",
            "${CONDA_PREFIX:-/nonexistent}/nvvm/libdevice/libdevice.10.bc",
            "${CONDA_PREFIX:-/nonexistent}/lib/libdevice.10.bc",
            "/usr/lib/cuda/nvvm/libdevice/libdevice.10.bc",
            "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
        ]
    )
    libnvvm_candidates = find_existing_paths(
        [
            "/usr/local/cuda/nvvm/lib64/libnvvm.so",
            "/usr/local/cuda-*/nvvm/lib64/libnvvm.so",
            "${CONDA_PREFIX:-/nonexistent}/nvvm/lib64/libnvvm.so",
            "/usr/local/cuda/nvvm/lib64/libnvvm.so",
            "/usr/local/cuda/nvvm/lib64/libnvvm.so",
        ]
    )

    preferred_libdevice = libdevice_candidates[0] if libdevice_candidates else ""
    preferred_libnvvm = libnvvm_candidates[0] if libnvvm_candidates else ""
    preferred_cuda_home = str(PurePosixPath(preferred_libdevice).parents[2]) if preferred_libdevice else ""

    blocking_issues: list[str] = []
    if not libdevice_candidates:
        blocking_issues.append("libdevice.10.bc not found in probed WSL locations.")
    if not libnvvm_candidates:
        blocking_issues.append("libnvvm.so not found in probed WSL locations.")
    if numba_probe.get("device_roundtrip_error"):
        blocking_issues.append(f"Numba CUDA roundtrip failed: {numba_probe['device_roundtrip_error']}")

    audit = {
        "python_env": python_info.get("python_executable"),
        "python_version": python_info.get("python_version"),
        "torch_version": python_info.get("torch_version"),
        "torch_cuda_available": python_info.get("torch_cuda_available"),
        "torch_cuda_version": python_info.get("torch_cuda_version"),
        "cuda_home_env": python_info.get("cuda_home"),
        "cuda_path_env": python_info.get("cuda_path"),
        "conda_prefix_env": python_info.get("conda_prefix"),
        "ld_library_path_env": python_info.get("ld_library_path"),
        "path_env": python_info.get("path"),
        "libdevice_candidates": libdevice_candidates,
        "libnvvm_candidates": libnvvm_candidates,
        "preferred_cuda_home": preferred_cuda_home,
        "preferred_libdevice": preferred_libdevice,
        "preferred_libnvvm": preferred_libnvvm,
        "numba_probe": numba_probe,
        "blocking_issues": blocking_issues,
        "recommended_env": {
            "CUDA_HOME": preferred_cuda_home,
            "NUMBAPRO_NVVM": preferred_libnvvm,
            "NUMBAPRO_LIBDEVICE": preferred_libdevice,
            "XLA_FLAGS": f"--xla_gpu_cuda_data_dir={preferred_cuda_home}" if preferred_cuda_home else "",
            "NUMBA_FORCE_CUDA_CC": "8.9",
        },
        "manual_fix_suggestions": [
            "Confirm /usr/local/cuda/nvvm/libdevice/libdevice.10.bc exists when using system CUDA.",
            "If using conda/pip CUDA components, install cuda-nvcc / cuda-nvrtc / cuda-libraries-dev or cudatoolkit-dev and point NUMBAPRO_LIBDEVICE to libdevice.10.bc.",
            "Keep LD_LIBRARY_PATH including the nvvm lib64 directory that contains libnvvm.so.",
            "If PyTorch 2.6+ checkpoint loading fails inside tools/test.py, run through a trusted wrapper that forces torch.load(weights_only=False) for local checkpoints you trust.",
        ],
    }

    audit_json = REPORT_ROOT / "cuda_libdevice_audit.json"
    audit_md = REPORT_ROOT / "cuda_libdevice_audit.md"
    audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# CUDA libdevice Audit",
        "",
        f"- Python env: `{audit['python_env']}`",
        f"- Python version: `{audit['python_version']}`",
        f"- Torch version: `{audit['torch_version']}`",
        f"- torch.cuda.is_available(): `{audit['torch_cuda_available']}`",
        f"- torch.version.cuda: `{audit['torch_cuda_version']}`",
        f"- CUDA_HOME: `{audit['cuda_home_env']}`",
        f"- CUDA_PATH: `{audit['cuda_path_env']}`",
        f"- CONDA_PREFIX: `{audit['conda_prefix_env']}`",
        f"- Preferred CUDA_HOME: `{audit['preferred_cuda_home']}`",
        "",
        "## libdevice.10.bc",
        "",
    ]
    lines.extend([f"- `{item}`" for item in libdevice_candidates] or ["- Not found."])
    lines.extend(["", "## libnvvm.so", ""])
    lines.extend([f"- `{item}`" for item in libnvvm_candidates] or ["- Not found."])
    lines.extend(
        [
            "",
            "## Numba Probe",
            "",
            f"- numba installed: `{numba_probe.get('import_error') is None}`",
            f"- cuda.is_available(): `{numba_probe.get('cuda_is_available')}`",
            f"- device roundtrip: `{numba_probe.get('device_roundtrip')}`",
        ]
    )
    if numba_probe.get("device_roundtrip_error"):
        lines.append(f"- device roundtrip error: `{numba_probe['device_roundtrip_error']}`")
    lines.extend(["", "## Blocking Issues", ""])
    lines.extend([f"- {item}" for item in blocking_issues] or ["- No blocking issue detected in the probe environment."])
    lines.extend(["", "## Recommended Environment", ""])
    for key, value in audit["recommended_env"].items():
        lines.append(f"- `{key}={value}`")
    lines.extend(["", "## Manual Fix Suggestions", ""])
    lines.extend([f"- {item}" for item in audit["manual_fix_suggestions"]])
    audit_md.write_text("\n".join(lines), encoding="utf-8")
    print(str(audit_json))


if __name__ == "__main__":
    main()
