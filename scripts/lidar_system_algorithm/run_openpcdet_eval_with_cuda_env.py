from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"
SHELL_WRAPPER = PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "run_openpcdet_eval_with_cuda_env.sh"
WSL_DISTRO = "Ubuntu-24.04"
PRETRAINED_CKPT = Path(r"checkpoints/pointpillar_kitti.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenPCDet tools/test.py through a shell-level CUDA/NVVM environment wrapper."
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def to_wsl_path(path: Path | str) -> str:
    value = str(path)
    if len(value) > 2 and value[1:3] == ":\\":
        return f"/mnt/{value[0].lower()}{value[2:].replace('\\', '/')}"
    return value.replace("\\", "/")


def run_wsl_shell(args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    quoted = " ".join("'" + item.replace("'", "'\"'\"'") + "'" for item in args)
    cmd = f"bash {quoted}"
    return subprocess.run(
        ["wsl", "-d", WSL_DISTRO, "bash", "-lc", cmd],
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
        return ""
    return "\n".join(line for line in text.splitlines() if not line.startswith("w\x00s\x00l\x00:"))


def discover_experiments() -> list[dict[str, Any]]:
    latest_by_prefix: dict[str, Path] = {}
    for experiment_dir in sorted(REPORT_ROOT.iterdir()):
        if not experiment_dir.is_dir():
            continue
        for prefix in ("smoke_train_", "subset_finetune_"):
            if experiment_dir.name.startswith(prefix):
                latest_by_prefix[prefix] = experiment_dir

    candidates: list[dict[str, Any]] = []
    for experiment_dir in latest_by_prefix.values():
        status_path = experiment_dir / "train_status.json"
        cfg_file = experiment_dir / "run_config.yaml"
        if not status_path.exists() or not cfg_file.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        trained_checkpoint = Path(status.get("trained_checkpoint", ""))
        batch_size = int(status.get("batch_size", 2))
        candidates.append(
            {
                "experiment_name": experiment_dir.name,
                "cfg_file": cfg_file,
                "baseline_ckpt": PRETRAINED_CKPT,
                "trained_ckpt": trained_checkpoint if trained_checkpoint.exists() else None,
                "batch_size": batch_size,
                "status_path": status_path,
                "metrics_summary_path": experiment_dir / "metrics_summary.json",
            }
        )
    return candidates


def classify_status(stdout: str, stderr: str, returncode: int) -> dict[str, Any]:
    combined = f"{stdout}\n{stderr}"
    sanity_ok = "SANITY_CHECK_PASSED" in combined
    ctypes_ok = "ctypes loaded libnvvm ok" in combined
    ldd_missing = "not found" in extract_section(combined, "=== ldd libnvvm ===")
    tools_ok = "TOOLS_TEST_COMPLETED" in combined and returncode == 0

    if tools_ok:
        status = "completed"
        tools_status = "completed"
        reason = ""
    elif not ctypes_ok:
        status = "failed"
        tools_status = "failed_due_libnvvm_loader"
        reason = "ctypes failed to load libnvvm before tools/test.py"
    elif "libNVVM cannot be found" in combined:
        status = "failed"
        tools_status = "failed_due_missing_libnvvm"
        reason = "Numba still cannot resolve libNVVM from shell-level environment"
    elif "Missing libdevice file" in combined:
        status = "failed"
        tools_status = "failed_due_missing_libdevice"
        reason = "Numba found NVVM but cannot locate libdevice"
    elif sanity_ok:
        status = "failed"
        tools_status = "failed_tools_eval_runtime"
        reason = f"tools/test.py failed after shell sanity check, returncode={returncode}"
    else:
        status = "failed"
        tools_status = f"failed_shell_wrapper_exit_{returncode}"
        reason = f"shell wrapper failed before completing sanity check, returncode={returncode}"

    return {
        "status": status,
        "sanity_check_status": "completed" if sanity_ok else "failed",
        "ctypes_load_libnvvm": ctypes_ok,
        "ldd_has_missing_dependency": ldd_missing,
        "numba_cuda_sanity_line": find_line(combined, "numba cuda available:") or find_line(
            combined, "numba cuda check failed:"
        ),
        "tools_eval_status": tools_status,
        "reason": reason,
    }


def find_line(text: str, needle: str) -> str:
    for line in text.splitlines():
        if needle in line:
            return line.strip()
    return ""


def extract_section(text: str, header: str) -> str:
    lines = text.splitlines()
    try:
        start = lines.index(header) + 1
    except ValueError:
        return ""
    end = len(lines)
    for idx in range(start, len(lines)):
        if lines[idx].startswith("==="):
            end = idx
            break
    return "\n".join(lines[start:end])


def run_attempt(experiment: dict[str, Any], attempt_name: str, ckpt: Path, eval_tag: str) -> dict[str, Any]:
    args = [
        to_wsl_path(SHELL_WRAPPER),
        "--cfg-file",
        to_wsl_path(experiment["cfg_file"]),
        "--ckpt",
        to_wsl_path(ckpt),
        "--extra-tag",
        str(experiment["experiment_name"]),
        "--eval-tag",
        eval_tag,
        "--batch-size",
        str(experiment["batch_size"]),
    ]
    started_at = datetime.now().isoformat(timespec="seconds")
    result = run_wsl_shell(args)
    ended_at = datetime.now().isoformat(timespec="seconds")
    stdout = clean_wsl_output(result.stdout)
    stderr = clean_wsl_output(result.stderr)
    classified = classify_status(stdout, stderr, result.returncode)
    return {
        "experiment_name": experiment["experiment_name"],
        "attempt_name": attempt_name,
        "cfg_file": str(experiment["cfg_file"]),
        "ckpt": str(ckpt),
        "eval_tag": eval_tag,
        "command": ["wsl", "-d", WSL_DISTRO, "bash", "-lc", "bash " + " ".join(args)],
        "started_at": started_at,
        "ended_at": ended_at,
        "returncode": result.returncode,
        "shell_wrapper": str(SHELL_WRAPPER),
        "used_shell_level_cuda_env": True,
        "used_torch_load_weights_only_false_wrapper": True,
        **classified,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "full_stdout": stdout,
        "full_stderr": stderr,
    }


def summarize_tools_status(records: list[dict[str, Any]]) -> str:
    if any(record["tools_eval_status"] == "completed" for record in records):
        return "completed"
    statuses = {record["tools_eval_status"] for record in records}
    if "failed_due_missing_libnvvm" in statuses:
        return "failed_due_missing_libnvvm"
    if "failed_due_libnvvm_loader" in statuses:
        return "failed_due_libnvvm_loader"
    if "failed_due_missing_libdevice" in statuses:
        return "failed_due_missing_libdevice"
    if "failed_tools_eval_runtime" in statuses:
        return "failed_tools_eval_runtime"
    return sorted(statuses)[0] if statuses else "not_run"


def update_experiment_status(status_path: Path, metrics_path: Path, retry_records: list[dict[str, Any]]) -> None:
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else dict(status_payload)
    tools_status = summarize_tools_status(retry_records)
    success = tools_status == "completed"
    experiment_status = {
        "training_status": "completed",
        "checkpoint_status": "completed" if status_payload.get("trained_checkpoint") else "not_available",
        "external_eval_wrapper_status": "completed",
        "opencpdet_tools_eval_status": tools_status,
        "deployment_diagnostics_status": "completed"
        if metrics_payload.get("training_mode") == "subset_finetune"
        else "not_run",
        "overall_claim_level": "subset_training_eval_completed_with_native_tools_eval"
        if success
        else "subset_training_eval_completed_with_tools_eval_issue",
    }
    top_level_status = "completed" if success else "completed_with_tools_eval_issue"
    for payload in (status_payload, metrics_payload):
        payload["status"] = top_level_status
        payload.pop("partial_reason", None)
        payload["experiment_status"] = experiment_status
        payload["opencpdet_tools_eval_shell_retry"] = retry_records
    status_path.write_text(json.dumps(status_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def update_cuda_audit_md(shell_status: dict[str, Any]) -> None:
    audit_md = REPORT_ROOT / "cuda_libdevice_audit.md"
    old = audit_md.read_text(encoding="utf-8") if audit_md.exists() else "# CUDA libdevice / NVVM Audit\n"
    marker = "\n## Shell-level OpenPCDet tools/test.py Retry\n"
    old = old.split(marker)[0].rstrip()
    lines = [
        old,
        marker.strip(),
        "",
        f"- Shell wrapper: `{SHELL_WRAPPER}`",
        f"- Shell retry log: `{REPORT_ROOT / 'openpcdet_tools_eval_shell_retry.log'}`",
        f"- Shell retry status: `{REPORT_ROOT / 'openpcdet_tools_eval_shell_retry_status.json'}`",
        f"- Overall shell retry status: `{shell_status.get('status')}`",
        f"- `ctypes.CDLL(libnvvm)`: `{shell_status.get('ctypes_load_libnvvm_any')}`",
        f"- `ldd` missing dependency: `{shell_status.get('ldd_has_missing_dependency_any')}`",
        f"- tools/test.py status: `{shell_status.get('tools_eval_status')}`",
        "",
        "If tools/test.py still fails after shell-level exports, keep using the external official eval wrapper as the measured AP source and treat native tools/test.py as a WSL CUDA/Numba environment issue, not a training failure.",
    ]
    audit_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_summary_report(per_experiment: dict[str, list[dict[str, Any]]]) -> None:
    report_json = REPORT_ROOT / "pointpillars_training_finetune_report.json"
    if report_json.exists():
        payload = json.loads(report_json.read_text(encoding="utf-8"))
        experiments = payload.get("experiments", {})
        for key in ("smoke_train", "subset_finetune"):
            experiment = experiments.get(key)
            if not isinstance(experiment, dict):
                continue
            experiment_dir = str(experiment.get("experiment_dir", "")).replace("\\", "/").rstrip("/").split("/")[-1]
            records = per_experiment.get(experiment_dir, [])
            tools_status = summarize_tools_status(records)
            success = tools_status == "completed"
            experiment["status"] = "completed" if success else "completed_with_tools_eval_issue"
            experiment["experiment_status"] = {
                "training_status": "completed",
                "checkpoint_status": "completed",
                "external_eval_wrapper_status": "completed",
                "opencpdet_tools_eval_status": tools_status,
                "deployment_diagnostics_status": "completed" if key == "subset_finetune" else "not_run",
                "overall_claim_level": "subset_training_eval_completed_with_native_tools_eval"
                if success
                else "subset_training_eval_completed_with_tools_eval_issue",
            }
        report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report_md = REPORT_ROOT / "pointpillars_training_finetune_report.md"
    if report_md.exists():
        text = report_md.read_text(encoding="utf-8")
        marker = "\n## Native tools/test.py shell retry\n"
        text = text.split(marker)[0].rstrip()
        text += (
            marker
            + "\n"
            + f"- Shell wrapper: `{SHELL_WRAPPER}`\n"
            + f"- Shell retry status: `{REPORT_ROOT / 'openpcdet_tools_eval_shell_retry_status.json'}`\n"
            + "- Training logs, checkpoints and external eval wrapper outputs were not regenerated or deleted.\n"
            + "- `tools/test.py` is only marked completed when the shell retry log contains `TOOLS_TEST_COMPLETED`.\n"
        )
        report_md.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parse_args()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    per_experiment: dict[str, list[dict[str, Any]]] = {}
    log_chunks: list[str] = []

    for experiment in discover_experiments():
        attempts: list[tuple[str, Path, str]] = [
            ("baseline_pretrained", Path(experiment["baseline_ckpt"]), "baseline_subset_eval_shell_retry")
        ]
        trained_ckpt = experiment.get("trained_ckpt")
        if isinstance(trained_ckpt, Path) and trained_ckpt.exists():
            attempts.append(("trained_checkpoint", trained_ckpt, "final_subset_eval_shell_retry"))

        records: list[dict[str, Any]] = []
        for attempt_name, ckpt, eval_tag in attempts:
            record = run_attempt(experiment, attempt_name, ckpt, eval_tag)
            records.append({k: v for k, v in record.items() if k not in {"full_stdout", "full_stderr"}})
            all_records.append(records[-1])
            log_chunks.append(f"## {experiment['experiment_name']} / {attempt_name}\n")
            log_chunks.append(record["full_stdout"] or "<no stdout>")
            log_chunks.append("\n--- STDERR ---\n")
            log_chunks.append(record["full_stderr"] or "<no stderr>")
            log_chunks.append("\n\n")

        per_experiment[str(experiment["experiment_name"])] = records
        update_experiment_status(Path(experiment["status_path"]), Path(experiment["metrics_summary_path"]), records)

    tools_status = summarize_tools_status(all_records)
    status_payload = {
        "status": "completed" if tools_status == "completed" else "partial",
        "started_by": "run_openpcdet_eval_with_cuda_env.py",
        "shell_wrapper": str(SHELL_WRAPPER),
        "log_path": str(REPORT_ROOT / "openpcdet_tools_eval_shell_retry.log"),
        "attempt_count": len(all_records),
        "sanity_check_status": "completed"
        if any(record["sanity_check_status"] == "completed" for record in all_records)
        else "failed",
        "ctypes_load_libnvvm": any(record["ctypes_load_libnvvm"] for record in all_records),
        "ctypes_load_libnvvm_any": any(record["ctypes_load_libnvvm"] for record in all_records),
        "ldd_has_missing_dependency": any(record["ldd_has_missing_dependency"] for record in all_records),
        "ldd_has_missing_dependency_any": any(record["ldd_has_missing_dependency"] for record in all_records),
        "numba_cuda_sanity": [record.get("numba_cuda_sanity_line", "") for record in all_records],
        "tools_eval_status": tools_status,
        "reason": "" if tools_status == "completed" else "; ".join(
            sorted({record["reason"] for record in all_records if record.get("reason")})
        ),
        "attempts": all_records,
        "manual_fix_suggestions": [
            "Install a full CUDA developer toolkit inside WSL so libnvvm.so is visible from a standard loader path such as /usr/local/cuda/nvvm/lib64.",
            "Alternatively create a persistent loader-visible path or symlink for libnvvm.so, then run ldconfig if using a system library path.",
            "Keep NUMBAPRO_NVVM pointing to the libnvvm.so file and NUMBAPRO_LIBDEVICE pointing to the directory/file containing libdevice.10.bc.",
            "Do not treat the external eval wrapper as native tools/test.py; keep its AP results separate unless this shell retry completes.",
        ],
    }
    (REPORT_ROOT / "openpcdet_tools_eval_shell_retry_status.json").write_text(
        json.dumps(status_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (REPORT_ROOT / "openpcdet_tools_eval_shell_retry.log").write_text("\n".join(log_chunks), encoding="utf-8")
    update_cuda_audit_md(status_payload)
    update_summary_report(per_experiment)


if __name__ == "__main__":
    main()
