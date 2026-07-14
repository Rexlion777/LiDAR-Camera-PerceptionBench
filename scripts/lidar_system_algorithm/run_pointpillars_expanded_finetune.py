from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import ensure_dir, read_json_or_default, write_csv, write_json, write_markdown
from scripts.lidar_system_algorithm.run_pointpillars_subset_training import (
    OPENPCDET_ROOT,
    PRETRAINED_CKPT,
    TRAINING_OUTPUT_ROOT,
    TRAINING_REPORT_ROOT,
    WSL_DISTRO,
    WSL_ENV_PREFIX,
    WSL_PYTHON,
    audit_training_environment,
    clean_wsl_output,
    collect_logs,
    ensure_kitti_layout_and_infos,
    latest_checkpoint,
    parse_eval_metrics,
    parse_loss_and_lr,
    resolve_opencpdet_output,
    run_training_command,
    to_wsl_path,
    write_claim_files,
)


UTC = timezone.utc
EXPANDED_SPLIT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune" / "expanded_splits"
CONFIG_TEMPLATE = PROJECT_ROOT / "configs" / "lidar_system_algorithm" / "pointpillar_kitti_subset_finetune.yaml"
SHELL_NATIVE_EVAL = PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "run_openpcdet_eval_with_cuda_env.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanded PointPillars subset fine-tuning on fixed expanded_splits.")
    parser.add_argument("--train-split", default="train_1000")
    parser.add_argument("--val-split", default="val_200")
    parser.add_argument("--mode-name", default="expanded_finetune_1000")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def slug_time() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_split_ids(name: str) -> list[str]:
    path = EXPANDED_SPLIT_ROOT / f"{name}.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_info_pkl_from_split(split_name: str, source_pool: str) -> str:
    data_root = OPENPCDET_ROOT / "data" / "kitti"
    source_pkl = "kitti_infos_train.pkl" if source_pool == "train" else "kitti_infos_val.pkl"
    source_ids = read_split_ids(split_name)
    out_name = f"kitti_infos_{split_name}.pkl"
    script = f"""
cd {shlex.quote(to_wsl_path(PROJECT_ROOT))}
PYTHONPATH=.:'external/OpenPCDet' {shlex.quote(WSL_PYTHON)} - <<'PY'
import json, pickle
from pathlib import Path
data_root = Path({json.dumps(to_wsl_path(data_root))})
sample_ids = set({json.dumps(source_ids)})
with open(data_root / {json.dumps(source_pkl)}, 'rb') as f:
    infos = pickle.load(f)
subset = [item for item in infos if item['point_cloud']['lidar_idx'] in sample_ids]
out_path = data_root / {json.dumps(out_name)}
with open(out_path, 'wb') as f:
    pickle.dump(subset, f)
print(json.dumps({{"out_name": out_path.name, "count": len(subset)}}))
PY
""".strip()
    res = subprocess.run(["wsl", "-d", WSL_DISTRO, "bash", "-lc", script], cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, check=False)
    if res.returncode != 0:
        raise RuntimeError(clean_wsl_output(res.stderr or res.stdout))
    payload = json.loads(clean_wsl_output(res.stdout).strip().splitlines()[-1])
    if int(payload["count"]) != len(source_ids):
        raise RuntimeError(f"Info subset mismatch for {split_name}: expected {len(source_ids)}, got {payload['count']}")
    return str(payload["out_name"])


def make_run_config(exp_dir: Path, args: argparse.Namespace, train_info_pkl: str, val_info_pkl: str) -> Path:
    template = yaml.safe_load(CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    template["DATA_CONFIG"]["_BASE_CONFIG_"] = "tools/cfgs/dataset_configs/kitti_dataset.yaml"
    template["DATA_CONFIG"]["DATA_PATH"] = "data/kitti"
    template["DATA_CONFIG"]["DATA_SPLIT"]["train"] = args.train_split
    template["DATA_CONFIG"]["DATA_SPLIT"]["test"] = args.val_split
    template["DATA_CONFIG"]["INFO_PATH"]["train"] = [train_info_pkl]
    template["DATA_CONFIG"]["INFO_PATH"]["test"] = [val_info_pkl]
    if template["DATA_CONFIG"].get("DATA_AUGMENTOR", {}).get("AUG_CONFIG_LIST"):
        for aug_cfg in template["DATA_CONFIG"]["DATA_AUGMENTOR"]["AUG_CONFIG_LIST"]:
            if aug_cfg.get("NAME") == "gt_sampling":
                aug_cfg["USE_ROAD_PLANE"] = False
    template["OPTIMIZATION"]["NUM_EPOCHS"] = int(args.epochs)
    template["OPTIMIZATION"]["BATCH_SIZE_PER_GPU"] = int(args.batch_size)
    template["OPTIMIZATION"]["LR"] = float(args.learning_rate)
    template["TRAINING_FINETUNE_META"].update(
        {
            "training_mode": args.mode_name,
            "max_samples": len(read_split_ids(args.train_split)),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "pretrained_checkpoint": str(PRETRAINED_CKPT),
            "output_dir": str(TRAINING_OUTPUT_ROOT / exp_dir.name),
            "seed": int(args.seed),
            "train_split_file": f"{args.train_split}.txt",
            "val_split_file": f"{args.val_split}.txt",
            "expanded_split_manifest": str(EXPANDED_SPLIT_ROOT / "split_manifest.json"),
        }
    )
    run_cfg = exp_dir / "run_config.yaml"
    run_cfg.write_text(yaml.safe_dump(template, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return run_cfg


def copy_artifacts(exp_dir: Path, output_mirror: Path, opencpdet_output: Path, ckpt_dir: Path, log_paths: dict[str, Path | None]) -> dict[str, Any]:
    ensure_dir(output_mirror)
    copied_ckpts = []
    for ckpt in sorted(ckpt_dir.glob("*.pth")):
        dst = output_mirror / ckpt.name
        shutil.copy2(ckpt, dst)
        copied_ckpts.append(str(dst))
    copied_logs = {}
    for name, src in log_paths.items():
        if src and src.exists():
            dst = exp_dir / ("train.log" if name == "train_log" else "eval.log")
            shutil.copy2(src, dst)
            shutil.copy2(src, output_mirror / dst.name)
            copied_logs[name] = str(dst)
    return {"checkpoints": copied_ckpts, "logs": copied_logs, "opencpdet_output": str(opencpdet_output)}


def run_shell_native_eval(run_cfg: Path, ckpt: Path, extra_tag: str, eval_tag: str, batch_size: int) -> dict[str, Any]:
    args = [
        to_wsl_path(SHELL_NATIVE_EVAL),
        "--cfg-file",
        to_wsl_path(run_cfg),
        "--ckpt",
        to_wsl_path(ckpt),
        "--extra-tag",
        extra_tag,
        "--eval-tag",
        eval_tag,
        "--batch-size",
        str(batch_size),
    ]
    command = f"bash {' '.join(shlex.quote(x) for x in args)}"
    result = subprocess.run(
        ["wsl", "-d", WSL_DISTRO, "bash", "-lc", command],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=7200,
        check=False,
    )
    stdout = clean_wsl_output(result.stdout)
    stderr = clean_wsl_output(result.stderr)
    tools_ok = result.returncode == 0 and "TOOLS_TEST_COMPLETED" in f"{stdout}\n{stderr}"
    return {
        "status": "completed" if tools_ok else "failed",
        "returncode": result.returncode,
        "command": command,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def run_external_eval(run_cfg: Path, ckpt: Path, split_name: str, output_dir: Path, pred_dir: Path, max_frames: int = 0) -> dict[str, Any]:
    split_file = EXPANDED_SPLIT_ROOT / f"{split_name}.txt"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "run_kitti_official_eval.py"),
        "--kitti-root",
        str(PROJECT_ROOT / "data" / "kitti_object_raw" / "extracted"),
        "--openpcdet-root",
        "external/OpenPCDet",
        "--cfg-file",
        str(run_cfg),
        "--ckpt",
        str(ckpt),
        "--split-file",
        str(split_file),
        "--output-dir",
        str(output_dir),
        "--pred-dir",
        str(pred_dir),
        "--max-frames",
        str(max_frames),
    ]
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200, check=False)
    payload = read_json_or_default(output_dir / "kitti_official_eval.json", {})
    return {
        "status": "completed" if completed.returncode == 0 and isinstance(payload, dict) and payload.get("status") == "completed" else "failed",
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "eval_json": str(output_dir / "kitti_official_eval.json"),
        "payload": payload,
    }


def comparison_rows(eval_payload: dict[str, Any]) -> list[dict[str, Any]]:
    result_dict = eval_payload.get("official_result_dict", {}) if isinstance(eval_payload, dict) else {}
    return [{"metric": key, "value": value} for key, value in sorted(result_dict.items())]


def eval_mean_ap(eval_payload: dict[str, Any]) -> float | None:
    result_dict = eval_payload.get("official_result_dict", {}) if isinstance(eval_payload, dict) else {}
    keys = ("Car_3d/moderate_R40", "Pedestrian_3d/moderate_R40", "Cyclist_3d/moderate_R40")
    vals = [result_dict.get(key) for key in keys if isinstance(result_dict.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None


def write_eval_log(
    exp_dir: Path,
    native_baseline: dict[str, Any],
    native_final: dict[str, Any],
    baseline_external: dict[str, Any],
    final_external: dict[str, Any],
) -> None:
    lines = [
        "# Expanded Fine-tune Eval Log",
        "",
        "[native_tools_test.py][baseline]",
        f"status={native_baseline.get('status')}",
        f"returncode={native_baseline.get('returncode')}",
        f"command={native_baseline.get('command', '')}",
        "",
        native_baseline.get("stdout_tail", ""),
        native_baseline.get("stderr_tail", ""),
        "",
        "[native_tools_test.py][final]",
        f"status={native_final.get('status')}",
        f"returncode={native_final.get('returncode')}",
        f"command={native_final.get('command', '')}",
        "",
        native_final.get("stdout_tail", ""),
        native_final.get("stderr_tail", ""),
        "",
        "[external_official_eval][baseline]",
        f"status={baseline_external.get('status')}",
        f"returncode={baseline_external.get('returncode')}",
        f"eval_json={baseline_external.get('eval_json', '')}",
        "",
        baseline_external.get("stdout_tail", ""),
        baseline_external.get("stderr_tail", ""),
        "",
        "[external_official_eval][final]",
        f"status={final_external.get('status')}",
        f"returncode={final_external.get('returncode')}",
        f"eval_json={final_external.get('eval_json', '')}",
        "",
        final_external.get("stdout_tail", ""),
        final_external.get("stderr_tail", ""),
        "",
        "Boundary: this eval.log records explicit native tools/test.py runs and external official eval wrapper runs.",
        "Boundary: it does not claim OpenPCDet train.py inline repeat-eval succeeded.",
    ]
    write_markdown(exp_dir / "eval.log", "\n".join(lines))


def main() -> None:
    args = parse_args()
    audit_training_environment()
    ensure_kitti_layout_and_infos()

    exp_name = f"{args.mode_name}_{slug_time()}"
    exp_dir = ensure_dir(TRAINING_REPORT_ROOT / exp_name)
    output_mirror = ensure_dir(TRAINING_OUTPUT_ROOT / exp_name)
    if args.skip_existing and (exp_dir / "train_status.json").exists():
        print((exp_dir / "train_status.json").read_text(encoding="utf-8"))
        return

    train_ids = read_split_ids(args.train_split)
    val_ids = read_split_ids(args.val_split)
    (exp_dir / "dataset_split_used.txt").write_text(
        "\n".join(
            [
                f"train_split_name={args.train_split}",
                f"val_split_name={args.val_split}",
                f"train_count={len(train_ids)}",
                f"val_count={len(val_ids)}",
                "",
                "[train_ids]",
                *train_ids,
                "",
                "[val_ids]",
                *val_ids,
            ]
        ),
        encoding="utf-8",
    )

    train_info_pkl = build_info_pkl_from_split(args.train_split, "train")
    val_info_pkl = build_info_pkl_from_split(args.val_split, "train")
    run_cfg = make_run_config(exp_dir, args, train_info_pkl, val_info_pkl)

    started_at = now_iso()
    train_result, train_command = run_training_command(
        run_cfg=run_cfg,
        plan=type("Plan", (), {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "use_pretrained": True})(),
        extra_tag=exp_name,
        seed=args.seed,
        workers=args.workers,
        resume=False,
    )
    finished_at = now_iso()

    opencpdet_output = resolve_opencpdet_output(run_cfg, exp_name)
    ckpt_dir = opencpdet_output / "ckpt"
    final_ckpt = latest_checkpoint(ckpt_dir)
    log_paths = collect_logs(opencpdet_output)
    copied = copy_artifacts(exp_dir, output_mirror, opencpdet_output, ckpt_dir, log_paths)

    native_baseline = run_shell_native_eval(run_cfg, PRETRAINED_CKPT, exp_name, "baseline_subset_eval_native", args.eval_batch_size)
    native_final = run_shell_native_eval(run_cfg, final_ckpt, exp_name, "final_subset_eval_native", args.eval_batch_size) if final_ckpt else {"status": "failed", "reason": "missing_final_checkpoint"}

    baseline_external_dir = ensure_dir(exp_dir / "baseline_official_eval")
    final_external_dir = ensure_dir(exp_dir / "final_official_eval")
    baseline_external = run_external_eval(
        run_cfg,
        PRETRAINED_CKPT,
        args.val_split,
        baseline_external_dir,
        PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "results" / f"{exp_name}_baseline_txt",
    )
    final_external = run_external_eval(
        run_cfg,
        final_ckpt,
        args.val_split,
        final_external_dir,
        PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "results" / f"{exp_name}_final_txt",
    ) if final_ckpt else {"status": "failed", "reason": "missing_final_checkpoint", "payload": {}}
    write_eval_log(exp_dir, native_baseline, native_final, baseline_external, final_external)

    train_log_text = (exp_dir / "train.log").read_text(encoding="utf-8", errors="ignore") if (exp_dir / "train.log").exists() else ""
    loss_rows, lr_rows = parse_loss_and_lr(train_log_text)
    write_csv(
        exp_dir / "loss_curve.csv",
        loss_rows or [{"iteration": 0, "epoch": 0, "epoch_total": 0, "training_loss": "", "training_loss_avg": ""}],
        ["iteration", "epoch", "epoch_total", "training_loss", "training_loss_avg"],
    )
    write_csv(
        exp_dir / "lr_schedule.csv",
        lr_rows or [{"iteration": 0, "epoch": 0, "epoch_total": 0, "learning_rate": ""}],
        ["iteration", "epoch", "epoch_total", "learning_rate"],
    )
    final_payload = final_external.get("payload", {}) if isinstance(final_external, dict) else {}
    write_csv(
        exp_dir / "eval_ap_summary.csv",
        comparison_rows(final_payload) or [{"metric": "status", "value": ""}],
        ["metric", "value"],
    )
    write_json(exp_dir / "checkpoint_index.json", [{"checkpoint": path} for path in copied["checkpoints"]])
    write_markdown(
        exp_dir / "safe_claims.md",
        "\n".join(
            [
                "- Completed expanded subset fine-tuning on fixed train/val splits.",
                "- Native OpenPCDet tools/test.py eval was executed through the shell-level CUDA/NVVM wrapper.",
                "- External official eval wrapper was also kept for comparable AP output on the same val split.",
            ]
        ),
    )
    write_markdown(
        exp_dir / "forbidden_claims.md",
        "\n".join(
            [
                "- Do not claim full KITTI training.",
                "- Do not claim SOTA.",
                "- Do not call this complete convergence after only 3 epochs.",
                "- Do not write subset val as full KITTI val.",
            ]
        ),
    )
    write_markdown(exp_dir / "train_command.txt", train_command)
    write_markdown(exp_dir / "eval_command.txt", "\n".join([native_final.get("command", ""), str(final_external_dir / "kitti_official_eval.json")]))

    inline_eval_status = "completed" if train_result.returncode == 0 else "failed_due_train_py_inline_eval"
    training_status = "completed" if final_ckpt else "partial"
    external_status = "completed" if baseline_external.get("status") == "completed" and final_external.get("status") == "completed" else "partial"
    native_status = "completed" if native_baseline.get("status") == "completed" and native_final.get("status") == "completed" else "failed_tools_eval_runtime"
    overall = "subset_training_eval_completed_with_native_tools_eval" if training_status == "completed" and external_status == "completed" and native_status == "completed" else "subset_training_eval_completed_with_tools_eval_issue"

    metrics_summary = {
        "experiment_name": exp_name,
        "training_mode": args.mode_name,
        "status": "completed" if overall == "subset_training_eval_completed_with_native_tools_eval" else "partial",
        "train_sample_count": len(train_ids),
        "val_sample_count": len(val_ids),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "used_pretrained_checkpoint": True,
        "pretrained_checkpoint": str(PRETRAINED_CKPT),
        "trained_checkpoint": str(final_ckpt) if final_ckpt else None,
        "opencpdet_output_dir": str(opencpdet_output),
        "baseline_eval": baseline_external.get("payload", {}),
        "final_eval": final_external.get("payload", {}),
        "start_time": started_at,
        "end_time": finished_at,
        "seed": args.seed,
        "run_config": str(run_cfg),
        "train_command": train_command,
        "native_eval": {"baseline": native_baseline, "final": native_final},
        "external_eval_wrapper": {"baseline": baseline_external, "final": final_external},
        "experiment_status": {
            "training_status": training_status,
            "checkpoint_status": "completed" if final_ckpt else "missing_checkpoint",
            "external_eval_wrapper_status": external_status,
            "opencpdet_tools_eval_status": native_status,
            "train_py_inline_eval_status": inline_eval_status,
            "deployment_diagnostics_status": "not_run",
            "overall_claim_level": overall,
        },
    }
    write_json(exp_dir / "metrics_summary.json", metrics_summary)
    train_status = {
        **metrics_summary,
        "stdout_tail": clean_wsl_output(train_result.stdout[-4000:]),
        "stderr_tail": clean_wsl_output(train_result.stderr[-4000:]),
    }
    write_json(exp_dir / "train_status.json", train_status)
    print(json.dumps({"status": metrics_summary["status"], "experiment_dir": str(exp_dir), "trained_checkpoint": metrics_summary["trained_checkpoint"], "final_mean_ap": eval_mean_ap(final_payload)}, indent=2))


if __name__ == "__main__":
    main()
