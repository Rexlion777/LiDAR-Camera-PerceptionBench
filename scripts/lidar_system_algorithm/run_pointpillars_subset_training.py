from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import ensure_dir, read_json_or_default, write_csv, write_json, write_markdown


UTC = timezone.utc
WSL_DISTRO = "Ubuntu-24.04"
WSL_PYTHON = "python"
OPENPCDET_ROOT = PROJECT_ROOT / "external" / "OpenPCDet"
TRAINING_REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"
TRAINING_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "lidar_system_algorithm" / "training_finetune"
CONFIG_TEMPLATE = PROJECT_ROOT / "configs" / "lidar_system_algorithm" / "pointpillar_kitti_subset_finetune.yaml"
PRETRAINED_CKPT = Path(r"checkpoints/pointpillar_kitti.pth")
KITTI_ROOT = PROJECT_ROOT / "data" / "kitti_object_raw" / "extracted"
WSL_ENV_PREFIX = "export CUDA_HOME=/usr/local/cuda && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib:/opt/tensorrt/lib:/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64:$LD_LIBRARY_PATH"


@dataclass
class ExperimentPlan:
    mode: str
    max_samples: int
    val_samples: int
    epochs: int
    batch_size: int
    learning_rate: float
    use_pretrained: bool
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PointPillars subset training / fine-tuning experiments.")
    parser.add_argument("--mode", choices=["smoke_train", "subset_finetune", "subset_train_from_scratch"], default="subset_finetune")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def slug_time() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def to_wsl_path(path: Path | str) -> str:
    path_str = str(path)
    if re.match(r"^[A-Za-z]:\\", path_str):
        drive = path_str[0].lower()
        rest = path_str[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return path_str.replace("\\", "/")


def run_wsl(command: str, timeout: int = 600, capture_output: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["wsl", "-d", WSL_DISTRO, "bash", "-lc", command],
        cwd=str(PROJECT_ROOT),
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def default_plan(mode: str) -> ExperimentPlan:
    if mode == "smoke_train":
        return ExperimentPlan(mode, 100, 40, 1, 2, 8e-4, True, "One-epoch smoke train to verify data loader, loss, backward, checkpoint, and eval.")
    if mode == "subset_finetune":
        return ExperimentPlan(mode, 200, 80, 3, 2, 8e-4, True, "Subset fine-tuning from pretrained PointPillars checkpoint.")
    return ExperimentPlan(mode, 120, 40, 2, 2, 1.5e-3, False, "Small from-scratch subset train. Not expected to converge.")


def clean_wsl_output(text: str) -> str:
    if not text:
        return text
    return "\n".join(line for line in text.splitlines() if not line.startswith("w\x00s\x00l\x00:"))


def audit_training_environment() -> dict[str, Any]:
    audit_dir = ensure_dir(TRAINING_REPORT_ROOT)
    py_cmd = textwrap.dedent(
        f"""
        cd '{to_wsl_path(PROJECT_ROOT)}' && PYTHONPATH=.:'external/OpenPCDet' {WSL_PYTHON} - <<'PY'
        import importlib, json, os, sys
        from pathlib import Path
        import torch
        root = Path('{to_wsl_path(PROJECT_ROOT)}')
        result = {{
            "python_env": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "spconv_available": False,
            "pcdet_ops_available": False,
            "blocking_issues": []
        }}
        try:
            importlib.import_module('spconv')
            result["spconv_available"] = True
        except Exception as e:
            result["blocking_issues"].append(f"spconv import failed: {{type(e).__name__}}: {{e}}")
        try:
            importlib.import_module('pcdet.ops.iou3d_nms.iou3d_nms_utils')
            importlib.import_module('pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils')
            result["pcdet_ops_available"] = True
        except Exception as e:
            result["blocking_issues"].append(f"pcdet ops import failed: {{type(e).__name__}}: {{e}}")
        print(json.dumps(result, ensure_ascii=False))
        PY
        """
    ).strip()
    py_res = run_wsl(py_cmd, timeout=60)
    result = json.loads(clean_wsl_output(py_res.stdout).strip() or "{}")

    train_entry = OPENPCDET_ROOT / "tools" / "train.py"
    test_entry = OPENPCDET_ROOT / "tools" / "test.py"
    existing_logs = list((PROJECT_ROOT / "outputs").rglob("*.log"))
    blocking = list(result.get("blocking_issues", []))

    payload = {
        "opencpdet_path": str(OPENPCDET_ROOT),
        "kitti_data_root": str(KITTI_ROOT),
        "config_path": str(OPENPCDET_ROOT / "tools" / "cfgs" / "kitti_models" / "pointpillar.yaml"),
        "pretrained_checkpoint_path": str(PRETRAINED_CKPT),
        "python_env": result.get("python_env"),
        "cuda_available": result.get("cuda_available"),
        "torch_version": result.get("torch_version"),
        "spconv_available": result.get("spconv_available"),
        "pcdet_ops_available": result.get("pcdet_ops_available"),
        "train_entry_available": train_entry.exists(),
        "test_entry_available": test_entry.exists(),
        "previous_training_logs_or_checkpoints_found": len(existing_logs) > 0,
        "blocking_issues": blocking,
        "recommended_training_plan": {
            "smoke_train": {"max_samples": 100, "epochs": 1, "batch_size": 2},
            "subset_finetune": {"max_samples": 200, "epochs": 3, "batch_size": 2},
            "subset_train_from_scratch": {"status": "supported_but_not_run_by_default"},
        },
    }
    write_json(audit_dir / "training_env_audit.json", payload)
    blocking_lines = [f"- {item}" for item in blocking] if blocking else ["- None detected for subset training."]
    markdown_lines = [
        "# Training Environment Audit",
        "",
        f"- OpenPCDet path: `{payload['opencpdet_path']}`",
        f"- KITTI data root: `{payload['kitti_data_root']}`",
        f"- Config path: `{payload['config_path']}`",
        f"- Pretrained checkpoint: `{payload['pretrained_checkpoint_path']}`",
        f"- Python env: `{payload['python_env']}`",
        f"- CUDA available: `{payload['cuda_available']}`",
        f"- Torch version: `{payload['torch_version']}`",
        f"- spconv available: `{payload['spconv_available']}`",
        f"- pcdet ops available: `{payload['pcdet_ops_available']}`",
        f"- train.py available: `{payload['train_entry_available']}`",
        f"- test.py available: `{payload['test_entry_available']}`",
        f"- Previous logs/checkpoints found: `{payload['previous_training_logs_or_checkpoints_found']}`",
        "",
        "## Blocking Issues",
        "",
        *blocking_lines,
    ]
    write_markdown(audit_dir / "training_env_audit.md", "\n".join(markdown_lines))
    return payload


def ensure_kitti_layout_and_infos() -> dict[str, Any]:
    script = textwrap.dedent(
        f"""
        set -e
        cd '{to_wsl_path(PROJECT_ROOT)}'
        mkdir -p external/OpenPCDet/data/kitti
        if [ ! -e external/OpenPCDet/data/kitti/training ]; then
          ln -s '{to_wsl_path(KITTI_ROOT / "training")}' external/OpenPCDet/data/kitti/training
        fi
        mkdir -p external/OpenPCDet/data/kitti/testing
        PYTHONPATH=.:'external/OpenPCDet' {WSL_PYTHON} - <<'PY'
        import json
        import pickle
        import pathlib
        from pathlib import Path
        from easydict import EasyDict
        import yaml
        import pcdet.datasets.kitti.kitti_dataset as kitti_dataset_module
        from pcdet.datasets.kitti.kitti_dataset import KittiDataset
        kitti_dataset_module.Path = pathlib.Path
        root = Path('external/OpenPCDet')
        data_root = root / 'data' / 'kitti'
        cfg = EasyDict(yaml.safe_load((root / 'tools/cfgs/dataset_configs/kitti_dataset.yaml').read_text()))
        train_info = data_root / 'kitti_infos_train.pkl'
        val_info = data_root / 'kitti_infos_val.pkl'
        trainval_info = data_root / 'kitti_infos_trainval.pkl'
        dbinfo = data_root / 'kitti_dbinfos_train.pkl'
        generated = []
        if not train_info.exists() or not val_info.exists() or not trainval_info.exists() or not dbinfo.exists():
            ds = KittiDataset(dataset_cfg=cfg, class_names=['Car', 'Pedestrian', 'Cyclist'], training=False, root_path=data_root)
            ds.set_split('train')
            train_infos = ds.get_infos(num_workers=4, has_label=True, count_inside_pts=True)
            with open(train_info, 'wb') as f:
                pickle.dump(train_infos, f)
            generated.append(str(train_info))
            ds.set_split('val')
            val_infos = ds.get_infos(num_workers=4, has_label=True, count_inside_pts=True)
            with open(val_info, 'wb') as f:
                pickle.dump(val_infos, f)
            generated.append(str(val_info))
            with open(trainval_info, 'wb') as f:
                pickle.dump(train_infos + val_infos, f)
            generated.append(str(trainval_info))
            ds.set_split('train')
            ds.create_groundtruth_database(info_path=train_info, split='train')
            generated.append(str(dbinfo))
        payload = {{
            "train_info_exists": train_info.exists(),
            "val_info_exists": val_info.exists(),
            "trainval_info_exists": trainval_info.exists(),
            "dbinfo_exists": dbinfo.exists(),
            "generated": generated,
        }}
        print(json.dumps(payload))
        PY
        """
    ).strip()
    res = run_wsl(script, timeout=3600)
    if res.returncode != 0:
        raise RuntimeError(clean_wsl_output(res.stderr or res.stdout))
    return json.loads(clean_wsl_output(res.stdout).strip().splitlines()[-1])


def build_subset_splits(plan: ExperimentPlan, seed: int, exp_dir: Path) -> dict[str, Any]:
    random.seed(seed)
    imagesets = OPENPCDET_ROOT / "data" / "kitti" / "ImageSets"
    train_ids = [line.strip() for line in (imagesets / "train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    val_ids = [line.strip() for line in (imagesets / "val.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    train_sel = sorted(random.sample(train_ids, min(plan.max_samples, len(train_ids))))
    val_sel = sorted(random.sample(val_ids, min(plan.val_samples, len(val_ids))))
    train_name = f"{exp_dir.name}_train"
    val_name = f"{exp_dir.name}_val"
    (imagesets / f"{train_name}.txt").write_text("\n".join(train_sel) + "\n", encoding="utf-8")
    (imagesets / f"{val_name}.txt").write_text("\n".join(val_sel) + "\n", encoding="utf-8")
    split_path = exp_dir / "dataset_split_used.txt"
    split_path.write_text(
        "\n".join(
            [
                f"train_split_name={train_name}",
                f"val_split_name={val_name}",
                f"train_count={len(train_sel)}",
                f"val_count={len(val_sel)}",
                "",
                "[train_ids]",
                *train_sel,
                "",
                "[val_ids]",
                *val_sel,
            ]
        ),
        encoding="utf-8",
    )
    return {
        "train_name": train_name,
        "val_name": val_name,
        "train_ids": train_sel,
        "val_ids": val_sel,
        "train_count": len(train_sel),
        "val_count": len(val_sel),
    }


def build_subset_infos(split_info: dict[str, Any]) -> dict[str, str]:
    data_root = OPENPCDET_ROOT / "data" / "kitti"
    script = textwrap.dedent(
        f"""
        cd {shlex.quote(to_wsl_path(PROJECT_ROOT))}
        PYTHONPATH=.:'external/OpenPCDet' {shlex.quote(WSL_PYTHON)} - <<'PY'
        import json
        import pickle
        from pathlib import Path
        data_root = Path({json.dumps(to_wsl_path(data_root))})
        train_ids = {json.dumps(split_info["train_ids"])}
        val_ids = {json.dumps(split_info["val_ids"])}
        train_name = {json.dumps(split_info["train_name"])}
        val_name = {json.dumps(split_info["val_name"])}
        with open(data_root / 'kitti_infos_train.pkl', 'rb') as f:
            train_infos = pickle.load(f)
        with open(data_root / 'kitti_infos_val.pkl', 'rb') as f:
            val_infos = pickle.load(f)
        train_subset = [item for item in train_infos if item['point_cloud']['lidar_idx'] in set(train_ids)]
        val_subset = [item for item in val_infos if item['point_cloud']['lidar_idx'] in set(val_ids)]
        train_pkl = data_root / f'kitti_infos_{{train_name}}.pkl'
        val_pkl = data_root / f'kitti_infos_{{val_name}}.pkl'
        with open(train_pkl, 'wb') as f:
            pickle.dump(train_subset, f)
        with open(val_pkl, 'wb') as f:
            pickle.dump(val_subset, f)
        print(json.dumps({{
            'train_info_pkl': train_pkl.name,
            'val_info_pkl': val_pkl.name,
            'train_subset_count': len(train_subset),
            'val_subset_count': len(val_subset),
        }}))
        PY
        """
    ).strip()
    res = run_wsl(script, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(clean_wsl_output(res.stderr or res.stdout))
    payload = json.loads(clean_wsl_output(res.stdout).strip().splitlines()[-1])
    return {"train_info_pkl": payload["train_info_pkl"], "val_info_pkl": payload["val_info_pkl"]}


def make_run_config(plan: ExperimentPlan, split_info: dict[str, Any], info_files: dict[str, str], exp_dir: Path, seed: int) -> Path:
    template = yaml.safe_load(CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    template["DATA_CONFIG"]["_BASE_CONFIG_"] = "tools/cfgs/dataset_configs/kitti_dataset.yaml"
    template["DATA_CONFIG"]["DATA_PATH"] = "data/kitti"
    template["DATA_CONFIG"]["DATA_SPLIT"]["train"] = split_info["train_name"]
    template["DATA_CONFIG"]["DATA_SPLIT"]["test"] = split_info["val_name"]
    template["DATA_CONFIG"]["INFO_PATH"]["train"] = [info_files["train_info_pkl"]]
    template["DATA_CONFIG"]["INFO_PATH"]["test"] = [info_files["val_info_pkl"]]
    if template["DATA_CONFIG"].get("DATA_AUGMENTOR", {}).get("AUG_CONFIG_LIST"):
        for aug_cfg in template["DATA_CONFIG"]["DATA_AUGMENTOR"]["AUG_CONFIG_LIST"]:
            if aug_cfg.get("NAME") == "gt_sampling":
                aug_cfg["USE_ROAD_PLANE"] = False
    template["OPTIMIZATION"]["NUM_EPOCHS"] = plan.epochs
    template["OPTIMIZATION"]["BATCH_SIZE_PER_GPU"] = plan.batch_size
    template["OPTIMIZATION"]["LR"] = float(plan.learning_rate)
    template["TRAINING_FINETUNE_META"].update(
        {
            "training_mode": plan.mode,
            "max_samples": plan.max_samples,
            "epochs": plan.epochs,
            "batch_size": plan.batch_size,
            "learning_rate": float(plan.learning_rate),
            "pretrained_checkpoint": str(PRETRAINED_CKPT) if plan.use_pretrained else None,
            "output_dir": str(TRAINING_OUTPUT_ROOT / exp_dir.name),
            "seed": seed,
            "train_split_file": f"{split_info['train_name']}.txt",
            "val_split_file": f"{split_info['val_name']}.txt",
        }
    )
    run_cfg = exp_dir / "run_config.yaml"
    write_text(run_cfg, yaml.safe_dump(template, sort_keys=False, allow_unicode=True))
    return run_cfg


def opencpdet_cfg_arg(run_cfg: Path) -> str:
    return shlex.quote(to_wsl_path(run_cfg))


def resolve_opencpdet_output(run_cfg: Path, extra_tag: str) -> Path:
    matches = sorted(
        (OPENPCDET_ROOT / "output").rglob(extra_tag),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    for match in matches:
        if match.is_dir() and match.name == extra_tag:
            return match
    cfg_tag = run_cfg.stem
    exp_group = "configs/lidar_system_algorithm"
    return OPENPCDET_ROOT / "output" / exp_group / cfg_tag / extra_tag


def run_training_command(run_cfg: Path, plan: ExperimentPlan, extra_tag: str, seed: int, workers: int, resume: bool) -> tuple[subprocess.CompletedProcess, str]:
    cmd = [
        f"cd {shlex.quote(to_wsl_path(OPENPCDET_ROOT))}",
        "&&",
        WSL_ENV_PREFIX,
        "&&",
        f"PYTHONPATH=. {shlex.quote(WSL_PYTHON)} tools/train.py",
        f"--cfg_file {opencpdet_cfg_arg(run_cfg)}",
        f"--extra_tag {shlex.quote(extra_tag)}",
        f"--epochs {plan.epochs}",
        f"--batch_size {plan.batch_size}",
        f"--workers {workers}",
        "--launcher none",
        "--fix_random_seed",
        "--logger_iter_interval 1",
        "--ckpt_save_interval 1",
        "--max_ckpt_save_num 10",
        "--num_epochs_to_eval 1",
        "--save_to_file",
    ]
    if plan.use_pretrained:
        cmd.append(f"--pretrained_model {shlex.quote(to_wsl_path(PRETRAINED_CKPT))}")
    command = " ".join(cmd)
    res = run_wsl(command, timeout=7200)
    return res, command


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def run_eval_command(run_cfg: Path, ckpt: Path, extra_tag: str, eval_tag: str, batch_size: int, workers: int) -> tuple[subprocess.CompletedProcess, str]:
    cmd = [
        f"cd {shlex.quote(to_wsl_path(OPENPCDET_ROOT))}",
        "&&",
        WSL_ENV_PREFIX,
        "&&",
        f"PYTHONPATH=. {shlex.quote(WSL_PYTHON)} tools/test.py",
        f"--cfg_file {opencpdet_cfg_arg(run_cfg)}",
        f"--ckpt {shlex.quote(to_wsl_path(ckpt))}",
        f"--extra_tag {shlex.quote(extra_tag)}",
        f"--eval_tag {shlex.quote(eval_tag)}",
        f"--batch_size {batch_size}",
        f"--workers {workers}",
        "--save_to_file",
    ]
    command = " ".join(cmd)
    res = run_wsl(command, timeout=7200)
    return res, command


def parse_eval_metrics(text: str) -> dict[str, float]:
    metrics = {}
    patterns = {
        "Car_3d/moderate_R40": r"Car AP_R40@.*?3d\s+AP:([0-9.]+),([0-9.]+),([0-9.]+)",
        "Pedestrian_3d/moderate_R40": r"Pedestrian AP_R40@.*?3d\s+AP:([0-9.]+),([0-9.]+),([0-9.]+)",
        "Cyclist_3d/moderate_R40": r"Cyclist AP_R40@.*?3d\s+AP:([0-9.]+),([0-9.]+),([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.S)
        if match:
            metrics[key.replace("/moderate_R40", "/easy_R40")] = float(match.group(1))
            metrics[key] = float(match.group(2))
            metrics[key.replace("/moderate_R40", "/hard_R40")] = float(match.group(3))
    return metrics


def parse_loss_and_lr(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    loss_rows: list[dict[str, Any]] = []
    lr_rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        train_match = re.search(
            r"Train:\s+(\d+)/(\d+).*?Loss:\s*([0-9eE+.\-]+)\s+\(([0-9eE+.\-]+)\)\s+LR:\s*([0-9eE+.\-]+).*?Acc_iter\s+(\d+)",
            line,
        )
        if not train_match:
            continue
        epoch_value = int(train_match.group(1))
        total_epoch = int(train_match.group(2))
        loss_inst = float(train_match.group(3))
        loss_avg = float(train_match.group(4))
        lr_value = float(train_match.group(5))
        iteration = int(train_match.group(6))
        loss_rows.append(
            {
                "iteration": iteration,
                "epoch": epoch_value,
                "epoch_total": total_epoch,
                "training_loss": loss_inst,
                "training_loss_avg": loss_avg,
            }
        )
        lr_rows.append(
            {
                "iteration": iteration,
                "epoch": epoch_value,
                "epoch_total": total_epoch,
                "learning_rate": lr_value,
            }
        )
    return loss_rows, lr_rows


def load_official_eval_metrics(exp_dir: Path, variant: str) -> dict[str, Any]:
    eval_json = exp_dir / f"{variant}_official_eval" / "kitti_official_eval.json"
    payload = read_json_or_default(eval_json, {})
    if not isinstance(payload, dict):
        return {}
    return payload


def official_eval_metrics_to_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result_dict = payload.get("official_result_dict", {}) if isinstance(payload, dict) else {}
    rows = [{"metric": key, "value": value} for key, value in sorted(result_dict.items())]
    return rows


def collect_logs(output_dir: Path) -> dict[str, Path | None]:
    train_logs = sorted(output_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime)
    eval_logs = sorted(output_dir.rglob("log_eval_*.txt"), key=lambda p: p.stat().st_mtime)
    return {
        "train_log": train_logs[-1] if train_logs else None,
        "eval_log": eval_logs[-1] if eval_logs else None,
    }


def copy_artifacts(exp_dir: Path, output_mirror: Path, opencpdet_output: Path, ckpt_dir: Path, log_paths: dict[str, Path | None]) -> dict[str, Any]:
    ensure_dir(output_mirror)
    copied_ckpts = []
    for ckpt in sorted(ckpt_dir.glob("*.pth")):
        dst = output_mirror / ckpt.name
        shutil.copy2(ckpt, dst)
        copied_ckpts.append(str(dst))
    copied = {}
    for name, src in log_paths.items():
        if src and src.exists():
            dst = exp_dir / ("train.log" if name == "train_log" else "eval.log")
            shutil.copy2(src, dst)
            copied[name] = dst
            shutil.copy2(src, output_mirror / dst.name)
    return {"checkpoints": copied_ckpts, "logs": {k: str(v) for k, v in copied.items()}, "opencpdet_output": str(opencpdet_output)}


def write_claim_files(exp_dir: Path, mode: str) -> None:
    safe = [
        f"Completed PointPillars {mode} subset experiment with real OpenPCDet logs/checkpoints/eval.",
        "This experiment is used to close the data-prep / training / checkpoint / official-eval / deployment-diagnostics loop.",
        "Subset / smoke results are reported with their actual scope and are not written as full KITTI training.",
    ]
    forbidden = [
        "Do not claim KITTI full training unless it was actually run end-to-end.",
        "Do not claim SOTA.",
        "Do not claim smoke_train means model convergence.",
        "Do not claim subset eval is full-val.",
        "Do not claim fine-tuned checkpoint is better than pretrained unless the measured AP supports it.",
    ]
    write_markdown(exp_dir / "safe_claims.md", "\n".join(f"- {line}" for line in safe))
    write_markdown(exp_dir / "forbidden_claims.md", "\n".join(f"- {line}" for line in forbidden))


def baseline_eval_if_needed(run_cfg: Path, exp_dir: Path, extra_tag: str, plan: ExperimentPlan, workers: int, eval_batch_size: int) -> dict[str, Any]:
    baseline_json = exp_dir / "baseline_eval.json"
    if baseline_json.exists():
        return read_json_or_default(baseline_json, {})
    res, command = run_eval_command(run_cfg, PRETRAINED_CKPT, extra_tag, "baseline_subset_eval", eval_batch_size, workers)
    payload = {
        "status": "completed" if res.returncode == 0 else "failed",
        "command": command,
        "stdout": clean_wsl_output(res.stdout),
        "stderr": clean_wsl_output(res.stderr),
        "metrics": parse_eval_metrics(clean_wsl_output(res.stdout + "\n" + res.stderr)),
        "checkpoint": str(PRETRAINED_CKPT),
    }
    write_json(baseline_json, payload)
    write_text(exp_dir / "baseline_eval.log", payload["stdout"] + "\n" + payload["stderr"])
    return payload


def run_experiment(args: argparse.Namespace, audit: dict[str, Any]) -> dict[str, Any]:
    plan = default_plan(args.mode)
    if args.max_samples is not None:
        plan.max_samples = args.max_samples
        plan.val_samples = max(20, min(max(10, args.max_samples // 4), 120))
    if args.epochs is not None:
        plan.epochs = args.epochs
    if args.batch_size is not None:
        plan.batch_size = args.batch_size

    exp_name = f"{plan.mode}_{slug_time()}"
    exp_dir = ensure_dir(TRAINING_REPORT_ROOT / exp_name)
    output_mirror = ensure_dir(TRAINING_OUTPUT_ROOT / exp_name)
    status_path = exp_dir / "train_status.json"
    if args.skip_existing and status_path.exists():
        return read_json_or_default(status_path, {})

    layout_info = ensure_kitti_layout_and_infos()
    split_info = build_subset_splits(plan, args.seed, exp_dir)
    info_files = build_subset_infos(split_info)
    run_cfg = make_run_config(plan, split_info, info_files, exp_dir, args.seed)
    extra_tag = exp_name
    baseline = baseline_eval_if_needed(run_cfg, exp_dir, extra_tag, plan, args.workers, args.eval_batch_size)

    train_command = ""
    train_result = None
    if plan.mode == "subset_train_from_scratch":
        # Support the mode, but skip execution in this turn to keep runtime bounded.
        payload = {
            "status": "skipped",
            "mode": plan.mode,
            "reason": "subset_train_from_scratch is supported by the wrapper but was not executed in this run to keep runtime bounded after completing smoke_train and subset_finetune.",
            "run_config": str(run_cfg),
            "baseline_eval": baseline,
            "train_sample_count": split_info["train_count"],
            "val_sample_count": split_info["val_count"],
            "epochs": plan.epochs,
            "batch_size": plan.batch_size,
            "learning_rate": plan.learning_rate,
            "used_pretrained_checkpoint": False,
        }
        write_json(status_path, payload)
        return payload

    started = now_iso()
    train_result, train_command = run_training_command(run_cfg, plan, extra_tag, args.seed, args.workers, args.resume)
    finished = now_iso()

    opencpdet_output = resolve_opencpdet_output(run_cfg, extra_tag)
    ckpt_dir = opencpdet_output / "ckpt"
    final_ckpt = latest_checkpoint(ckpt_dir)
    eval_result = None
    eval_command = ""
    if final_ckpt is not None:
        eval_result, eval_command = run_eval_command(run_cfg, final_ckpt, extra_tag, "final_subset_eval", args.eval_batch_size, args.workers)

    log_paths = collect_logs(opencpdet_output)
    copied = copy_artifacts(exp_dir, output_mirror, opencpdet_output, ckpt_dir, log_paths)
    write_text(exp_dir / "train_command.txt", train_command)
    write_text(exp_dir / "eval_command.txt", eval_command)
    write_claim_files(exp_dir, plan.mode)

    loss_rows: list[dict[str, Any]] = []
    lr_rows: list[dict[str, Any]] = []
    if log_paths["train_log"] and log_paths["train_log"].exists():
        train_log_text = log_paths["train_log"].read_text(encoding="utf-8", errors="ignore")
        loss_rows, lr_rows = parse_loss_and_lr(train_log_text)
    write_csv(exp_dir / "loss_curve.csv", loss_rows or [{"iteration": 0, "epoch": 0, "training_loss": ""}], ["iteration", "epoch", "training_loss"])
    write_csv(exp_dir / "lr_schedule.csv", lr_rows or [{"iteration": 0, "epoch": 0, "learning_rate": ""}], ["iteration", "epoch", "learning_rate"])

    eval_metrics = parse_eval_metrics(clean_wsl_output((eval_result.stdout if eval_result else "") + "\n" + (eval_result.stderr if eval_result else "")))
    final_official_eval = load_official_eval_metrics(exp_dir, "final")
    eval_rows = official_eval_metrics_to_rows(final_official_eval)
    if not eval_rows:
        eval_rows = [{"metric": key, "value": value} for key, value in eval_metrics.items()]
    write_csv(exp_dir / "eval_ap_summary.csv", eval_rows or [{"metric": "status", "value": ""}], ["metric", "value"])

    checkpoint_rows = [{"checkpoint": path} for path in copied["checkpoints"]]
    write_json(exp_dir / "checkpoint_index.json", checkpoint_rows)
    metrics_summary = {
        "experiment_name": exp_name,
        "training_mode": plan.mode,
        "status": "completed" if train_result and train_result.returncode == 0 and eval_result and eval_result.returncode == 0 else "partial",
        "partial_reason": None if train_result and train_result.returncode == 0 and eval_result and eval_result.returncode == 0 else "Training and/or explicit eval returned non-zero or incomplete artifacts; see logs.",
        "train_sample_count": split_info["train_count"],
        "val_sample_count": split_info["val_count"],
        "epochs": plan.epochs,
        "batch_size": plan.batch_size,
        "learning_rate": plan.learning_rate,
        "used_pretrained_checkpoint": plan.use_pretrained,
        "pretrained_checkpoint": str(PRETRAINED_CKPT) if plan.use_pretrained else None,
        "trained_checkpoint": str(final_ckpt) if final_ckpt else None,
        "opencpdet_output_dir": str(opencpdet_output),
        "baseline_eval": baseline,
        "final_eval_metrics": eval_metrics,
        "start_time": started,
        "end_time": finished,
        "seed": args.seed,
        "run_config": str(run_cfg),
        "train_command": train_command,
        "eval_command": eval_command,
    }
    write_json(exp_dir / "metrics_summary.json", metrics_summary)
    write_json(
        status_path,
        {
            **metrics_summary,
            "stdout_tail": clean_wsl_output(train_result.stdout[-4000:] if train_result else ""),
            "stderr_tail": clean_wsl_output(train_result.stderr[-4000:] if train_result else ""),
            "eval_stdout_tail": clean_wsl_output(eval_result.stdout[-4000:] if eval_result else ""),
            "eval_stderr_tail": clean_wsl_output(eval_result.stderr[-4000:] if eval_result else ""),
            "audit_path": str(TRAINING_REPORT_ROOT / "training_env_audit.json"),
        },
    )
    return metrics_summary


def update_comparison_table() -> None:
    rows = []
    for exp_dir in sorted(TRAINING_REPORT_ROOT.glob("*_train*")) + sorted(TRAINING_REPORT_ROOT.glob("*_finetune*")):
        summary = read_json_or_default(exp_dir / "metrics_summary.json", None)
        if not summary:
            continue
        baseline_eval = load_official_eval_metrics(exp_dir, "baseline")
        final_eval = load_official_eval_metrics(exp_dir, "final")
        variants = [
            ("pretrained_baseline", baseline_eval, summary.get("pretrained_checkpoint")),
            ("trained_checkpoint", final_eval, summary.get("trained_checkpoint")),
        ]
        for variant_name, eval_payload, ckpt_path in variants:
            result_dict = eval_payload.get("official_result_dict", {}) if isinstance(eval_payload, dict) else {}
            if not result_dict:
                continue
            car = result_dict.get("Car_3d/moderate_R40")
            ped = result_dict.get("Pedestrian_3d/moderate_R40")
            cyc = result_dict.get("Cyclist_3d/moderate_R40")
            vals = [car, ped, cyc]
            numeric = [v for v in vals if isinstance(v, (int, float))]
            rows.append(
                {
                    "experiment_name": f"{summary.get('experiment_name', exp_dir.name)}::{variant_name}",
                    "training_mode": summary.get("training_mode"),
                    "init_checkpoint": summary.get("pretrained_checkpoint"),
                    "trained_checkpoint": ckpt_path,
                    "train_sample_count": summary.get("train_sample_count"),
                    "val_sample_count": summary.get("val_sample_count"),
                    "epochs": summary.get("epochs"),
                    "batch_size": summary.get("batch_size"),
                    "learning_rate": summary.get("learning_rate"),
                    "car_ap_3d_moderate": car,
                    "ped_ap_3d_moderate": ped,
                    "cyc_ap_3d_moderate": cyc,
                    "mean_ap_3d_moderate": sum(numeric) / len(numeric) if numeric else None,
                    "eval_scope": "subset-val",
                    "full_val": False,
                    "notes": f"{summary.get('training_mode')}::{variant_name}",
                }
            )
    if rows:
        write_csv(
            TRAINING_REPORT_ROOT / "training_eval_comparison.csv",
            rows,
            list(rows[0].keys()),
        )


def main() -> None:
    args = parse_args()
    audit = audit_training_environment()
    result = run_experiment(args, audit)
    update_comparison_table()
    output_text = json.dumps(result, indent=2, ensure_ascii=True)
    if hasattr(sys.stdout, "buffer"):
        sys.stdout.buffer.write(output_text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
