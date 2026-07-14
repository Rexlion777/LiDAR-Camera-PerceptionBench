from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import ensure_dir, read_json_or_default, write_csv, write_json, write_markdown


REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"
FIG_ROOT = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures" / "training_finetune"
PLOT_ROOT = REPORT_ROOT / "plot_data"
ORIGIN_ROOT = REPORT_ROOT / "origin_plot_data"
README_PATH = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "README.md"
RESUME_PATH = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "resume_bullets.md"
INTERVIEW_PATH = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "interview_qa.md"


def latest_exp(prefix: str) -> Path | None:
    candidates = sorted(REPORT_ROOT.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_plot_csv(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    write_csv(PLOT_ROOT / f"{name}.csv", rows, fields)
    write_csv(ORIGIN_ROOT / f"{name}_origin.csv", rows, fields)


def save_fig(fig: plt.Figure, name: str) -> dict[str, str]:
    ensure_dir(FIG_ROOT)
    outputs = {}
    for ext in ("png", "svg", "pdf"):
        path = FIG_ROOT / f"{name}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        outputs[ext] = str(path)
    plt.close(fig)
    return outputs


def summary_card(title: str, lines: list[str]) -> plt.Figure:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.04, 0.94, title, fontsize=22, weight="bold", va="top")
    y = 0.86
    for line in lines:
        ax.text(0.05, y, line, fontsize=14, va="top")
        y -= 0.075
    return fig


def generate_figures(snapshot: dict[str, Any], split_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]], deployment_rows: list[dict[str, Any]], expanded_status: dict[str, Any]) -> dict[str, dict[str, str]]:
    ensure_dir(FIG_ROOT)
    ensure_dir(PLOT_ROOT)
    ensure_dir(ORIGIN_ROOT)
    outputs: dict[str, dict[str, str]] = {}

    primary_splits = [row for row in split_rows if row["split_name"] in {"train_1000", "val_200", "holdout_eval_500"}]
    class_rows = []
    for row in primary_splits:
        class_rows.append({"split_name": row["split_name"], "car_gt_count": float(row["car_gt_count"]), "ped_gt_count": float(row["ped_gt_count"]), "cyc_gt_count": float(row["cyc_gt_count"])})
    save_plot_csv("08_split_distribution_class", class_rows, ["split_name", "car_gt_count", "ped_gt_count", "cyc_gt_count"])
    fig, ax = plt.subplots(figsize=(12, 9))
    x = np.arange(len(class_rows))
    width = 0.24
    ax.bar(x - width, [r["car_gt_count"] for r in class_rows], width, label="Car")
    ax.bar(x, [r["ped_gt_count"] for r in class_rows], width, label="Pedestrian")
    ax.bar(x + width, [r["cyc_gt_count"] for r in class_rows], width, label="Cyclist")
    ax.set_xticks(x)
    ax.set_xticklabels([r["split_name"] for r in class_rows], rotation=10)
    ax.set_ylabel("GT Count")
    ax.set_title("Subset / Holdout Split Class Distribution")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    outputs["08_split_distribution_class"] = save_fig(fig, "08_split_distribution_class")

    range_rows = []
    for row in primary_splits:
        range_rows.append({"split_name": row["split_name"], "near_gt_count": float(row["near_gt_count"]), "mid_gt_count": float(row["mid_gt_count"]), "far_gt_count": float(row["far_gt_count"])})
    save_plot_csv("09_split_distribution_range", range_rows, ["split_name", "near_gt_count", "mid_gt_count", "far_gt_count"])
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.bar(x - width, [r["near_gt_count"] for r in range_rows], width, label="0-20m")
    ax.bar(x, [r["mid_gt_count"] for r in range_rows], width, label="20-40m")
    ax.bar(x + width, [r["far_gt_count"] for r in range_rows], width, label="40m+")
    ax.set_xticks(x)
    ax.set_xticklabels([r["split_name"] for r in range_rows], rotation=10)
    ax.set_ylabel("GT Count")
    ax.set_title("Subset / Holdout Split Range Distribution")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    outputs["09_split_distribution_range"] = save_fig(fig, "09_split_distribution_range")

    loss_path = latest_exp("expanded_finetune_1000") / "loss_curve.csv"
    loss_rows = load_csv(loss_path)
    save_plot_csv("10_expanded_loss_curve", loss_rows, list(loss_rows[0].keys()))
    fig, ax = plt.subplots(figsize=(12, 9))
    iterations = [int(row["iteration"]) for row in loss_rows if row["iteration"]]
    losses = [float(row["training_loss_avg"]) for row in loss_rows if row["training_loss_avg"] != ""]
    ax.plot(iterations, losses, linewidth=2.0)
    ax.set_title("Expanded Subset Fine-tuning Loss Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Training Loss")
    ax.grid(True, alpha=0.3)
    outputs["10_expanded_loss_curve"] = save_fig(fig, "10_expanded_loss_curve")

    ap_rows = []
    for row in holdout_rows:
        ap_rows.append(
            {
                "model_name": row["model_name"],
                "car_ap": float(row["car_ap_3d_moderate"]),
                "ped_ap": float(row["ped_ap_3d_moderate"]),
                "cyc_ap": float(row["cyc_ap_3d_moderate"]),
                "mean_ap": float(row["mean_ap_3d_moderate"]),
            }
        )
    save_plot_csv("11_holdout_ap_comparison", ap_rows, ["model_name", "car_ap", "ped_ap", "cyc_ap", "mean_ap"])
    fig, ax = plt.subplots(figsize=(12, 9))
    x = np.arange(4)
    width = 0.25
    labels = ["Car", "Pedestrian", "Cyclist", "Mean"]
    for idx, row in enumerate(ap_rows):
        ax.bar(x + (idx - 1) * width, [row["car_ap"], row["ped_ap"], row["cyc_ap"], row["mean_ap"]], width, label=row["model_name"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("3D AP_R40")
    ax.set_title("Holdout-500 AP Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    outputs["11_holdout_ap_comparison"] = save_fig(fig, "11_holdout_ap_comparison")

    pred_rows = []
    for row in deployment_rows:
        pred_rows.append({"model_name": row["model_name"], "total_boxes": float(row["total_boxes"]), "mean_boxes_per_frame": float(row["mean_boxes_per_frame"]), "mean_ap_3d_moderate": float(row["mean_ap_3d_moderate"])})
    save_plot_csv("12_holdout_prediction_count_comparison", pred_rows, ["model_name", "total_boxes", "mean_boxes_per_frame", "mean_ap_3d_moderate"])
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.bar([r["model_name"] for r in pred_rows], [r["mean_boxes_per_frame"] for r in pred_rows], color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel("Mean Boxes Per Frame")
    ax.set_title("Holdout-500 Prediction Count Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=10)
    outputs["12_holdout_prediction_count_comparison"] = save_fig(fig, "12_holdout_prediction_count_comparison")

    score_rows = []
    for row in deployment_rows:
        score_rows.append({"model_name": row["model_name"], "score_mean": float(row["score_mean"]), "score_p50": float(row["score_p50"]), "score_p95": float(row["score_p95"])})
    save_plot_csv("13_score_distribution_pretrained_vs_finetuned", score_rows, ["model_name", "score_mean", "score_p50", "score_p95"])
    fig, ax = plt.subplots(figsize=(12, 9))
    x = np.arange(len(score_rows))
    width = 0.25
    ax.bar(x - width, [r["score_mean"] for r in score_rows], width, label="score_mean")
    ax.bar(x, [r["score_p50"] for r in score_rows], width, label="score_p50")
    ax.bar(x + width, [r["score_p95"] for r in score_rows], width, label="score_p95")
    ax.set_xticks(x)
    ax.set_xticklabels([r["model_name"] for r in score_rows], rotation=10)
    ax.set_ylabel("Score")
    ax.set_title("Holdout Score Distribution Summary")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    outputs["13_score_distribution_pretrained_vs_finetuned"] = save_fig(fig, "13_score_distribution_pretrained_vs_finetuned")

    deployment_json = read_json_or_default(REPORT_ROOT / "expanded_deployment_diagnostics.json", {})
    range_dist_rows = []
    for row in deployment_json.get("rows", []):
        rng = row["range_hist_norm"]
        range_dist_rows.append({"model_name": row["model_name"], "near_ratio": rng[0], "mid_ratio": rng[1], "far_ratio": rng[2], "very_far_ratio": rng[3] if len(rng) > 3 else 0.0})
    save_plot_csv("14_range_distribution_pretrained_vs_finetuned", range_dist_rows, ["model_name", "near_ratio", "mid_ratio", "far_ratio", "very_far_ratio"])
    fig, ax = plt.subplots(figsize=(12, 9))
    bottoms = np.zeros(len(range_dist_rows))
    labels = [r["model_name"] for r in range_dist_rows]
    for key, label, color in [("near_ratio", "0-20m", "#4C78A8"), ("mid_ratio", "20-40m", "#F58518"), ("far_ratio", "40-60m", "#54A24B"), ("very_far_ratio", "60m+", "#B279A2")]:
        values = np.array([r[key] for r in range_dist_rows], dtype=float)
        ax.bar(labels, values, bottom=bottoms, label=label, color=color)
        bottoms += values
    ax.set_ylabel("Normalized Detection Range Ratio")
    ax.set_title("Holdout Range Distribution by Model")
    ax.legend()
    plt.xticks(rotation=10)
    outputs["14_range_distribution_pretrained_vs_finetuned"] = save_fig(fig, "14_range_distribution_pretrained_vs_finetuned")

    sample_rows = [
        {"experiment": "smoke_train", "train_samples": 100, "val_samples": 25, "epochs": 1},
        {"experiment": "subset_finetune", "train_samples": 200, "val_samples": 50, "epochs": 3},
        {"experiment": "expanded_finetune_1000", "train_samples": 1000, "val_samples": 200, "epochs": 3},
    ]
    save_plot_csv("15_training_sample_scale_summary", sample_rows, ["experiment", "train_samples", "val_samples", "epochs"])
    fig = summary_card(
        "Training Sample Scale Summary",
        [
            "smoke_train: 100 train / 25 val / 1 epoch",
            "subset_finetune: 200 train / 50 val / 3 epochs",
            "expanded_finetune_1000: 1000 train / 200 val / 3 epochs",
            "Primary comparison in this round uses fixed holdout_eval_500, not 50-frame subset-val.",
        ],
    )
    outputs["15_training_sample_scale_summary"] = save_fig(fig, "15_training_sample_scale_summary")

    fig = summary_card(
        "Expanded Pipeline Summary",
        [
            "expanded_splits -> expanded_finetune_1000 -> native tools/test.py eval",
            "same checkpoint -> external official eval on holdout_eval_500",
            "same holdout predictions -> deployment diagnostics (box / score / class / range / empty / invalid)",
            "Boundary: subset / holdout only, full_val=false",
        ],
    )
    outputs["16_expanded_pipeline_summary"] = save_fig(fig, "16_expanded_pipeline_summary")
    return outputs


def update_materials(report_path: Path, expanded_status: dict[str, Any], holdout_rows: list[dict[str, Any]]) -> None:
    holdout_lookup = {row["model_name"]: row for row in holdout_rows}
    expanded = holdout_lookup["expanded_finetune_1000_200_epoch3"]
    baseline = holdout_lookup["pretrained_baseline"]
    current = holdout_lookup["subset_finetune_200_50_epoch3"]

    readme = f"""# LiDAR 3D 检测模型部署验收与异常归因系统

## Current Scope

- OpenPCDet / PointPillars runtime, official eval, deployment acceptance and diagnostics
- TensorRT milestone remains **backbone/head-only**, not full TensorRT detector
- Added an expanded PointPillars subset training / fine-tuning loop with fixed holdout evaluation

## Expanded Training / Holdout Milestone

- Current snapshot: `reports/lidar_system_algorithm/training_finetune/current_status_snapshot.md`
- Expanded split report: `reports/lidar_system_algorithm/training_finetune/expanded_splits/split_distribution_report.md`
- Expanded fine-tune report: `reports/lidar_system_algorithm/training_finetune/expanded_training_finetune_report.md`
- Holdout comparison: `reports/lidar_system_algorithm/training_finetune/expanded_holdout_eval_comparison.csv`
- Deployment diagnostics: `reports/lidar_system_algorithm/training_finetune/expanded_deployment_diagnostics.csv`

## Actual Boundary

- `smoke_train`: pipeline verification only
- `subset_finetune`: 200 train / 50 val
- `expanded_finetune_1000`: 1000 train / 200 val, 3 epochs, pretrained init
- Main comparison uses `holdout_eval_500`; `full_val=false`

## Safe Claims

- Completed PointPillars 1000-sample subset fine-tuning with fixed 500-frame holdout evaluation
- Native OpenPCDet `tools/test.py` eval is repaired via shell-level CUDA/NVVM wrapper
- Holdout comparison and deployment diagnostics are based on real checkpoints and real prediction exports

## Forbidden Claims

- Do not claim full KITTI training
- Do not claim SOTA
- Do not call subset or holdout split full KITTI val
- Do not call backbone/head-only TensorRT a full TensorRT detector
"""
    README_PATH.write_text(readme, encoding="utf-8")

    resume = f"""# Resume Bullets

## Expanded Training Loop

### 稳健版
完成 PointPillars 1000-sample subset fine-tuning 与固定 500-frame holdout 评估，打通 KITTI 数据准备、训练/微调、checkpoint 管理、OpenPCDet 原生 tools/test.py 评估、external official AP 评估和部署诊断闭环，并基于真实日志对比微调前后 AP、预测框数量、score/range 分布与异常输出变化。

### 更强版
构建固定 train/val/holdout 划分，完成 PointPillars 1000/200 subset fine-tuning、500-frame holdout official AP 对比与部署诊断；结果显示 expanded fine-tune 在 holdout 上较 pretrained 基线呈现类别 trade-off，Pedestrian/Cyclist moderate 3D AP 提升而 Car moderate 下降，同时预测框数量上升，需要结合 false-positive 风险一起解释。

### 推荐一句话
完成 PointPillars 1000-sample subset fine-tuning 与固定 500-frame holdout 评估，打通 KITTI 数据准备、训练/微调、native OpenPCDet 评估、official AP 对比和部署诊断闭环，并基于真实 holdout 结果分析 AP、预测框数量与 score/range 分布变化。
"""
    RESUME_PATH.write_text(resume, encoding="utf-8")

    interview = f"""# Interview Q&A

## 你做过模型训练/微调吗？

做过，但我会明确说是 PointPillars 的 subset fine-tuning，不是 full KITTI training。现在这条链路已经从 100/25 smoke 和 200/50 subset 扩展到 1000/200 fine-tune，并且有固定 500-frame holdout 评估。

## 为什么要做固定 holdout？

50-frame subset-val 只能给初步信号，样本太小。固定 500-frame holdout 才能更稳地比较 pretrained、200/50 fine-tune 和 1000/200 fine-tune，不把小样本偶然性直接当结论。

## holdout 上的结果怎么解读？

这次 500-frame holdout 上，pretrained mean AP_R40 是 {float(baseline['mean_ap_3d_moderate']):.4f}，200/50 fine-tune 是 {float(current['mean_ap_3d_moderate']):.4f}，1000/200 expanded fine-tune 是 {float(expanded['mean_ap_3d_moderate']):.4f}。expanded 版本不是全类别同时上升，而是 Car moderate 下降、Pedestrian/Cyclist moderate 提升，所以我会把它解释成类别 trade-off，而不是简单说“fine-tune 变好了”。

## 预测框数量增加是不是性能提升？

不能直接这么写。holdout diagnostics 里 expanded 版本总框数比 pretrained 高，可能意味着 recall 增加，也可能意味着 false positive 风险增加，必须和 holdout AP、score 分布、class/range drift 一起看。

## native tools/test.py 评估现在是什么状态？

已经修通。WSL 里通过 shell-level CUDA/NVVM wrapper 解决了 libNVVM 动态库解析问题，所以 native `tools/test.py` eval 是 completed。

## 这能不能写成 full KITTI training？

不能。现在真实完成的是 subset fine-tuning + fixed holdout evaluation，full_val=false。
"""
    INTERVIEW_PATH.write_text(interview, encoding="utf-8")


def generate_report(snapshot: dict[str, Any], split_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]], deployment_json: dict[str, Any], expanded_status: dict[str, Any], figure_outputs: dict[str, dict[str, str]]) -> None:
    holdout_lookup = {row["model_name"]: row for row in holdout_rows}
    baseline = holdout_lookup["pretrained_baseline"]
    current = holdout_lookup["subset_finetune_200_50_epoch3"]
    expanded = holdout_lookup["expanded_finetune_1000_200_epoch3"]
    dep_lookup = {row["model_name"]: row for row in deployment_json["rows"]}
    expanded_dep = dep_lookup["expanded_finetune_1000_200_epoch3"]
    baseline_dep = dep_lookup["pretrained_baseline"]

    payload = {
        "status": "completed",
        "native_tools_test_eval_fixed": True,
        "full_val": False,
        "current_snapshot_path": str(REPORT_ROOT / "current_status_snapshot.json"),
        "split_summary_path": str(REPORT_ROOT / "expanded_splits" / "split_distribution_summary.csv"),
        "expanded_experiment_dir": str(latest_exp("expanded_finetune_1000")),
        "holdout_eval_comparison_path": str(REPORT_ROOT / "expanded_holdout_eval_comparison.csv"),
        "deployment_diagnostics_path": str(REPORT_ROOT / "expanded_deployment_diagnostics.csv"),
        "figures": figure_outputs,
        "results": {
            "holdout_500_mean_ap": {
                "pretrained": float(baseline["mean_ap_3d_moderate"]),
                "subset_200_50": float(current["mean_ap_3d_moderate"]),
                "expanded_1000_200": float(expanded["mean_ap_3d_moderate"]),
            },
            "expanded_vs_pretrained_box_delta": int(expanded_dep["total_boxes"] - baseline_dep["total_boxes"]),
        },
        "safe_claims": [
            "Expanded to a fixed 1000/200 subset fine-tuning setup with a common 500-frame holdout evaluation.",
            "Native OpenPCDet tools/test.py eval is fixed and completed.",
            "Main comparison in this round is holdout-based, not 50-frame subset-val based.",
        ],
        "forbidden_claims": [
            "Do not claim full KITTI training.",
            "Do not claim SOTA.",
            "Do not call 500-frame holdout full KITTI val.",
            "Do not interpret higher box count as direct performance improvement.",
        ],
    }
    write_json(REPORT_ROOT / "expanded_training_finetune_report.json", payload)

    lines = [
        "# Expanded PointPillars Training / Fine-tuning Report",
        "",
        "## 1. Why expand sample scope",
        "",
        "- 100/25 smoke 和 200/50 subset_finetune 只证明训练-评估链路能跑通，但不足以作为主要泛化结论。",
        "- 本轮把训练扩展到 1000/200，并引入固定 500-frame holdout，对 pretrained / current 200/50 / expanded 1000/200 做同一评估口径比较。",
        "",
        "## 2. Previous 200/50 limitation",
        "",
        "- 50-frame subset-val AP 只能作为初步信号。",
        "- 本轮报告把 500-frame holdout 作为主要对比依据。",
        "",
        "## 3. OpenPCDet native eval fixed",
        "",
        "- native `tools/test.py` eval 已通过 shell-level CUDA/NVVM wrapper 修复。",
        "- 但当前所有训练/holdout 结果仍明确标注 `full_val=false`。",
        "",
        "## 4. Split construction",
        "",
        "- train_1000 / val_200 / holdout_eval_500 为本轮主线。",
        "- train_2000 / val_500 / holdout_eval_1000 已生成，但未在本轮跑完更重的训练/评估。",
        "",
        "## 5. Split distribution",
        "",
        *[
            f"- {row['split_name']}: samples={row['sample_count']}, Car={row['car_gt_count']}, Ped={row['ped_gt_count']}, Cyc={row['cyc_gt_count']}, near={row['near_gt_count']}, mid={row['mid_gt_count']}, far={row['far_gt_count']}"
            for row in split_rows
            if row["split_name"] in {"train_1000", "val_200", "holdout_eval_500"}
        ],
        "",
        "## 6. Expanded fine-tuning settings",
        "",
        f"- Experiment dir: `{latest_exp('expanded_finetune_1000')}`",
        f"- epochs: `{expanded_status['epochs']}`",
        f"- batch size: `{expanded_status['batch_size']}`",
        f"- learning rate: `{expanded_status['learning_rate']}`",
        f"- pretrained checkpoint init: `{expanded_status['used_pretrained_checkpoint']}`",
        "",
        "## 7. Training logs",
        "",
        f"- train.log: `{latest_exp('expanded_finetune_1000') / 'train.log'}`",
        f"- loss curve csv: `{latest_exp('expanded_finetune_1000') / 'loss_curve.csv'}`",
        "- OpenPCDet `train.py` inline repeat-eval 仍会触发 PyTorch 2.6 `weights_only` 问题，但训练本身已完成，checkpoint 已保存，并已通过显式 native eval + external eval 补齐。",
        "",
        "## 8. Holdout eval comparison",
        "",
        f"- pretrained baseline mean AP_R40: `{float(baseline['mean_ap_3d_moderate']):.4f}`",
        f"- current 200/50 subset_finetune mean AP_R40: `{float(current['mean_ap_3d_moderate']):.4f}`",
        f"- expanded 1000/200 fine-tune mean AP_R40: `{float(expanded['mean_ap_3d_moderate']):.4f}`",
        f"- Car moderate: `{float(baseline['car_ap_3d_moderate']):.4f}` -> `{float(current['car_ap_3d_moderate']):.4f}` -> `{float(expanded['car_ap_3d_moderate']):.4f}`",
        f"- Pedestrian moderate: `{float(baseline['ped_ap_3d_moderate']):.4f}` -> `{float(current['ped_ap_3d_moderate']):.4f}` -> `{float(expanded['ped_ap_3d_moderate']):.4f}`",
        f"- Cyclist moderate: `{float(baseline['cyc_ap_3d_moderate']):.4f}` -> `{float(current['cyc_ap_3d_moderate']):.4f}` -> `{float(expanded['cyc_ap_3d_moderate']):.4f}`",
        "",
        "## 9. Deployment diagnostics on holdout",
        "",
        f"- pretrained total boxes: `{baseline_dep['total_boxes']}`",
        f"- current 200/50 total boxes: `{dep_lookup['subset_finetune_200_50_epoch3']['total_boxes']}`",
        f"- expanded 1000/200 total boxes: `{expanded_dep['total_boxes']}`",
        f"- expanded prediction_count_drift vs baseline: `{expanded_dep['prediction_count_drift']:.4f}`",
        f"- expanded score_distribution_drift vs baseline: `{expanded_dep['score_distribution_drift']:.4f}`",
        f"- expanded class_distribution_drift vs baseline: `{expanded_dep['class_distribution_drift']:.4f}`",
        f"- expanded range_distribution_drift vs baseline: `{expanded_dep['range_distribution_drift']:.4f}`",
        "",
        "## 10. What improved",
        "",
        "- 与 200/50 subset_finetune 相比，1000/200 expanded 版本在 holdout 上的 mean AP_R40 更高。",
        "- 与 pretrained 相比，expanded 版本在 Pedestrian/Cyclist moderate 3D AP_R40 更高。",
        "",
        "## 11. What did not improve",
        "",
        "- expanded 版本的 Car moderate 3D AP_R40 低于 pretrained 基线。",
        "- 因此本轮不能把 expanded fine-tune 简化为“整体更好”。",
        "",
        "## 12. Possible false-positive risk",
        "",
        f"- expanded vs pretrained 总框数增加 `{int(expanded_dep['total_boxes'] - baseline_dep['total_boxes'])}`。",
        "- 预测框数量增加可能意味着 recall 增加，但也可能增加 false positive 风险。",
        "- 本轮会把它和 holdout AP、score/class/range drift 一起解释，而不是直接写成性能提升。",
        "",
        "## 13. Limitations",
        "",
        "- 本轮仍然是 subset training / fixed holdout，不是 full KITTI training。",
        "- holdout_eval_1000 split 已生成，但本轮未运行 1000-frame holdout 评估以控制总耗时。",
        "- expanded_finetune_2000 只完成 split 规划，未在本轮执行训练。",
        "",
        "## 14. Safe claims",
        "",
        *[f"- {item}" for item in payload["safe_claims"]],
        "",
        "## 15. Forbidden claims",
        "",
        *[f"- {item}" for item in payload["forbidden_claims"]],
        "",
        "## 16. Resume bullet",
        "",
        "- 完成 PointPillars 1000-sample subset fine-tuning 与固定 500-frame holdout 评估，打通 KITTI 数据准备、训练/微调、native OpenPCDet 评估、official AP 对比和部署诊断闭环，并基于真实 holdout 结果分析 AP、预测框数量与 score/range 分布变化。",
        "",
        "## 17. Interview QA",
        "",
        "- 重点讲 fixed holdout、native eval 修复、类别 trade-off 和 false-positive risk，而不是把 fine-tune 讲成单向提升。",
    ]
    write_markdown(REPORT_ROOT / "expanded_training_finetune_report.md", "\n".join(lines))


def main() -> None:
    snapshot = read_json_or_default(REPORT_ROOT / "current_status_snapshot.json", {})
    split_rows = load_csv(REPORT_ROOT / "expanded_splits" / "split_distribution_summary.csv")
    holdout_rows = load_csv(REPORT_ROOT / "expanded_holdout_eval_comparison.csv")
    deployment_rows = load_csv(REPORT_ROOT / "expanded_deployment_diagnostics.csv")
    expanded_status = json.loads((latest_exp("expanded_finetune_1000") / "train_status.json").read_text(encoding="utf-8"))
    figure_outputs = generate_figures(snapshot, split_rows, holdout_rows, deployment_rows, expanded_status)
    deployment_json = read_json_or_default(REPORT_ROOT / "expanded_deployment_diagnostics.json", {})
    generate_report(snapshot, split_rows, holdout_rows, deployment_json, expanded_status, figure_outputs)
    update_materials(REPORT_ROOT / "expanded_training_finetune_report.md", expanded_status, holdout_rows)
    print(json.dumps({"status": "completed", "report": str(REPORT_ROOT / "expanded_training_finetune_report.md")}, indent=2))


if __name__ == "__main__":
    main()
