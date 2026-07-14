#!/usr/bin/env python3
"""Generate the public portfolio figures from machine-readable evidence.

The quantitative charts only require the repository evidence files.  The
qualitative LiDAR/camera panel additionally accepts a local KITTI training root
containing image_2, velodyne and calib.  Dataset files are never copied into the
repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence" / "raw"
OUTPUT = ROOT / "assets" / "portfolio"

INK = "#152238"
MUTED = "#66758B"
GRID = "#DCE4EE"
BLUE = "#2563EB"
BLUE_LIGHT = "#DBEAFE"
TEAL = "#0F766E"
TEAL_LIGHT = "#CCFBF1"
GOLD = "#D97706"
GOLD_LIGHT = "#FEF3C7"
ORANGE = "#EA580C"
PINK = "#BE185D"
BG = "#F6F8FC"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelcolor": MUTED,
            "axes.edgecolor": "#C8D2E0",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def clean_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / name, dpi=200, bbox_inches="tight", pad_inches=0.18, facecolor=fig.get_facecolor())
    plt.close(fig)


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.955, title, color=INK, fontsize=22, fontweight="bold", va="top")
    fig.text(0.055, 0.915, subtitle, color=MUTED, fontsize=11, va="top")


def generate_accuracy_deployment() -> None:
    summary = json.loads((ROOT / "evidence" / "summary.json").read_text(encoding="utf-8"))
    ap = summary["official_kitti_validation"]
    fig = plt.figure(figsize=(14, 7.6))
    add_header(
        fig,
        "Accuracy and deployment performance",
        "KITTI validation: 3,769 frames · latency values are measured at their declared module boundaries",
    )
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.97, bottom=0.10, top=0.84, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    classes = ["Car", "Pedestrian", "Cyclist"]
    values = [ap["car"], ap["pedestrian"], ap["cyclist"]]
    colors = [BLUE, TEAL, GOLD]
    bars = ax.barh(classes[::-1], values[::-1], color=colors[::-1], height=0.56)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Moderate 3D AP")
    ax.set_title("Official KITTI 3D AP", loc="left", color=INK, pad=14)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 1.0, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", color=INK, fontweight="bold")
    clean_axes(ax, "x")

    ax = fig.add_subplot(gs[0, 1])
    labels = ["BEV backbone + head", "Tracking association"]
    before = np.array([summary["tensorrt_core_latency_ms"]["pytorch"], summary["tracking_association_latency_ms"]["legacy"]])
    after = np.array([summary["tensorrt_core_latency_ms"]["tensorrt"], summary["tracking_association_latency_ms"]["optimized"]])
    speedup = before / after
    ypos = np.array([1, 0])
    for y, b, a, s in zip(ypos, before, after, speedup):
        ax.plot([a, b], [y, y], color="#A8B6C8", linewidth=5, solid_capstyle="round")
        ax.scatter([b], [y], s=150, color="#9AA8BA", zorder=3, edgecolor="white", linewidth=1.5)
        ax.scatter([a], [y], s=170, color=BLUE, zorder=4, edgecolor="white", linewidth=1.5)
        ax.text(b + 1.1, y, f"{b:.2f} ms", va="center", color=MUTED)
        ax.text(max(a - 0.8, 0), y + 0.16, f"{a:.3f} ms", ha="center", color=BLUE, fontweight="bold")
        ax.text((a + b) / 2, y - 0.22, f"{s:.1f}× faster", ha="center", color=INK, fontweight="bold")
    ax.set_yticks(ypos, labels)
    ax.set_xlim(0, 52)
    ax.set_ylim(-0.5, 1.45)
    ax.set_xlabel("Latency (ms, lower is better)")
    ax.set_title("Measured acceleration", loc="left", color=INK, pad=14)
    clean_axes(ax, "x")
    save(fig, "02_accuracy_and_deployment.png")


def generate_robustness_landscape() -> None:
    data = pd.read_csv(EVIDENCE / "per_setting_ap.csv")
    fig = plt.figure(figsize=(15, 7.8))
    add_header(
        fig,
        "Perception robustness across controlled perturbations",
        "Mean moderate 3D AP · 200-frame dense evaluation slice · three independent stress-test axes",
    )
    gs = fig.add_gridspec(1, 3, left=0.055, right=0.98, bottom=0.11, top=0.83, wspace=0.27)
    specs = [
        ("point_dropout", "Point dropout", "Dropped points (%)", lambda x: pd.to_numeric(x) * 100, BLUE),
        ("range_crop", "Detection range", "Maximum range (m)", lambda x: pd.to_numeric(x, errors="coerce"), TEAL),
        ("postprocess_score_threshold", "Score threshold", "Threshold", lambda x: pd.to_numeric(x, errors="coerce"), GOLD),
    ]
    for ax, (kind, title, xlabel, converter, color) in zip([fig.add_subplot(gs[0, i]) for i in range(3)], specs):
        subset = data[data.perturbation_type == kind].copy()
        subset["x"] = converter(subset.perturbation_value)
        subset = subset.dropna(subset=["x"]).sort_values("x")
        x = subset.x.to_numpy(float)
        y = subset.mean_ap_3d_moderate.to_numpy(float)
        ax.plot(x, y, color=color, linewidth=2.8, marker="o", markersize=5, markeredgecolor="white", markeredgewidth=1.1)
        ax.fill_between(x, y, 0, color=color, alpha=0.08)
        ax.axhline(32.194, color="#8492A6", linestyle="--", linewidth=1.1)
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] else min(x), 32.8, "baseline 32.19", color=MUTED, fontsize=9)
        ax.set_ylim(0, 36)
        ax.set_title(title, loc="left", color=INK, pad=12)
        ax.set_xlabel(xlabel)
        if kind == "point_dropout":
            ax.set_ylabel("Mean moderate 3D AP")
        clean_axes(ax)
    save(fig, "03_robustness_landscape.png")


def generate_calibration_sensitivity() -> None:
    data = pd.read_csv(EVIDENCE / "calibration_sync_robustness.csv")
    fig = plt.figure(figsize=(14, 7.4))
    add_header(
        fig,
        "Camera-LiDAR calibration sensitivity",
        "Mean and P95 reprojection displacement across 20 KITTI frames",
    )
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.97, bottom=0.11, top=0.83, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    yaw = data[data.experiment == "yaw"].groupby("perturbation", as_index=False).agg(
        mean=("avg_reprojection_shift_px", "mean"), p95=("p95_reprojection_shift_px", "mean")
    )
    ax.plot(yaw.perturbation, yaw.p95, color=BLUE, linewidth=1.8, linestyle="--", label="P95")
    ax.plot(yaw.perturbation, yaw["mean"], color=BLUE, linewidth=3, marker="o", markersize=4, label="Mean")
    ax.fill_between(yaw.perturbation, yaw["mean"], yaw.p95, color=BLUE, alpha=0.12)
    ax.set_title("Yaw perturbation", loc="left", color=INK, pad=12)
    ax.set_xlabel("Yaw error (degrees)")
    ax.set_ylabel("Reprojection shift (pixels)")
    ax.legend(frameon=False, loc="upper center", ncols=2)
    clean_axes(ax)

    ax = fig.add_subplot(gs[0, 1])
    tr = data[data.experiment == "translation"].groupby(["axis", "perturbation"], as_index=False).agg(
        shift=("avg_reprojection_shift_px", "mean")
    )
    styles = {"x": (TEAL, "o"), "y": (GOLD, "s"), "z": (PINK, "^")}
    for axis, group in tr.groupby("axis"):
        color, marker = styles[axis]
        ax.plot(group.perturbation, group["shift"], color=color, linewidth=2.6, marker=marker, markersize=5, label=f"{axis.upper()} axis")
    ax.set_title("Translation perturbation", loc="left", color=INK, pad=12)
    ax.set_xlabel("Translation error (m)")
    ax.set_ylabel("Reprojection shift (pixels)")
    ax.legend(frameon=False, loc="upper center", ncols=3)
    clean_axes(ax)
    save(fig, "04_calibration_sensitivity.png")


def generate_deployment_parity() -> None:
    ap = pd.read_csv(EVIDENCE / "per_setting_ap.csv")
    ap = ap[ap.perturbation_type == "deployment_precision"].set_index("perturbation_value")
    latency = pd.read_csv(EVIDENCE / "deployment_precision_latency.csv").set_index("runtime_variant")
    baseline = ap.loc["openpcdet_original"]
    classes = ["Car", "Pedestrian", "Cyclist"]
    columns = ["car_ap_3d_moderate", "ped_ap_3d_moderate", "cyc_ap_3d_moderate"]
    variants = [("wrapper_pytorch", "PyTorch wrapper", TEAL), ("trt_backbone_head_only", "TensorRT", BLUE)]

    fig = plt.figure(figsize=(14.5, 7.4))
    add_header(fig, "Deployment accuracy parity and latency", "200-frame paired evaluation · AP delta is measured against the OpenPCDet baseline")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.97, bottom=0.11, top=0.82, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(classes))
    offsets = [-0.12, 0.12]
    for (variant, label, color), offset in zip(variants, offsets):
        delta = np.array([ap.loc[variant, col] - baseline[col] for col in columns])
        ax.scatter(delta, y + offset, s=105, color=color, label=label, edgecolor="white", linewidth=1.3, zorder=3)
        for x, yy in zip(delta, y + offset):
            ax.text(x + (0.004 if x >= 0 else -0.004), yy, f"{x:+.3f}", ha="left" if x >= 0 else "right", va="center", color=color, fontsize=9.5, fontweight="bold")
    ax.axvline(0, color=INK, linewidth=1.2)
    ax.set_yticks(y, classes)
    ax.set_xlim(-0.075, 0.075)
    ax.set_xlabel("Moderate 3D AP delta")
    ax.set_title("Model-to-runtime AP parity", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="lower right")
    clean_axes(ax, "x")

    ax = fig.add_subplot(gs[0, 1])
    labels = ["OpenPCDet", "PyTorch wrapper", "TensorRT"]
    keys = ["openpcdet_original", "wrapper_pytorch", "trt_backbone_head_only"]
    x = np.arange(3)
    width = 0.34
    core = [latency.loc[k, "core_latency_ms"] for k in keys]
    online = [latency.loc[k, "online_latency_ms"] for k in keys]
    b1 = ax.bar(x - width / 2, core, width, label="Core", color=BLUE)
    b2 = ax.bar(x + width / 2, online, width, label="Online total", color=GOLD)
    ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=9, color=INK)
    ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=9, color=INK)
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Core and online latency", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="upper right")
    clean_axes(ax)
    save(fig, "05_deployment_parity.png")


def generate_latency_distribution() -> None:
    data = pd.read_csv(EVIDENCE / "tensorrt_backbone_head_only_latency_1000.csv")
    data = data.iloc[5:].copy()
    series = [("pytorch_core_ms", "PyTorch", GOLD), ("trt_core_ms", "TensorRT", BLUE)]
    fig = plt.figure(figsize=(14.5, 7.4))
    add_header(fig, "Matched-frame latency distribution", "995 KITTI frames after warm-up · the same frames are evaluated by both runtime variants")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.97, bottom=0.11, top=0.82, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    for col, label, color in series:
        values = np.sort(data[col].dropna().to_numpy())
        cdf = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, cdf, color=color, linewidth=2.8, label=label)
    ax.set_xlim(0, np.percentile(data.pytorch_core_ms, 99.5) * 1.08)
    ax.set_xlabel("Core latency (ms)")
    ax.set_ylabel("Cumulative share")
    ax.set_title("Empirical latency CDF", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="lower right")
    clean_axes(ax)

    ax = fig.add_subplot(gs[0, 1])
    stats = []
    for col, label, color in series:
        values = data[col].dropna().to_numpy()
        stats.append((label, np.percentile(values, 50), np.percentile(values, 95), color))
    y = np.arange(2)
    for yy, (label, p50, p95, color) in zip(y, stats):
        ax.plot([p50, p95], [yy, yy], color=color, linewidth=6, solid_capstyle="round")
        ax.scatter([p50, p95], [yy, yy], s=[120, 145], color=["white", color], edgecolor=color, linewidth=2, zorder=3)
        ax.text(p50, yy + 0.13, f"P50 {p50:.2f}", ha="center", color=MUTED, fontsize=9.5)
        ax.text(p95, yy - 0.13, f"P95 {p95:.2f}", ha="center", color=color, fontweight="bold")
    ax.set_yticks(y, [s[0] for s in stats])
    ax.set_ylim(-0.35, 1.35)
    ax.set_xlabel("Core latency (ms)")
    ax.set_title("P50–P95 latency interval", loc="left", color=INK, pad=13)
    clean_axes(ax, "x")
    save(fig, "06_latency_distribution.png")


def generate_per_frame_runtime() -> None:
    data = pd.read_csv(EVIDENCE / "tensorrt_backbone_head_only_latency_1000.csv").iloc[5:].copy()
    speedup = data.pytorch_core_ms / data.trt_core_ms
    fig = plt.figure(figsize=(14.5, 7.4))
    add_header(fig, "Per-frame runtime behavior", "995 matched KITTI frames · warm-up samples removed")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.97, bottom=0.11, top=0.82, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    limit = max(np.percentile(data.pytorch_core_ms, 99), np.percentile(data.trt_core_ms, 99))
    ax.scatter(data.pytorch_core_ms, data.trt_core_ms, s=17, color=BLUE, alpha=0.38, edgecolors="none")
    ax.plot([0, limit], [0, limit], color=INK, linewidth=1.2, linestyle="--", label="equal latency")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, min(limit, np.percentile(data.trt_core_ms, 99.8) * 1.3))
    ax.set_xlabel("PyTorch core latency (ms)")
    ax.set_ylabel("TensorRT core latency (ms)")
    ax.set_title("Paired-frame latency", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="upper left")
    clean_axes(ax)

    ax = fig.add_subplot(gs[0, 1])
    clipped = speedup.clip(upper=np.percentile(speedup, 99))
    ax.hist(clipped, bins=32, color=TEAL, alpha=0.88, edgecolor="white", linewidth=0.6)
    median = float(speedup.median())
    ax.axvline(median, color=INK, linestyle="--", linewidth=1.5)
    ax.text(median, ax.get_ylim()[1] * 0.92, f"median {median:.2f}×", ha="center", color=INK, fontweight="bold")
    ax.set_xlabel("Per-frame speedup")
    ax.set_ylabel("Frame count")
    ax.set_title("Speedup distribution", loc="left", color=INK, pad=13)
    clean_axes(ax)
    save(fig, "07_per_frame_runtime.png")


def generate_stage_profile() -> None:
    data = pd.read_csv(EVIDENCE / "online_latency_profile.csv")
    keep = ["data_load_ms", "calibration_parse_ms", "point_preprocess_ms", "voxelization_or_pillarization_ms", "model_forward_ms", "nms_ms", "postprocess_ms", "tracking_ms"]
    data = data[data.stage.isin(keep)].copy().sort_values("mean_ms")
    labels = data.stage.str.replace("_ms", "", regex=False).str.replace("_", " ", regex=False)
    fig, ax = plt.subplots(figsize=(13.8, 7.8))
    add_header(fig, "Online pipeline stage profile", "20-frame profile · mean and P95 latency by system stage")
    fig.subplots_adjust(left=0.25, right=0.95, bottom=0.11, top=0.82)
    y = np.arange(len(data))
    ax.barh(y, data.p95_ms, color=BLUE_LIGHT, edgecolor=BLUE, height=0.58, label="P95")
    ax.barh(y, data.mean_ms, color=BLUE, height=0.34, label="Mean")
    for yy, value in zip(y, data.mean_ms):
        ax.text(value + 0.15, yy, f"{value:.2f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Stage-level mean and tail latency", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="lower right")
    clean_axes(ax, "x")
    save(fig, "08_stage_profile.png")


def generate_voxelization_ablation() -> None:
    data = pd.read_csv(EVIDENCE / "voxelization_ablation.csv")
    fig = plt.figure(figsize=(14.5, 7.4))
    add_header(fig, "Voxelization resource trade-off", "20 KITTI frames per setting · pillar count, memory footprint, and preprocessing tail latency")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.97, bottom=0.11, top=0.82, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(data))
    bars = ax.bar(x, data.pillar_count_mean / 1000, color=TEAL, width=0.58)
    ax.bar_label(bars, labels=[f"{v/1000:.1f}k" for v in data.pillar_count_mean], padding=3, color=INK, fontsize=9)
    ax.set_xticks(x, [f"{v:.2f} m" for v in data.pillar_size])
    ax.set_ylabel("Mean pillar count (thousands)")
    ax.set_title("Pillar density", loc="left", color=INK, pad=13)
    clean_axes(ax)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(data.pillar_size, data.preprocess_p95_ms, color=GOLD, marker="o", linewidth=2.8, label="Preprocess P95 (ms)")
    ax.set_xlabel("Pillar size (m)")
    ax.set_ylabel("Preprocess P95 (ms)", color=GOLD)
    ax.tick_params(axis="y", colors=GOLD)
    ax2 = ax.twinx()
    ax2.plot(data.pillar_size, data.memory_bytes_mean / 1024**2, color=BLUE, marker="s", linewidth=2.8, label="Memory (MiB)")
    ax2.set_ylabel("Mean tensor memory (MiB)", color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE)
    ax.set_title("Latency and memory", loc="left", color=INK, pad=13)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    lines = ax.lines + ax2.lines
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper center")
    save(fig, "09_voxelization_ablation.png")


def generate_class_robustness() -> None:
    data = pd.read_csv(EVIDENCE / "per_setting_failure_by_class.csv")
    data = data[data.perturbation_type == "point_dropout"].copy()
    data["dropout"] = data.perturbation_value.astype(float) * 100
    palette = {"Car": BLUE, "Pedestrian": GOLD, "Cyclist": TEAL}

    fig = plt.figure(figsize=(14.5, 7.4))
    add_header(fig, "Class-level resilience under LiDAR sparsity", "200-frame stress test per setting · recall is measured after matching predictions to KITTI ground truth")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.97, bottom=0.11, top=0.82, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    for class_name, group in data.groupby("class_name"):
        group = group.sort_values("dropout")
        ax.plot(group.dropout, group.recall, marker="o", linewidth=2.6, markersize=5, color=palette[class_name], label=class_name)
    ax.set_xlabel("Random point dropout (%)")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 0.9)
    ax.set_title("Recall degradation curve", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="lower left")
    clean_axes(ax)

    ax = fig.add_subplot(gs[0, 1])
    base = data[data.dropout == 0].set_index("class_name").recall
    severe = data[data.dropout == 80].set_index("class_name").recall
    classes = ["Car", "Pedestrian", "Cyclist"]
    decline = [(base[c] - severe[c]) * 100 for c in classes]
    bars = ax.bar(classes, decline, color=[palette[c] for c in classes], width=0.58)
    ax.bar_label(bars, labels=[f"{v:.1f} pp" for v in decline], padding=4, color=INK, fontweight="bold")
    ax.set_ylabel("Recall loss at 80% dropout (pp)")
    ax.set_title("Class-specific failure sensitivity", loc="left", color=INK, pad=13)
    clean_axes(ax)
    save(fig, "10_class_robustness.png")


def generate_range_robustness() -> None:
    data = pd.read_csv(EVIDENCE / "per_setting_failure_by_range.csv")
    data = data[data.perturbation_type == "point_dropout"].copy()
    data["dropout"] = data.perturbation_value.astype(float) * 100
    ranges = ["0-20m", "20-40m", "40-60m", "60m+"]
    dropouts = sorted(data.dropout.unique())
    matrix = data.pivot(index="range_bin", columns="dropout", values="recall").reindex(ranges)[dropouts]

    fig, ax = plt.subplots(figsize=(14.5, 7.4))
    add_header(fig, "Range-aware LiDAR failure map", "200-frame stress test per setting · each cell reports matched-detection recall")
    fig.subplots_adjust(left=0.13, right=0.94, bottom=0.16, top=0.81)
    image = ax.imshow(matrix.to_numpy(), cmap="YlGnBu", vmin=0, vmax=0.95, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9, color="white" if value > 0.57 else INK, fontweight="bold" if j in (0, matrix.shape[1] - 1) else "normal")
    ax.set_xticks(np.arange(len(dropouts)), [f"{v:.0f}%" for v in dropouts])
    ax.set_yticks(np.arange(len(ranges)), ranges)
    ax.set_xlabel("Random point dropout")
    ax.set_ylabel("Object distance")
    ax.set_title("Recall by distance band and point availability", loc="left", color=INK, pad=13)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label("Recall")
    save(fig, "11_range_robustness.png")


def generate_prediction_health() -> None:
    data = pd.read_csv(EVIDENCE / "per_setting_prediction_health.csv")
    data = data[data.perturbation_type == "point_dropout"].copy()
    data["dropout"] = data.perturbation_value.astype(float) * 100
    data = data.sort_values("dropout")

    fig = plt.figure(figsize=(14.5, 7.4))
    add_header(fig, "Prediction-health observability", "200-frame stress test per setting · count, confidence, and distribution drift are monitored together")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.97, bottom=0.11, top=0.82, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    for col, label, color in [("car_box_count", "Car", BLUE), ("ped_box_count", "Pedestrian", GOLD), ("cyc_box_count", "Cyclist", TEAL)]:
        ax.plot(data.dropout, data[col], marker="o", linewidth=2.5, color=color, label=label)
    ax.set_xlabel("Random point dropout (%)")
    ax.set_ylabel("Predicted boxes")
    ax.set_title("Output population by class", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="upper right")
    clean_axes(ax)

    ax = fig.add_subplot(gs[0, 1])
    drifts = [
        ("prediction_count_drift", "Count", BLUE),
        ("score_distribution_drift", "Score", GOLD),
        ("class_distribution_drift", "Class mix", TEAL),
        ("range_distribution_drift", "Range mix", PINK),
    ]
    for col, label, color in drifts:
        values = data[col].astype(float)
        normalized = values / max(values.max(), 1e-12)
        ax.plot(data.dropout, normalized, marker="o", linewidth=2.4, color=color, label=label)
    ax.set_xlabel("Random point dropout (%)")
    ax.set_ylabel("Normalized drift (0–1 within metric)")
    ax.set_title("Multi-signal health drift", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="upper left", ncols=2)
    clean_axes(ax)
    save(fig, "12_prediction_health.png")


def generate_tracking_latency() -> None:
    data = pd.read_csv(EVIDENCE / "tracking_optimized_summary.csv")
    fig = plt.figure(figsize=(14.5, 7.4))
    add_header(fig, "Vectorized tracking association", "50-frame replay · legacy and optimized association receive the same detections")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.97, bottom=0.11, top=0.82, wspace=0.29)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(data.frame_id, data.legacy_association_latency_ms, color=GOLD, linewidth=2.3, label="Legacy")
    ax.plot(data.frame_id, data.association_latency_ms, color=BLUE, linewidth=2.3, label="Vectorized")
    ax.set_yscale("log")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Association latency (ms, log scale)")
    ax.set_title("Per-frame association latency", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="upper left")
    clean_axes(ax)

    ax = fig.add_subplot(gs[0, 1])
    sizes = data.association_matrix_size.clip(lower=1)
    scatter = ax.scatter(sizes, data.association_latency_ms, c=data.gated_pair_count, cmap="viridis", s=46, alpha=0.78, edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Association matrix cells")
    ax.set_ylabel("Vectorized latency (ms)")
    ax.set_title("Cost versus matching workload", loc="left", color=INK, pad=13)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Pairs retained by gating")
    clean_axes(ax)
    save(fig, "13_tracking_latency.png")


def generate_tracking_lifecycle() -> None:
    data = pd.read_csv(EVIDENCE / "tracking_optimized_summary.csv")
    fig = plt.figure(figsize=(14.5, 7.4))
    add_header(fig, "Track lifecycle and state consistency", "50-frame replay · track population, creation, and expiration are audited frame by frame")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.97, bottom=0.11, top=0.82, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(data.frame_id, data.detection_count, color=BLUE_LIGHT, alpha=0.85, label="Detections")
    ax.plot(data.frame_id, data.detection_count, color=BLUE, linewidth=2.2)
    ax.plot(data.frame_id, data.visible_track_count, color=TEAL, linewidth=2.2, label="Visible tracks")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Object count")
    ax.set_title("Detection-to-track population", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="upper right")
    clean_axes(ax)

    ax = fig.add_subplot(gs[0, 1])
    width = 0.8
    ax.bar(data.frame_id, data.spawned_tracks, width=width, color=TEAL, label="Spawned")
    ax.bar(data.frame_id, -data.expired_tracks, width=width, color=GOLD, label="Expired")
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Track lifecycle events")
    ax.set_title("Creation and retirement events", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="upper right")
    clean_axes(ax)
    save(fig, "14_tracking_lifecycle.png")


def generate_inference_population() -> None:
    data = pd.read_csv(EVIDENCE / "pointpillars_inference_results.csv")
    classes = ["Car", "Pedestrian", "Cyclist"]
    colors = [BLUE, GOLD, TEAL]
    fig = plt.figure(figsize=(14.5, 7.4))
    add_header(fig, "PointPillars output population", f"{len(data):,} decoded detections · score and spatial range before downstream tracking")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.97, bottom=0.11, top=0.82, wspace=0.29)
    ax = fig.add_subplot(gs[0, 0])
    grouped = [data.loc[data.class_name == c, "score"].to_numpy() for c in classes]
    violin = ax.violinplot(grouped, showmeans=True, showmedians=True, widths=0.75)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(0.72)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        violin[key].set_color(INK)
        violin[key].set_linewidth(1.1)
    ax.set_xticks(np.arange(1, 4), classes)
    ax.set_ylabel("Confidence score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Confidence distribution", loc="left", color=INK, pad=13)
    clean_axes(ax)

    ax = fig.add_subplot(gs[0, 1])
    for class_name, color in zip(classes, colors):
        subset = data[data.class_name == class_name]
        distance = np.hypot(subset.center_x, subset.center_y)
        ax.hist(distance, bins=np.arange(0, 81, 5), histtype="step", linewidth=2.4, color=color, label=f"{class_name} (n={len(subset)})")
    ax.set_xlabel("Planar range (m)")
    ax.set_ylabel("Detection count")
    ax.set_title("Spatial range distribution", loc="left", color=INK, pad=13)
    ax.legend(frameon=False, loc="upper right")
    clean_axes(ax)
    save(fig, "15_inference_population.png")


def generate_evidence_coverage() -> None:
    csv_files = sorted(EVIDENCE.glob("*.csv"))
    row_counts = {path.stem: len(pd.read_csv(path)) for path in csv_files}
    categories = {
        "Deployment & latency": [name for name in row_counts if "tensorrt" in name or "latency" in name or "deployment" in name],
        "Robustness & health": [name for name in row_counts if "per_setting" in name or "health" in name],
        "Geometry & calibration": [name for name in row_counts if "calibration" in name],
        "Inference & tracking": [name for name in row_counts if "inference" in name or "tracking" in name or "dbscan" in name],
        "Resource ablation": [name for name in row_counts if "voxelization" in name],
    }
    category_rows = {label: sum(row_counts[name] for name in names) for label, names in categories.items()}
    category_files = {label: len(names) for label, names in categories.items()}

    fig = plt.figure(figsize=(14.5, 7.4))
    add_header(fig, "Machine-readable evidence coverage", f"{len(csv_files)} CSV tables · {sum(row_counts.values()):,} records · generated charts remain traceable to repository evidence")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.97, bottom=0.11, top=0.82, wspace=0.38)
    labels = list(category_rows)
    y = np.arange(len(labels))
    palette = [BLUE, TEAL, GOLD, PINK, ORANGE]
    ax = fig.add_subplot(gs[0, 0])
    values = np.array([category_rows[label] for label in labels])
    bars = ax.barh(y, values, color=palette, height=0.58)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=4, color=INK, fontsize=9.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("CSV records")
    ax.set_xscale("log")
    ax.set_xlim(1, max(values) * 2.1)
    ax.set_title("Evidence volume by subsystem", loc="left", color=INK, pad=13)
    clean_axes(ax, "x")

    ax = fig.add_subplot(gs[0, 1])
    files = np.array([category_files[label] for label in labels])
    wedges, _ = ax.pie(files, startangle=90, colors=palette, wedgeprops={"width": 0.34, "edgecolor": "white"})
    ax.text(0, 0.08, f"{len(csv_files)}", ha="center", va="center", fontsize=30, color=INK, fontweight="bold")
    ax.text(0, -0.12, "evidence tables", ha="center", va="center", fontsize=10, color=MUTED)
    ax.legend(wedges, [f"{label} · {category_files[label]}" for label in labels], frameon=False, loc="center left", bbox_to_anchor=(0.93, 0.5), fontsize=9.5)
    ax.set_title("Table coverage", loc="left", color=INK, pad=13)
    save(fig, "16_evidence_coverage.png")


def parse_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        values[key] = np.fromstring(raw, sep=" ")
    p2 = values["P2"].reshape(3, 4)
    r0 = np.eye(4)
    r0[:3, :3] = values["R0_rect"].reshape(3, 3)
    tr = np.eye(4)
    tr[:3, :4] = values["Tr_velo_to_cam"].reshape(3, 4)
    return p2, r0, tr


def box_corners(row: pd.Series) -> np.ndarray:
    dx, dy, dz = row.size_x, row.size_y, row.size_z
    corners = np.array(
        [[x, y, z] for z in (-dz / 2, dz / 2) for y in (-dy / 2, dy / 2) for x in (-dx / 2, dx / 2)],
        dtype=float,
    )
    c, s = np.cos(row.yaw), np.sin(row.yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return corners @ rotation.T + np.array([row.center_x, row.center_y, row.center_z])


def project_points(points: np.ndarray, p2: np.ndarray, r0: np.ndarray, tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hom = np.c_[points, np.ones(len(points))]
    cam = (r0 @ tr @ hom.T).T
    uvw = (p2 @ cam.T).T
    valid = uvw[:, 2] > 0.1
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)
    return uv, valid


def generate_qualitative(kitti_root: Path, frame_id: str = "000005") -> None:
    from matplotlib.image import imread

    image = imread(kitti_root / "image_2" / f"{frame_id}.png")
    points = np.fromfile(kitti_root / "velodyne" / f"{frame_id}.bin", dtype=np.float32).reshape(-1, 4)
    p2, r0, tr = parse_calibration(kitti_root / "calib" / f"{frame_id}.txt")
    detections = pd.read_csv(EVIDENCE / "pointpillars_inference_results.csv")
    frame_numeric = int(frame_id)
    boxes = detections[detections.frame_id.astype(int) == frame_numeric].copy()
    boxes = boxes[boxes.score >= 0.20].sort_values("score", ascending=False)
    colors = {"Car": BLUE, "Pedestrian": GOLD, "Cyclist": TEAL}
    edges = [(0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7), (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7)]

    fig = plt.figure(figsize=(16, 9), facecolor="#071323")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.12], width_ratios=[1.2, 1.2, 0.76], hspace=0.055, wspace=0.045)
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(image)
    for _, row in boxes.iterrows():
        uv, valid = project_points(box_corners(row), p2, r0, tr)
        if not valid.all():
            continue
        for i, j in edges:
            ax_img.plot(uv[[i, j], 0], uv[[i, j], 1], color=colors.get(row.class_name, "white"), linewidth=2.2)
    ax_img.set_xlim(0, image.shape[1])
    ax_img.set_ylim(image.shape[0], 0)
    ax_img.axis("off")
    ax_img.text(0.018, 0.93, "CAMERA–LiDAR GEOMETRY", transform=ax_img.transAxes, color="white", fontsize=18, fontweight="bold")
    ax_img.text(0.018, 0.855, "3D detections projected through the KITTI calibration chain", transform=ax_img.transAxes, color="#D7E2F1", fontsize=11)

    ax_bev = fig.add_subplot(gs[1, :2], facecolor="#071323")
    mask = (points[:, 0] > 0) & (points[:, 0] < 70) & (np.abs(points[:, 1]) < 40) & (points[:, 2] > -2.5) & (points[:, 2] < 2.0)
    pts = points[mask][::2]
    ax_bev.scatter(pts[:, 1], pts[:, 0], s=0.16, color="#92A8C2", alpha=0.55, rasterized=True)
    for _, row in boxes.iterrows():
        corners = box_corners(row)[:4, :2]
        order = [0, 1, 3, 2]
        poly = np.c_[corners[order, 1], corners[order, 0]]
        ax_bev.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=colors.get(row.class_name, "white"), linewidth=2.2))
    ax_bev.set_xlim(-40, 40)
    ax_bev.set_ylim(0, 70)
    ax_bev.set_aspect("equal")
    ax_bev.set_xlabel("Lateral distance (m)", color="#9FB0C5")
    ax_bev.set_ylabel("Forward distance (m)", color="#9FB0C5")
    ax_bev.tick_params(colors="#8298B2")
    ax_bev.spines[:].set_color("#29405D")
    ax_bev.grid(color="#19304B", linewidth=0.7)
    ax_bev.text(0.02, 0.965, "BIRD'S-EYE-VIEW PERCEPTION", transform=ax_bev.transAxes, color="white", fontsize=13, fontweight="bold", va="top")

    summary = json.loads((ROOT / "evidence" / "summary.json").read_text(encoding="utf-8"))
    ax_info = fig.add_subplot(gs[1, 2], facecolor="#0B1B30")
    ax_info.axis("off")
    ax_info.text(0.08, 0.92, "SYSTEM SNAPSHOT", color="#82A7D6", fontsize=12, fontweight="bold")
    cards = [
        ("3,769", "validation frames"),
        ("107", "stress-test settings"),
        ("1.86×", "TensorRT core speedup"),
        ("56.6×", "tracking association speedup"),
    ]
    for idx, (value, label) in enumerate(cards):
        y = 0.80 - idx * 0.19
        patch = FancyBboxPatch((0.07, y - 0.08), 0.86, 0.145, boxstyle="round,pad=0.012,rounding_size=0.025", facecolor="#102943", edgecolor="#244866", linewidth=1.0, transform=ax_info.transAxes)
        ax_info.add_patch(patch)
        ax_info.text(0.12, y + 0.005, value, color="#5DDAFF", fontsize=23, fontweight="bold", transform=ax_info.transAxes)
        ax_info.text(0.12, y - 0.05, label, color="#B8C9DB", fontsize=10.5, transform=ax_info.transAxes)
    ax_info.text(0.08, 0.025, "PointPillars · OpenPCDet · KITTI  |  Calibration · Robustness · TensorRT", color="#728CA8", fontsize=8.5, transform=ax_info.transAxes)
    save(fig, "01_system_overview.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", type=Path, help="KITTI training root containing image_2, velodyne and calib")
    parser.add_argument("--frame-id", default="000005")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    generate_accuracy_deployment()
    generate_robustness_landscape()
    generate_calibration_sensitivity()
    generate_deployment_parity()
    generate_latency_distribution()
    generate_per_frame_runtime()
    generate_stage_profile()
    generate_voxelization_ablation()
    generate_class_robustness()
    generate_range_robustness()
    generate_prediction_health()
    generate_tracking_latency()
    generate_tracking_lifecycle()
    generate_inference_population()
    generate_evidence_coverage()
    if args.kitti_root:
        generate_qualitative(args.kitti_root, args.frame_id)


if __name__ == "__main__":
    main()
