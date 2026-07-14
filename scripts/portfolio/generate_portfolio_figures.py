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
    if args.kitti_root:
        generate_qualitative(args.kitti_root, args.frame_id)


if __name__ == "__main__":
    main()
