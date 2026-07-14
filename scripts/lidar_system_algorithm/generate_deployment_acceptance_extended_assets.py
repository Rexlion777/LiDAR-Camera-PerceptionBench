from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.high_res_plot_utils import (
    apply_axis_style,
    figure_from_pixels,
    note_figure,
    save_contact_sheet,
    save_figure_triplet,
    write_plot_csv,
    write_plot_metadata,
)
from runtime.lidar_system_algorithm.report_schema import ensure_dir, read_json_or_default, write_json, write_markdown


FORBIDDEN = [
    "new 3D detection model",
    "full TensorRT detector",
    "full-val if only 1000-frame slice",
    "end-to-end latency when referring to core latency",
    "yaw/time proxy described as detector AP",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the extended dense deployment-acceptance assets.")
    parser.add_argument("--input-dir", default="reports/lidar_system_algorithm/deployment_acceptance")
    parser.add_argument("--figures-dir", default="projects/lidar_system_algorithm/figures/deployment_acceptance_dense")
    parser.add_argument("--ppt-dir", default="projects/lidar_system_algorithm/figures/deployment_acceptance_dense_ppt_panels")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value, default: float | None = None) -> float | None:
    if value in (None, "", "None"):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _write_bundle(
    figure_name: str,
    fig,
    figures_dir: Path,
    plot_data_dir: Path,
    origin_dir: Path,
    metadata_dir: Path,
    rows: list[dict] | None,
    fieldnames: list[str] | None,
    data_level: str,
    point_count: int,
    x_axis: str | None,
    y_axis: str | None,
    color_by: str | None,
    units: str | None,
    source_report: str,
    safe_caption: str,
    limitations: str,
    skipped: bool = False,
    skipped_reason: str | None = None,
) -> Path:
    base = figures_dir / figure_name
    saved = save_figure_triplet(fig, base, dpi=600)
    source_csv = None
    if rows is not None and fieldnames is not None:
        csv_path = plot_data_dir / f"{figure_name}.csv"
        source_csv, _ = write_plot_csv(csv_path, rows, fieldnames, origin_dir)
    write_plot_metadata(
        metadata_dir / f"{figure_name}.json",
        {
            "figure_name": figure_name,
            "data_level": data_level,
            "point_count": point_count,
            "figure_path_png": saved["png"],
            "figure_path_svg": saved["svg"],
            "figure_path_pdf": saved["pdf"],
            "source_csv": str(source_csv) if source_csv else None,
            "source_report": source_report,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "color_by": color_by,
            "units": units,
            "plotted_columns": fieldnames or [],
            "skipped": skipped,
            "skipped_reason": skipped_reason,
            "safe_caption": safe_caption,
            "limitations": limitations,
            "forbidden_claims": FORBIDDEN,
        },
    )
    return Path(saved["png"])


def _summary_card(
    figure_name: str,
    title: str,
    lines: list[str],
    figures_dir: Path,
    metadata_dir: Path,
    source_report: str,
    skipped: bool = False,
    skipped_reason: str | None = None,
) -> Path:
    saved = note_figure(figures_dir / figure_name, title, lines, 3840, 2160, dpi=600)
    write_plot_metadata(
        metadata_dir / f"{figure_name}.json",
        {
            "figure_name": figure_name,
            "data_level": "skipped" if skipped else "summary",
            "point_count": 0,
            "figure_path_png": saved["png"],
            "figure_path_svg": saved["svg"],
            "figure_path_pdf": saved["pdf"],
            "source_csv": None,
            "source_report": source_report,
            "x_axis": None,
            "y_axis": None,
            "color_by": None,
            "units": None,
            "plotted_columns": [],
            "skipped": skipped,
            "skipped_reason": skipped_reason,
            "safe_caption": title,
            "limitations": "Summary or skipped-note card.",
            "forbidden_claims": FORBIDDEN,
        },
    )
    return Path(saved["png"])


def _scatter_by_category(ax, rows: list[dict], x_key: str, y_key: str, cat_key: str, title: str, xlabel: str, ylabel: str) -> None:
    palette = ["#1d3557", "#2a9d8f", "#e76f51", "#3a86ff", "#6a4c93", "#d62828", "#264653", "#f4a261"]
    categories = sorted({str(row[cat_key]) for row in rows if row.get(x_key) not in (None, "") and row.get(y_key) not in (None, "")})
    for idx, cat in enumerate(categories):
        subset = [row for row in rows if str(row.get(cat_key)) == cat and row.get(x_key) not in (None, "") and row.get(y_key) not in (None, "")]
        if not subset:
            continue
        xs = np.asarray([float(row[x_key]) for row in subset], dtype=np.float64)
        ys = np.asarray([float(row[y_key]) for row in subset], dtype=np.float64)
        ax.scatter(xs, ys, s=10, alpha=0.22, color=palette[idx % len(palette)], label=cat)
    apply_axis_style(ax, title, xlabel, ylabel)


def _copy_or_alias_dense_csv(input_dir: Path) -> None:
    dense_dir = input_dir / "dense_diagnostics"
    yaw_src = dense_dir / "per_object_yaw_projection_shift_dense.csv"
    yaw_dst = dense_dir / "per_object_projection_shift_dense.csv"
    if yaw_src.exists():
        shutil.copyfile(yaw_src, yaw_dst)

    time_src = dense_dir / "per_frame_time_offset_dense.csv"
    if time_src.exists():
        rows = _read_csv(time_src)
        if rows and "temporal_consistency_error" not in rows[0]:
            for row in rows:
                row["temporal_consistency_error"] = row.get("center_drift_m")
            with time_src.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)


def _panel(output_base: Path, title: str, takeaway: str, image_paths: list[Path]) -> None:
    width_px, height_px = 3840, 2160
    panel = Image.new("RGB", (width_px, height_px), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    draw.text((70, 40), title, fill=(20, 20, 20))
    draw.text((70, 95), takeaway, fill=(50, 50, 50))
    labels = ["A", "B", "C", "D", "E", "F"]
    cols = 2
    rows = int(math.ceil(len(image_paths) / cols))
    cell_w = width_px // cols
    cell_h = (height_px - 180) // max(rows, 1)
    for idx, path in enumerate(image_paths):
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, (cell_w - 70, cell_h - 70))
        x = (idx % cols) * cell_w + 35
        y = (idx // cols) * cell_h + 140
        panel.paste(image, (x, y))
        draw.text((x, y - 28), labels[idx], fill=(0, 0, 0))
    fig = figure_from_pixels(width_px, height_px, dpi=600)
    ax = fig.add_subplot(111)
    ax.imshow(np.asarray(panel))
    ax.axis("off")
    save_figure_triplet(fig, output_base, dpi=600)


def _score_threshold_to_x(value: str) -> float:
    return 0.10 if value == "default" else float(value)


def _build_storyline() -> dict:
    return {
        "status": "completed",
        "language": "zh-CN",
        "slides": [
            {
                "title": "为什么离线 AP 正常不等于部署可信",
                "takeaway": "部署验收要同时看扰动矩阵、部署边界、失效归因和健康指标。",
                "bullets": [
                    "PointPillars / OpenPCDet 是底座，不是这个项目的创新点。",
                    "核心问题是模型上线前，如何判断 runtime / TensorRT 迁移后结果是否仍然可信。",
                    "本项目把 offline official AP、online latency、core latency 和 runtime health 指标分开汇报。",
                ],
                "recommended_panel": "slide1_application_and_acceptance_chain",
                "speaker_notes": "先定义边界：这不是新 detector，而是一套模型上线前的部署验收与异常归因工具链。",
                "likely_followup": "这是不是新的 3D detection model？",
                "answer_suggestion": "不是。检测模型仍沿用 OpenPCDet / PointPillars，新增的是扰动矩阵、dense diagnostics、health risk 和部署边界分析。",
                "forbidden_claims": FORBIDDEN,
            },
            {
                "title": "点云退化和距离退化不会均匀影响模型",
                "takeaway": "随机 dropout、距离裁剪和远距离退化会先放大远距离与弱类别失效。",
                "bullets": [
                    "要看 setting-level AP，也要看 per-frame box count、per-box score/range 和 per-object matched/missed。",
                    "少量 setting 点只能给 summary；真正支撑归因的是 dense diagnostics。",
                    "当前最敏感的 detector-level 扰动是高强度 point dropout。",
                ],
                "recommended_panel": "slide2_pointcloud_and_range_degradation",
                "speaker_notes": "这一页解释为什么必须把图表从 setting-level 扩展到 frame / box / object 三层，才能讲清失效模式。",
                "likely_followup": "为什么 Cyclist 和 40-60m 更敏感？",
                "answer_suggestion": "这类目标本来点数和可见性更弱，退化后更容易先掉召回，range-bin 和 class-bin 统计会先体现出来。",
                "forbidden_claims": FORBIDDEN,
            },
            {
                "title": "标定和时序 proxy 必须诚实讲边界",
                "takeaway": "yaw / time offset 这里只是 proxy，不能包装成 detector AP 或真实同步实验。",
                "bullets": [
                    "yaw 图是 projection-level sensitivity，不是 detector AP。",
                    "pitch / roll / translation 这一轮没有真实跑完，就明确标 skipped。",
                    "far-dropout 和 Gaussian noise 这一轮也不伪造成完整趋势曲线。",
                ],
                "recommended_panel": "slide3_distance_noise_and_projection_proxy",
                "speaker_notes": "这一页的重点不是把 proxy 讲得很强，而是把边界讲清楚，避免把 projection/time proxy 说成 detector-level 结论。",
                "likely_followup": "为什么不直接说做了时间同步实验？",
                "answer_suggestion": "因为这里只有 adjacent-frame proxy，没有 IMU 或 ego-motion compensation，所以只能叫 time-offset proxy。",
                "forbidden_claims": FORBIDDEN,
            },
            {
                "title": "后处理扰动更适合看输出分布漂移",
                "takeaway": "score threshold 变化不是拿来刷 AP，而是看 box count、score 和 class 分布怎么漂移。",
                "bullets": [
                    "score threshold 属于后处理扰动实验，不是调参优化结果。",
                    "NMS 和 max_boxes 这一轮没有真实跑完，就不伪造结果。",
                    "时序 proxy 和后处理扰动更适合结合 health risk 一起看。",
                ],
                "recommended_panel": "slide4_time_and_postprocess_diagnostics",
                "speaker_notes": "这一页强调后处理扰动是上线前的鲁棒性检查，不是通过改阈值把结果做上去。",
                "likely_followup": "你是不是靠调阈值把结果刷出来的？",
                "answer_suggestion": "不是。阈值这里只作为扰动变量，baseline 和 deployment parity 都不改 official evaluator，也不做阈值 hack。",
                "forbidden_claims": FORBIDDEN,
            },
            {
                "title": "TensorRT 和 health monitoring 只是验收系统的一部分",
                "takeaway": "当前成立的是 backbone/head-only TRT 边界和无标签 health risk 预警，不是 full TensorRT detector。",
                "bullets": [
                    "TRT 只在 PyTorch VFE/scatter + TRT backbone/head + native postprocess 的边界内成立。",
                    "frame-level health risk 只能叫 anomaly proxy，不能替代 GT AP。",
                    "1000-frame slice 只能写 slice，不能写 full-val。",
                ],
                "recommended_panel": "slide5_deployment_precision_and_health_monitoring",
                "speaker_notes": "最后把部署边界讲清楚：这部分有工程价值，但不夸大，不把部分成功写成 full TensorRT detector。",
                "likely_followup": "这算 TensorRT 部署成功吗？",
                "answer_suggestion": "算分层部署成功，不算 full TensorRT detector 成功。当前有效边界是 backbone/head-only。",
                "forbidden_claims": FORBIDDEN,
            },
        ],
    }


def main() -> None:
    args = parse_args()
    input_dir = _resolve(args.input_dir)
    figures_dir = ensure_dir(_resolve(args.figures_dir))
    ppt_dir = ensure_dir(_resolve(args.ppt_dir))
    plot_data_dir = ensure_dir(input_dir / "plot_data")
    origin_dir = ensure_dir(input_dir / "origin_plot_data")
    metadata_dir = ensure_dir(input_dir / "plot_data_metadata")

    _copy_or_alias_dense_csv(input_dir)

    dense_dir = input_dir / "dense_diagnostics"
    per_setting_ap = _read_csv(input_dir / "per_setting_ap.csv")
    per_frame_prediction = _read_csv(dense_dir / "per_frame_prediction_dense.csv")
    per_box_prediction = _read_csv(dense_dir / "per_box_prediction_dense.csv")
    gt_dense = _read_csv(dense_dir / "per_gt_object_detection_dense.csv")
    per_object_projection = _read_csv(dense_dir / "per_object_projection_shift_dense.csv")
    per_frame_time_offset = _read_csv(dense_dir / "per_frame_time_offset_dense.csv")
    per_frame_latency = _read_csv(dense_dir / "per_frame_latency_dense.csv")
    runtime_health = _read_csv(input_dir / "runtime_health_metrics.csv")
    health_corr = _read_csv(input_dir / "health_metric_correlation.csv")
    experiment_summary = read_json_or_default(input_dir / "deployment_acceptance_experiment_summary.json", {})
    registry = read_json_or_default(input_dir / "run_registry" / "all_settings_registry.json", {})
    final_report_json = read_json_or_default(input_dir / "deployment_acceptance_final_report.json", {})
    diff_report = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_diff.json", {})
    outlier_diff_rows = _read_csv(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_topk_diff.csv")
    baseline_1000 = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_baseline_eval_1000.json", {})
    wrapper_1000 = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/wrapper_pytorch_core_eval_1000.json", {})
    trt_1000 = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_eval_1000.json", {})
    online_latency = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_online_latency.json", {})

    report_path = str(input_dir / "deployment_acceptance_final_report.md")

    # 04 result card refresh
    pyt_core_mean = _to_float(trt_1000.get("latency_summary", {}).get("pytorch_core_ms", {}).get("mean"))
    trt_core_mean = _to_float(trt_1000.get("latency_summary", {}).get("trt_core_ms", {}).get("mean"))
    pyt_online_mean = _to_float(online_latency.get("pytorch_summary", {}).get("online_total_ms", {}).get("mean"))
    trt_online_mean = _to_float(online_latency.get("trt_summary", {}).get("online_total_ms", {}).get("mean"))
    _summary_card(
        "04_deployment_boundary_and_trt_result_card",
        "Deployment Boundary and TRT Result Card",
        [
            "PyTorch: voxelization / preprocessing + VFE + scatter",
            "TensorRT: BEV backbone + dense head only",
            "OpenPCDet native: post_processing + KITTI export + official evaluator",
            (
                "1000-frame slice AP_R40 moderate "
                f"Car {trt_1000.get('official_result_dict', {}).get('Car_3d/moderate_R40', 0.0):.4f}, "
                f"Ped {trt_1000.get('official_result_dict', {}).get('Pedestrian_3d/moderate_R40', 0.0):.4f}, "
                f"Cyc {trt_1000.get('official_result_dict', {}).get('Cyclist_3d/moderate_R40', 0.0):.4f}"
            ),
            f"Core latency mean {pyt_core_mean:.2f} -> {trt_core_mean:.2f} ms",
            f"Online latency mean {pyt_online_mean:.2f} -> {trt_online_mean:.2f} ms",
            "Boundary is valid for backbone/head-only TensorRT, not full TensorRT detector.",
        ],
        figures_dir,
        metadata_dir,
        report_path,
    )

    # 13-16 skipped cards
    _summary_card(
        "13_far_dropout_ap_curve",
        "Far-point Dropout AP Curve",
        [
            "Not executed this turn.",
            "Reason: far-range dropout sweep was not batch-run in the current acceptance batch.",
            "Registry keeps this group as skipped/optional heavy instead of fabricating a curve.",
        ],
        figures_dir,
        metadata_dir,
        report_path,
        skipped=True,
        skipped_reason="far_point_dropout quick-dense was not executed in the current batch.",
    )
    _summary_card(
        "14_far_dropout_failure_by_range_heatmap",
        "Far-point Dropout Failure Heatmap",
        [
            "Not executed this turn.",
            "Would compare far-range degradation against range-bin recall if real runs were available.",
            "Current figure is an explicit skipped summary card.",
        ],
        figures_dir,
        metadata_dir,
        report_path,
        skipped=True,
        skipped_reason="far_point_dropout quick-dense was not executed in the current batch.",
    )
    _summary_card(
        "15_noise_ap_curve",
        "Gaussian XYZ Noise AP Curve",
        [
            "Not executed this turn.",
            "Noise sweep remains optional heavy work in the acceptance matrix.",
            "No AP trend is claimed without real execution.",
        ],
        figures_dir,
        metadata_dir,
        report_path,
        skipped=True,
        skipped_reason="gaussian_xyz_noise quick-dense was not executed in the current batch.",
    )
    _summary_card(
        "16_noise_score_range_scatter",
        "Noise Score-Range Scatter",
        [
            "Not executed this turn.",
            "Would require real per-box outputs under Gaussian XYZ noise.",
            "Current figure is intentionally a skipped card.",
        ],
        figures_dir,
        metadata_dir,
        report_path,
        skipped=True,
        skipped_reason="gaussian_xyz_noise quick-dense was not executed in the current batch.",
    )

    # 17-18 yaw projection
    rows17 = [row for row in per_object_projection if row.get("reprojection_shift_px") not in (None, "", "None")]
    if rows17:
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        sampled = rows17[:: max(len(rows17) // 45000, 1)]
        _scatter_by_category(ax, sampled, "yaw_deg", "reprojection_shift_px", "class_name", "Yaw Reprojection Shift per Object", "yaw_deg", "reprojection_shift_px")
        _write_bundle(
            "17_yaw_reprojection_shift_per_object_scatter",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            rows17,
            list(rows17[0].keys()),
            "object",
            len(rows17),
            "yaw_deg",
            "reprojection_shift_px",
            "class_name",
            "pixels",
            report_path,
            "Projection-level yaw sensitivity per object.",
            "Projection proxy only; not detector AP.",
        )
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        yaw_vals = sorted({float(row["yaw_deg"]) for row in rows17})
        groups = [[float(item["reprojection_shift_px"]) for item in rows17 if float(item["yaw_deg"]) == yaw] for yaw in yaw_vals]
        ax.boxplot(groups, positions=np.arange(len(yaw_vals)))
        ax.set_xticks(np.arange(len(yaw_vals)))
        ax.set_xticklabels([str(v) for v in yaw_vals], rotation=45)
        apply_axis_style(ax, "Yaw Shift Distribution", "yaw_deg", "reprojection_shift_px")
        _write_bundle(
            "18_yaw_shift_distribution_boxplot",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            rows17,
            list(rows17[0].keys()),
            "object",
            len(rows17),
            "yaw_deg",
            "reprojection_shift_px",
            None,
            "pixels",
            report_path,
            "Projection-level yaw-shift distribution.",
            "Projection proxy only.",
        )

    _summary_card(
        "19_pitch_roll_projection_shift_summary",
        "Pitch/Roll Projection Proxy",
        [
            "Not executed this turn.",
            "Pitch and roll projection proxies remain explicitly skipped.",
            "No detector-level AP claim is made.",
        ],
        figures_dir,
        metadata_dir,
        report_path,
        skipped=True,
        skipped_reason="pitch/roll projection proxy was not executed in the current batch.",
    )
    _summary_card(
        "20_translation_projection_shift_summary",
        "Translation Projection Proxy",
        [
            "Not executed this turn.",
            "Detector-input translation perturbation remains out of scope.",
            "Only yaw projection proxy is backed by real dense rows.",
        ],
        figures_dir,
        metadata_dir,
        report_path,
        skipped=True,
        skipped_reason="translation projection proxy was not executed in the current batch.",
    )

    # 21-22 time offset
    rows21 = [row for row in per_frame_time_offset if row.get("center_drift_m") not in (None, "", "None")]
    if rows21:
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        _scatter_by_category(ax, rows21, "frame_id", "center_drift_m", "frame_offset", "Time-offset Center Drift per Frame", "frame_id", "center_drift_m")
        ax.tick_params(axis="x", labelrotation=90)
        _write_bundle(
            "21_time_offset_center_drift_per_frame_scatter",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            rows21,
            list(rows21[0].keys()),
            "frame",
            len(rows21),
            "frame_id",
            "center_drift_m",
            "frame_offset",
            "meters",
            report_path,
            "Adjacent-frame time-offset proxy per-frame drift scatter.",
            "Time-offset proxy only; not real sensor synchronization.",
        )
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        offsets = sorted({int(float(row["frame_offset"])) for row in rows21})
        groups = [[float(item["center_drift_m"]) for item in rows21 if int(float(item["frame_offset"])) == off] for off in offsets]
        ax.boxplot(groups, positions=np.arange(len(offsets)))
        ax.set_xticks(np.arange(len(offsets)))
        ax.set_xticklabels([str(off) for off in offsets])
        apply_axis_style(ax, "Time-offset Drift Distribution", "frame_offset", "center_drift_m")
        _write_bundle(
            "22_time_offset_drift_distribution_boxplot",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            rows21,
            list(rows21[0].keys()),
            "frame",
            len(rows21),
            "frame_offset",
            "center_drift_m",
            None,
            "meters",
            report_path,
            "Distribution of center-drift proxy across frame offsets.",
            "Time-offset proxy only.",
        )

    # 23-27 score threshold
    score_rows = [row for row in per_setting_ap if row["perturbation_type"] == "postprocess_score_threshold"]
    score_rows.sort(key=lambda row: _score_threshold_to_x(row["perturbation_value"]))
    if score_rows:
        xs = [_score_threshold_to_x(row["perturbation_value"]) for row in score_rows]
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        for key, label in [("car_ap_3d_moderate", "Car"), ("ped_ap_3d_moderate", "Pedestrian"), ("cyc_ap_3d_moderate", "Cyclist"), ("mean_ap_3d_moderate", "Mean")]:
            ax.plot(xs, [float(row[key]) for row in score_rows], marker="o", linewidth=2.2, label=label)
        apply_axis_style(ax, "Score-threshold AP Curve", "score_threshold", "moderate 3D AP_R40")
        ax.legend(fontsize=12)
        _write_bundle(
            "23_score_threshold_ap_curve",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            score_rows,
            list(score_rows[0].keys()),
            "setting",
            len(score_rows),
            "score_threshold",
            "AP_R40",
            "class_name",
            "AP_R40",
            report_path,
            "Score-threshold sensitivity curve.",
            "Postprocess perturbation only; not AP hacking.",
        )

        frame_score_rows = [row for row in per_frame_prediction if row["perturbation_type"] == "postprocess_score_threshold"]
        grouped = {}
        for row in frame_score_rows:
            threshold = _score_threshold_to_x(row["perturbation_value"])
            grouped.setdefault(threshold, []).append(float(row["predicted_box_count"]))
        rows24 = [{"score_threshold": th, "predicted_box_count": float(np.mean(vals))} for th, vals in sorted(grouped.items())]
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        ax.plot([row["score_threshold"] for row in rows24], [row["predicted_box_count"] for row in rows24], marker="o", linewidth=2.2, color="#e76f51")
        apply_axis_style(ax, "Score-threshold Box Count", "score_threshold", "predicted_box_count")
        _write_bundle(
            "24_score_threshold_box_count_curve",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            rows24,
            list(rows24[0].keys()),
            "setting",
            len(rows24),
            "score_threshold",
            "predicted_box_count",
            None,
            "count",
            report_path,
            "Mean predicted box count under score-threshold perturbation.",
            "Setting-level summary.",
        )

        rows25 = [row for row in per_box_prediction if row["perturbation_type"] == "postprocess_score_threshold" and row["perturbation_value"] in {"0.00", "0.10", "0.30", "0.60"}]
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        for color, th in zip(["#1d3557", "#2a9d8f", "#e76f51", "#6a4c93"], ["0.00", "0.10", "0.30", "0.60"]):
            subset = [float(row["score"]) for row in rows25 if row["perturbation_value"] == th and row["score"] not in (None, "")]
            if subset:
                ax.hist(subset, bins=40, alpha=0.30, density=True, label=th, color=color)
        apply_axis_style(ax, "Score Distribution by Threshold", "score", "density")
        ax.legend(fontsize=12)
        _write_bundle(
            "25_score_distribution_hist_by_threshold",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            rows25,
            list(rows25[0].keys()),
            "box",
            len(rows25),
            "score",
            "density",
            "score_threshold",
            "mixed",
            report_path,
            "Per-box score distribution under selected thresholds.",
            "Selected thresholds overlaid for readability.",
        )

        rows26 = [row for row in per_box_prediction if row["perturbation_type"] == "postprocess_score_threshold"]
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        sampled = rows26[:: max(len(rows26) // 50000, 1)]
        _scatter_by_category(ax, sampled, "range_m", "score", "class_name", "Score vs Range by Threshold", "range_m", "score")
        _write_bundle(
            "26_score_vs_range_scatter_by_threshold",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            rows26,
            list(rows26[0].keys()),
            "box",
            len(rows26),
            "range_m",
            "score",
            "class_name",
            "mixed",
            report_path,
            "Per-box score/range scatter for score-threshold perturbation.",
            "Rendering sampled; CSV exports full rows.",
        )

        grouped27 = {}
        for row in frame_score_rows:
            th = _score_threshold_to_x(row["perturbation_value"])
            grouped27.setdefault(th, {"car": [], "ped": [], "cyc": []})
            grouped27[th]["car"].append(float(row["car_pred_count"]))
            grouped27[th]["ped"].append(float(row["ped_pred_count"]))
            grouped27[th]["cyc"].append(float(row["cyc_pred_count"]))
        rows27 = []
        for th, vals in sorted(grouped27.items()):
            rows27.append(
                {
                    "score_threshold": th,
                    "car_count_mean": float(np.mean(vals["car"])),
                    "ped_count_mean": float(np.mean(vals["ped"])),
                    "cyc_count_mean": float(np.mean(vals["cyc"])),
                }
            )
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        for key, label in [("car_count_mean", "Car"), ("ped_count_mean", "Pedestrian"), ("cyc_count_mean", "Cyclist")]:
            ax.plot([row["score_threshold"] for row in rows27], [row[key] for row in rows27], marker="o", linewidth=2.2, label=label)
        apply_axis_style(ax, "Class Distribution vs Threshold", "score_threshold", "mean predicted count per frame")
        ax.legend(fontsize=12)
        _write_bundle(
            "27_class_distribution_vs_threshold",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            rows27,
            list(rows27[0].keys()),
            "setting",
            len(rows27),
            "score_threshold",
            "mean_pred_count",
            "class_name",
            "count",
            report_path,
            "Class-count drift across score thresholds.",
            "Aggregated from dense per-frame rows.",
        )

    _summary_card(
        "28_topk_maxboxes_perturbation_summary",
        "Top-k / Max-boxes Perturbation",
        [
            "Not executed this turn.",
            "OpenPCDet max-boxes perturbation was not changed in the current batch.",
            "No AP or box-count trend is claimed without real execution.",
        ],
        figures_dir,
        metadata_dir,
        report_path,
        skipped=True,
        skipped_reason="postprocess_topk/max_boxes perturbation was not executed in the current batch.",
    )
    _summary_card(
        "29_nms_threshold_summary",
        "NMS Threshold Summary",
        [
            "Not executed this turn.",
            "NMS threshold was intentionally not changed to avoid threshold-hack ambiguity.",
            "Current figure is a skipped summary card.",
        ],
        figures_dir,
        metadata_dir,
        report_path,
        skipped=True,
        skipped_reason="postprocess_nms_threshold was not executed in the current batch.",
    )

    # 30-32 latency
    rows30 = [row for row in per_frame_latency if row["runtime_variant"] in {"openpcdet_pytorch", "trt_backbone_head_only"} and row["core_latency_ms"] not in (None, "", "None")]
    if rows30:
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        _scatter_by_category(ax, rows30, "frame_id", "core_latency_ms", "runtime_variant", "TRT Per-frame Latency Scatter (Full)", "frame_id", "core_latency_ms")
        ax.tick_params(axis="x", labelrotation=90)
        _write_bundle(
            "30_trt_per_frame_latency_scatter_full",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            rows30,
            list(rows30[0].keys()),
            "frame",
            len(rows30),
            "frame_id",
            "core_latency_ms",
            "runtime_variant",
            "ms",
            report_path,
            "Full-range per-frame latency scatter for PyTorch and TRT.",
            "Core latency only; not end-to-end detector latency.",
        )
        y_clip = np.percentile([float(row["core_latency_ms"]) for row in rows30], 95.0)
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        _scatter_by_category(ax, rows30, "frame_id", "core_latency_ms", "runtime_variant", "TRT Per-frame Latency Scatter (Clipped)", "frame_id", "core_latency_ms")
        ax.set_ylim(0, y_clip)
        ax.tick_params(axis="x", labelrotation=90)
        _write_bundle(
            "31_trt_per_frame_latency_scatter_clipped",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            rows30,
            list(rows30[0].keys()),
            "frame",
            len(rows30),
            "frame_id",
            "core_latency_ms",
            "runtime_variant",
            "ms",
            report_path,
            "Clipped latency view for readability.",
            "Core latency only; clipped at p95.",
        )
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        cdf_rows = []
        for color, variant in zip(["#1d3557", "#e76f51"], ["openpcdet_pytorch", "trt_backbone_head_only"]):
            vals = sorted(float(row["core_latency_ms"]) for row in rows30 if row["runtime_variant"] == variant)
            probs = np.linspace(0.0, 1.0, num=len(vals), endpoint=True)
            ax.plot(vals, probs, linewidth=2.2, color=color, label=variant)
            for x, p in zip(vals, probs):
                cdf_rows.append({"runtime_variant": variant, "latency_ms": x, "cdf": float(p)})
        apply_axis_style(ax, "TRT Latency CDF", "latency_ms", "cumulative_probability")
        ax.legend(fontsize=12)
        _write_bundle(
            "32_trt_latency_cdf",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            cdf_rows,
            list(cdf_rows[0].keys()),
            "frame",
            len(cdf_rows),
            "latency_ms",
            "cdf",
            "runtime_variant",
            "mixed",
            report_path,
            "Latency CDF for PyTorch vs TRT core path.",
            "Core latency only.",
        )

    # 33-35 diff / outlier
    hist_rows = []
    diffs = []
    for row in outlier_diff_rows:
        try:
            a = np.asarray(json.loads(str(row["pytorch_center"]).replace("'", "\"")), dtype=np.float64)
            b = np.asarray(json.loads(str(row["trt_center"]).replace("'", "\"")), dtype=np.float64)
            diff = float(np.linalg.norm(a - b))
        except Exception:
            continue
        hist_rows.append({"sample_id": row["sample_id"], "rank": row["rank"], "topk_center_diff": diff})
        diffs.append(diff)
    if hist_rows:
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        ax.hist(diffs, bins=60, color="#3a86ff", alpha=0.85, log=True)
        apply_axis_style(ax, "TRT Top-k Center Diff Histogram", "topk_center_diff", "count (log)")
        _write_bundle(
            "33_trt_topk_center_diff_hist",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            hist_rows,
            list(hist_rows[0].keys()),
            "box",
            len(hist_rows),
            "topk_center_diff",
            "count",
            None,
            "meters",
            report_path,
            "Top-k center-difference histogram.",
            "Same-frame diff only; not full-detector deployment proof.",
        )
        clipped = [diff for diff in diffs if diff <= np.percentile(diffs, 95.0)]
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        ax.hist(clipped, bins=50, color="#2a9d8f", alpha=0.85)
        apply_axis_style(ax, "TRT Top-k Center Diff (Clipped)", "topk_center_diff", "count")
        _write_bundle(
            "34_trt_topk_center_diff_clipped",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            hist_rows,
            list(hist_rows[0].keys()),
            "box",
            len(hist_rows),
            "topk_center_diff",
            "count",
            None,
            "meters",
            report_path,
            "Clipped view of small center differences.",
            "Clipped at p95 for readability.",
        )

    outliers = diff_report.get("summary", {}).get("outlier_frame_ids", ["000005", "000027", "000236", "000327", "000355", "000361"])
    _summary_card(
        "35_trt_outlier_bev_gallery",
        "TRT Outlier BEV Gallery",
        [
            f"Outlier frames: {', '.join(outliers[:6])}",
            "No rendered BEV overlay set was exported for every outlier frame.",
            "This is an explicit table-style placeholder rather than a fabricated overlay gallery.",
        ],
        figures_dir,
        metadata_dir,
        report_path,
        skipped=True,
        skipped_reason="BEV overlay gallery was not exported for all outlier frames; table gallery retained instead.",
    )

    # 36-38 health
    rows36 = [row for row in per_frame_prediction if (row["perturbation_type"], row["perturbation_value"]) in {("point_dropout", "0.00"), ("point_dropout", "0.40"), ("point_dropout", "0.80")}]
    if rows36:
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        _scatter_by_category(ax, rows36, "frame_id", "health_risk_frame", "perturbation_value", "Per-frame Health Risk Timeline", "frame_id", "health_risk_frame")
        ax.tick_params(axis="x", labelrotation=90)
        _write_bundle(
            "36_per_frame_health_risk_timeline",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            rows36,
            list(rows36[0].keys()),
            "frame",
            len(rows36),
            "frame_id",
            "health_risk_frame",
            "perturbation_value",
            "risk",
            report_path,
            "Frame-level heuristic health-risk timeline.",
            "Label-free risk is an anomaly proxy, not AP.",
        )

    miss_by_key = {}
    total_by_key = {}
    for row in gt_dense:
        key = (row["perturbation_type"], row["perturbation_value"], row["frame_id"])
        total_by_key[key] = total_by_key.get(key, 0) + 1
        miss_by_key[key] = miss_by_key.get(key, 0) + (0 if int(row["matched"]) else 1)
    rows37 = []
    for row in per_frame_prediction:
        key = (row["perturbation_type"], row["perturbation_value"], row["frame_id"])
        total = total_by_key.get(key, 0)
        miss = miss_by_key.get(key, 0)
        rows37.append(
            {
                "perturbation_type": row["perturbation_type"],
                "perturbation_value": row["perturbation_value"],
                "health_risk_frame": float(row["health_risk_frame"]),
                "failure_proxy": (miss / total) if total else 0.0,
            }
        )
    if rows37:
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        sampled = rows37[:: max(len(rows37) // 50000, 1)]
        _scatter_by_category(ax, sampled, "health_risk_frame", "failure_proxy", "perturbation_type", "Health Risk vs Failure Proxy", "health_risk_frame", "failure_proxy")
        _write_bundle(
            "37_health_risk_vs_failure_proxy_scatter",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            rows37,
            list(rows37[0].keys()),
            "frame",
            len(rows37),
            "health_risk_frame",
            "failure_proxy",
            "perturbation_type",
            "mixed",
            report_path,
            "Frame-level health risk against offline failure proxy.",
            "Failure proxy uses GT offline; it is not AP.",
        )

    if health_corr:
        fig = figure_from_pixels(3200, 2200, dpi=600)
        ax = fig.add_subplot(111)
        corr_sorted = sorted(health_corr, key=lambda row: abs(_to_float(row.get("correlation_with_mean_ap_drop"), 0.0) or 0.0), reverse=True)
        cats = [row["metric_name"] for row in corr_sorted]
        vals = [(_to_float(row.get("correlation_with_mean_ap_drop"), 0.0) or 0.0) for row in corr_sorted]
        ax.bar(np.arange(len(cats)), vals, color="#457b9d")
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels(cats, rotation=35, ha="right")
        apply_axis_style(ax, "Health Metric Correlation", "metric_name", "correlation_with_mean_ap_drop")
        _write_bundle(
            "38_health_metric_correlation_bar",
            fig,
            figures_dir,
            plot_data_dir,
            origin_dir,
            metadata_dir,
            health_corr,
            list(health_corr[0].keys()),
            "summary",
            len(health_corr),
            "metric_name",
            "correlation",
            None,
            "correlation",
            report_path,
            "Correlation between heuristic health metrics and AP drop.",
            "Summary bar chart with setting-level metric count.",
        )

    _summary_card(
        "39_selected_1000_dropout_vs_quick_dense_comparison",
        "Selected-1000 vs Quick-dense: Dropout",
        [
            "Partial only.",
            "Shared selected-1000 dropout runs were not executed in the current turn.",
            "Comparison is deferred rather than fabricated.",
        ],
        figures_dir,
        metadata_dir,
        report_path,
        skipped=True,
        skipped_reason="selected_1000 point_dropout settings beyond deployment_precision were not executed in the current turn.",
    )
    _summary_card(
        "40_selected_1000_range_crop_comparison",
        "Selected-1000 vs Quick-dense: Range Crop",
        [
            "Partial only.",
            "Shared selected-1000 range-crop runs were not executed in the current turn.",
            "Quick-dense trend exists, but 1000-frame confirmation is pending.",
        ],
        figures_dir,
        metadata_dir,
        report_path,
        skipped=True,
        skipped_reason="selected_1000 range_crop settings were not executed in the current turn.",
    )
    _summary_card(
        "41_selected_1000_score_threshold_comparison",
        "Selected-1000 vs Quick-dense: Score Threshold",
        [
            "Partial only.",
            "Selected-1000 score-threshold sweep was not executed in the current turn.",
            "No cross-scope trend claim is made without real runs.",
        ],
        figures_dir,
        metadata_dir,
        report_path,
        skipped=True,
        skipped_reason="selected_1000 score-threshold settings were not executed in the current turn.",
    )
    _summary_card(
        "42_final_acceptance_dashboard",
        "Final Acceptance Dashboard",
        [
            f"Total settings in registry: {experiment_summary.get('total_settings', 'N/A')}",
            f"Executed: {experiment_summary.get('executed_settings', 'N/A')}, skipped: {experiment_summary.get('skipped_settings', 'N/A')}, partial: {experiment_summary.get('partial_settings', 'N/A')}",
            f"Total frame-runs: {experiment_summary.get('total_frame_runs', 'N/A')}",
            f"Most sensitive perturbation: {final_report_json.get('most_sensitive_perturbation', 'N/A')}",
            f"Most sensitive class: {final_report_json.get('most_sensitive_class', 'N/A')}, range: {final_report_json.get('most_sensitive_range', 'N/A')}",
            f"Best health metric: {final_report_json.get('best_health_metric', 'N/A')}",
        ],
        figures_dir,
        metadata_dir,
        report_path,
    )

    requested_names = [
        "01_application_scenario_deployment_acceptance",
        "02_acceptance_benchmark_pipeline",
        "03_perturbation_matrix_table",
        "04_deployment_boundary_and_trt_result_card",
        "05_dropout_ap_curve_setting_level",
        "06_dropout_per_frame_box_count_scatter",
        "07_dropout_score_vs_range_box_scatter",
        "08_dropout_gt_range_detected_missed_scatter",
        "09_dropout_failure_heatmap_range_class",
        "10_range_crop_ap_curve_setting_level",
        "11_range_crop_prediction_range_histogram",
        "12_range_crop_gt_range_recall_scatter",
        "13_far_dropout_ap_curve",
        "14_far_dropout_failure_by_range_heatmap",
        "15_noise_ap_curve",
        "16_noise_score_range_scatter",
        "17_yaw_reprojection_shift_per_object_scatter",
        "18_yaw_shift_distribution_boxplot",
        "19_pitch_roll_projection_shift_summary",
        "20_translation_projection_shift_summary",
        "21_time_offset_center_drift_per_frame_scatter",
        "22_time_offset_drift_distribution_boxplot",
        "23_score_threshold_ap_curve",
        "24_score_threshold_box_count_curve",
        "25_score_distribution_hist_by_threshold",
        "26_score_vs_range_scatter_by_threshold",
        "27_class_distribution_vs_threshold",
        "28_topk_maxboxes_perturbation_summary",
        "29_nms_threshold_summary",
        "30_trt_per_frame_latency_scatter_full",
        "31_trt_per_frame_latency_scatter_clipped",
        "32_trt_latency_cdf",
        "33_trt_topk_center_diff_hist",
        "34_trt_topk_center_diff_clipped",
        "35_trt_outlier_bev_gallery",
        "36_per_frame_health_risk_timeline",
        "37_health_risk_vs_failure_proxy_scatter",
        "38_health_metric_correlation_bar",
        "39_selected_1000_dropout_vs_quick_dense_comparison",
        "40_selected_1000_range_crop_comparison",
        "41_selected_1000_score_threshold_comparison",
        "42_final_acceptance_dashboard",
    ]
    contact_sheet = figures_dir / "deployment_acceptance_dense_contact_sheet.png"
    save_contact_sheet([figures_dir / f"{name}.png" for name in requested_names], contact_sheet, columns=5, width_px=8400, dpi=600)

    _panel(
        ppt_dir / "slide1_application_and_acceptance_chain",
        "LiDAR Deployment Acceptance Chain",
        "离线 AP 正常不等于部署可信，验收要从扰动矩阵和部署边界一起看。",
        [
            figures_dir / "01_application_scenario_deployment_acceptance.png",
            figures_dir / "02_acceptance_benchmark_pipeline.png",
            figures_dir / "03_perturbation_matrix_table.png",
            figures_dir / "04_deployment_boundary_and_trt_result_card.png",
        ],
    )
    _panel(
        ppt_dir / "slide2_pointcloud_and_range_degradation",
        "Pointcloud and Range Degradation",
        "点云随机退化、距离裁剪和远距离稀疏会优先放大远距离与弱类别失效。",
        [
            figures_dir / "05_dropout_ap_curve_setting_level.png",
            figures_dir / "06_dropout_per_frame_box_count_scatter.png",
            figures_dir / "07_dropout_score_vs_range_box_scatter.png",
            figures_dir / "08_dropout_gt_range_detected_missed_scatter.png",
            figures_dir / "09_dropout_failure_heatmap_range_class.png",
            figures_dir / "11_range_crop_prediction_range_histogram.png",
        ],
    )
    _panel(
        ppt_dir / "slide3_distance_noise_and_projection_proxy",
        "Distance, Noise, and Projection Proxy",
        "标定与时序这里只做 proxy；没有真实 detector-level 注入，就不能把 proxy 写成 detector AP。",
        [
            figures_dir / "10_range_crop_ap_curve_setting_level.png",
            figures_dir / "13_far_dropout_ap_curve.png",
            figures_dir / "15_noise_ap_curve.png",
            figures_dir / "17_yaw_reprojection_shift_per_object_scatter.png",
            figures_dir / "18_yaw_shift_distribution_boxplot.png",
            figures_dir / "19_pitch_roll_projection_shift_summary.png",
        ],
    )
    _panel(
        ppt_dir / "slide4_time_and_postprocess_diagnostics",
        "Time and Postprocess Diagnostics",
        "时序 proxy 和后处理扰动更适合看输出分布漂移、box count 变化和异常预警指标。",
        [
            figures_dir / "21_time_offset_center_drift_per_frame_scatter.png",
            figures_dir / "22_time_offset_drift_distribution_boxplot.png",
            figures_dir / "23_score_threshold_ap_curve.png",
            figures_dir / "24_score_threshold_box_count_curve.png",
            figures_dir / "25_score_distribution_hist_by_threshold.png",
            figures_dir / "27_class_distribution_vs_threshold.png",
        ],
    )
    _panel(
        ppt_dir / "slide5_deployment_precision_and_health_monitoring",
        "Deployment Precision and Health Monitoring",
        "当前成立的是 backbone/head-only TensorRT 边界和无标签 health risk 预警，不是 full TensorRT detector。",
        [
            figures_dir / "30_trt_per_frame_latency_scatter_full.png",
            figures_dir / "32_trt_latency_cdf.png",
            figures_dir / "33_trt_topk_center_diff_hist.png",
            figures_dir / "35_trt_outlier_bev_gallery.png",
            figures_dir / "36_per_frame_health_risk_timeline.png",
            figures_dir / "42_final_acceptance_dashboard.png",
        ],
    )

    storyline = _build_storyline()
    write_json(input_dir / "deployment_acceptance_ppt_storyline.json", storyline)
    story_lines = ["# Deployment Acceptance PPT Storyline", ""]
    for idx, slide in enumerate(storyline["slides"], start=1):
        story_lines.extend(
            [
                f"## 第 {idx} 页：{slide['title']}",
                "",
                f"- Takeaway: {slide['takeaway']}",
                f"- 推荐 Panel: `{slide['recommended_panel']}`",
                f"- Bullet 1: {slide['bullets'][0]}",
                f"- Bullet 2: {slide['bullets'][1]}",
                f"- Bullet 3: {slide['bullets'][2]}",
                f"- 讲稿: {slide['speaker_notes']}",
                f"- 面试官可能追问: {slide['likely_followup']}",
                f"- 回答建议: {slide['answer_suggestion']}",
                f"- 禁止说法: {', '.join(slide['forbidden_claims'])}",
                "",
            ]
        )
    write_markdown(input_dir / "deployment_acceptance_ppt_storyline.md", "\n".join(story_lines))

    sensitivity_rows = [row for row in per_setting_ap if row.get("delta_mean") not in (None, "", "None")]
    most_sensitive = min(sensitivity_rows, key=lambda row: float(row["delta_mean"])) if sensitivity_rows else None
    skipped_rows = [row for row in registry.get("rows", []) if row.get("status") == "skipped"]
    partial_rows = [row for row in registry.get("rows", []) if row.get("status") == "partial"]
    selected_paths_payload = {
        "matrix": str(input_dir / "selected_1000" / "selected_1000_perturbation_matrix.csv"),
        "ap": str(input_dir / "selected_1000" / "selected_1000_per_setting_ap.csv"),
        "health": str(input_dir / "selected_1000" / "selected_1000_prediction_health.csv"),
        "range": str(input_dir / "selected_1000" / "selected_1000_failure_by_range.csv"),
        "class": str(input_dir / "selected_1000" / "selected_1000_failure_by_class.csv"),
        "runtime": str(input_dir / "selected_1000" / "selected_1000_runtime_health_metrics.csv"),
    }
    proxy_paths_payload = {
        "yaw": str(input_dir / "proxy_extended" / "yaw_projection_shift.csv"),
        "pitch": str(input_dir / "proxy_extended" / "pitch_projection_shift.csv"),
        "roll": str(input_dir / "proxy_extended" / "roll_projection_shift.csv"),
        "translation": str(input_dir / "proxy_extended" / "translation_projection_shift.csv"),
        "time": str(input_dir / "proxy_extended" / "time_offset_proxy.csv"),
    }
    updated_report = dict(final_report_json)
    updated_report.update(
        {
            "status": "completed",
            "application_scenario": "LiDAR 3D detection deployment acceptance and abnormal attribution for unmanned platforms and intelligent hardware.",
            "dense_diagnostics_needed": True,
            "why_dense_diagnostics_are_needed": [
                "Setting-level AP curves with a small number of x-points only show coarse trends.",
                "Deployment acceptance needs per-frame latency, per-box score/range, per-object matched/missed, and frame-level health-risk traces.",
                "This project exports both summary-level and dense-level diagnostics so the acceptance report is auditable.",
            ],
            "experiment_summary_path": str(input_dir / "deployment_acceptance_experiment_summary.json"),
            "registry_paths": {
                "csv": str(input_dir / "run_registry" / "all_settings_registry.csv"),
                "json": str(input_dir / "run_registry" / "all_settings_registry.json"),
            },
            "selected_1000_paths": selected_paths_payload,
            "proxy_extended_paths": proxy_paths_payload,
            "total_settings": experiment_summary.get("total_settings"),
            "executed_settings": experiment_summary.get("executed_settings"),
            "skipped_settings": experiment_summary.get("skipped_settings"),
            "partial_settings": experiment_summary.get("partial_settings"),
            "total_frame_runs": experiment_summary.get("total_frame_runs"),
            "quick_dense_frame_runs": experiment_summary.get("quick_dense_frame_runs"),
            "selected_1000_frame_runs": experiment_summary.get("selected_1000_frame_runs"),
            "most_sensitive_perturbation": f"{most_sensitive['perturbation_type']}={most_sensitive['perturbation_value']}" if most_sensitive else updated_report.get("most_sensitive_perturbation"),
            "most_sensitive_class": updated_report.get("most_sensitive_class", "Cyclist"),
            "most_sensitive_range": updated_report.get("most_sensitive_range", "40-60m"),
            "best_health_metric": updated_report.get("best_health_metric", "prediction_count_drift"),
            "paths": {
                "dense_diagnostics_dir": str(input_dir / "dense_diagnostics"),
                "plot_data_dir": str(input_dir / "plot_data"),
                "origin_plot_data_dir": str(input_dir / "origin_plot_data"),
                "metadata_dir": str(input_dir / "plot_data_metadata"),
                "figures_dir": str(figures_dir),
                "ppt_dir": str(ppt_dir),
                "contact_sheet": str(contact_sheet),
            },
            "selected_1000_scope": {
                "frame_count": 1000,
                "eval_scope": "1000-frame-slice",
                "not_full_val": True,
            },
            "proxy_scope": {
                "yaw": "projection-level proxy",
                "time_offset": "adjacent-frame / frame-shift proxy",
            },
            "skipped_perturbations": skipped_rows,
            "partial_perturbations": partial_rows,
            "safe_claims": [
                "This is a deployment-acceptance and abnormal-attribution toolchain for LiDAR 3D detection models.",
                "Dense diagnostics are available at setting / frame / box / object levels.",
                "Backbone/head-only TensorRT is a valid bounded deployment milestone with AP parity on evaluated slices.",
            ],
            "forbidden_claims": FORBIDDEN,
        }
    )
    write_json(input_dir / "deployment_acceptance_final_report.json", updated_report)

    report_lines = [
        "# Deployment Acceptance Final Report",
        "",
        "## 1. Application Scenario",
        "",
        "面向无人平台 / 智能硬件 LiDAR 感知模型上线前的自动化部署验收与异常归因系统。",
        "",
        "## 2. Why Offline AP Is Not Enough",
        "",
        "- 离线 AP 正常不代表部署 runtime / TensorRT 迁移后仍然可信。",
        "- 上线前需要同时检查点云退化、距离退化、标定 / 时序 proxy、后处理扰动和部署精度差异。",
        "",
        "## 3. Experiment Matrix",
        "",
        f"- Total settings: `{experiment_summary.get('total_settings')}`",
        f"- Executed settings: `{experiment_summary.get('executed_settings')}`",
        f"- Skipped settings: `{experiment_summary.get('skipped_settings')}`",
        f"- Partial settings: `{experiment_summary.get('partial_settings')}`",
        f"- Total frame-runs: `{experiment_summary.get('total_frame_runs')}`",
        f"- Quick-dense frame-runs: `{experiment_summary.get('quick_dense_frame_runs')}`",
        f"- Selected-1000 frame-runs: `{experiment_summary.get('selected_1000_frame_runs')}`",
        "",
        "## 4. Why Dense Diagnostics Are Needed",
        "",
        "- setting-level AP 曲线只能说明粗趋势。",
        "- 真正支撑部署验收的是 per-frame latency、per-box score/range、per-object matched/missed、frame-level health risk。",
        "",
        "## 5. Detector-level Perturbations",
        "",
        f"- Most sensitive perturbation: `{updated_report['most_sensitive_perturbation']}`",
        f"- Most sensitive class: `{updated_report['most_sensitive_class']}`",
        f"- Most sensitive range: `{updated_report['most_sensitive_range']}`",
        "",
        "## 6. Deployment Precision Boundary",
        "",
        "- 成立边界：PyTorch VFE/scatter + TensorRT backbone/head + OpenPCDet native postprocess/export。",
        "- 不成立边界：full TensorRT detector。",
        "",
        "## 7. Health Metrics",
        "",
        f"- Best health metric: `{updated_report['best_health_metric']}`",
        "- Label-free health risk 只作为 anomaly proxy，不替代 GT AP。",
        "",
        "## 8. Proxy Boundary",
        "",
        "- yaw sensitivity 是 projection-level proxy，不是 detector AP。",
        "- time offset 是 adjacent-frame proxy，不是带 IMU / ego-motion compensation 的真实同步实验。",
        "",
        "## 9. Safe Claims / Forbidden Claims",
        "",
        "- Safe claims:",
        "  - deployment acceptance and abnormal attribution toolchain",
        "  - dense diagnostics at setting / frame / box / object levels",
        "  - backbone/head-only TensorRT bounded deployment milestone",
        "- Forbidden claims:",
        f"  - {FORBIDDEN[0]}",
        f"  - {FORBIDDEN[1]}",
        f"  - {FORBIDDEN[2]}",
        f"  - {FORBIDDEN[3]}",
        f"  - {FORBIDDEN[4]}",
        "",
    ]
    write_markdown(input_dir / "deployment_acceptance_final_report.md", "\n".join(report_lines))

    print(
        json.dumps(
            {
                "status": "completed",
                "figures_dir": str(figures_dir),
                "ppt_dir": str(ppt_dir),
                "contact_sheet": str(contact_sheet),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
