from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.kitti_io import resolve_frame_assets
from runtime.lidar_system_algorithm.pointpillars_wrapper_runtime import run_pointpillars_modules, summarize_batch_dict
from runtime.lidar_system_algorithm.report_schema import write_json, write_markdown
from runtime.lidar_system_algorithm.tensorrt_debug_utils import pad_prepared_inputs


AUDIT_KEYS = [
    "voxels",
    "voxel_coords",
    "voxel_num_points",
    "batch_size",
    "pillar_features",
    "spatial_features",
    "spatial_features_2d",
    "batch_cls_preds",
    "batch_box_preds",
    "dir_cls_preds",
    "cls_preds_normalized",
    "has_class_labels",
    "batch_pred_labels",
    "pred_boxes",
    "pred_scores",
    "pred_labels",
    "valid_pillar_mask",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit batch_dict contract between OpenPCDet original and wrapper_pytorch_core.")
    parser.add_argument("--kitti-root", default="data/kitti_object_raw/extracted")
    parser.add_argument("--openpcdet-root", default="external/OpenPCDet")
    parser.add_argument("--cfg-file", default="external/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml")
    parser.add_argument("--ckpt", default="checkpoints/pointpillar_kitti.pth")
    parser.add_argument("--frame-ids", default="000002,000003,000004")
    parser.add_argument("--bucket-size", type=int, default=8192)
    parser.add_argument("--padding-strategy", default="unique_dummy_coord_padding")
    parser.add_argument("--zero-padded-pillars-after-vfe", action="store_true")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def main() -> None:
    args = parse_args()
    report_dir = _resolve(args.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json = report_dir / "wrapper_pytorch_core_batch_dict_audit.json"
    report_md = report_dir / "wrapper_pytorch_core_batch_dict_audit.md"

    try:
        import torch
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils
    except Exception as exc:
        payload = {"status": "skipped", "reason": f"runtime import failed: {type(exc).__name__}: {exc}"}
        write_json(report_json, payload)
        write_markdown(report_md, f"# Wrapper Batch Dict Audit\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
        return

    openpcdet_root = _resolve(args.openpcdet_root)
    os.chdir(openpcdet_root / "tools")
    cfg_from_yaml_file(str(_resolve(args.cfg_file)), cfg)

    class RuntimeDataset(DatasetTemplate):
        def __init__(self):
            super().__init__(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                training=False,
                root_path=openpcdet_root,
                logger=common_utils.create_logger(),
            )

        def __len__(self):
            return 0

        def __getitem__(self, index):
            raise IndexError(index)

        def prepare_runtime(self, points: np.ndarray, frame_id: str) -> dict:
            data_dict = {"points": points, "frame_id": frame_id}
            data_dict = self.set_lidar_aug_matrix(data_dict)
            data_dict = self.point_feature_encoder.forward(data_dict)
            for processor in self.data_processor.data_processor_queue:
                data_dict = processor(data_dict=data_dict)
            data_dict.pop("gt_names", None)
            return data_dict

    dataset = RuntimeDataset()
    logger = common_utils.create_logger()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(Path(args.ckpt).expanduser()), logger=logger, to_cpu=True)
    model.cuda().eval()
    modules_for_export = torch.nn.ModuleList(list(model.module_list)).cuda().eval()

    frame_ids = [item.strip() for item in args.frame_ids.split(",") if item.strip()]
    rows = []
    for frame_id in frame_ids:
        assets = resolve_frame_assets(_resolve(args.kitti_root), frame_id)
        points = np.fromfile(str(assets.lidar_path), dtype=np.float32).reshape(-1, 4)
        prepared = dataset.prepare_runtime(points, frame_id)
        valid_pillar_count = int(prepared["voxels"].shape[0])
        padded, pad_stats = pad_prepared_inputs(prepared, args.bucket_size, args.padding_strategy)

        original_batch = dataset.collate_batch([prepared])
        wrapper_batch = dataset.collate_batch([padded])
        load_data_to_gpu(original_batch)
        load_data_to_gpu(wrapper_batch)

        with torch.no_grad():
            original_forward, original_stages = run_pointpillars_modules(
                modules_for_export, original_batch, capture_stages=True
            )
            wrapper_forward, wrapper_stages = run_pointpillars_modules(
                modules_for_export,
                wrapper_batch,
                valid_pillar_count=valid_pillar_count,
                zero_padded_pillars_after_vfe=args.zero_padded_pillars_after_vfe,
                capture_stages=True,
            )
            original_pred, _ = model.post_processing(original_forward)
            wrapper_pred, _ = model.post_processing(wrapper_forward)

        stage_report = {
            "after_data_loading": {
                "original": summarize_batch_dict(original_batch, AUDIT_KEYS),
                "wrapper": summarize_batch_dict(wrapper_batch, AUDIT_KEYS),
            },
            "after_voxelization_or_preprocessing": {
                "original": summarize_batch_dict(original_batch, AUDIT_KEYS),
                "wrapper": summarize_batch_dict(wrapper_batch, AUDIT_KEYS),
            },
            "after_vfe": {
                "original": summarize_batch_dict(original_stages.get("after_vfe", {}), AUDIT_KEYS),
                "wrapper": summarize_batch_dict(wrapper_stages.get("after_vfe", {}), AUDIT_KEYS),
            },
            "after_scatter": {
                "original": summarize_batch_dict(original_stages.get("after_scatter", {}), AUDIT_KEYS),
                "wrapper": summarize_batch_dict(wrapper_stages.get("after_scatter", {}), AUDIT_KEYS),
            },
            "after_backbone": {
                "original": summarize_batch_dict(original_stages.get("after_backbone", {}), AUDIT_KEYS),
                "wrapper": summarize_batch_dict(wrapper_stages.get("after_backbone", {}), AUDIT_KEYS),
            },
            "after_dense_head": {
                "original": summarize_batch_dict(original_stages.get("after_dense_head", {}), AUDIT_KEYS),
                "wrapper": summarize_batch_dict(wrapper_stages.get("after_dense_head", {}), AUDIT_KEYS),
            },
            "before_post_processing": {
                "original": summarize_batch_dict(original_forward, AUDIT_KEYS),
                "wrapper": summarize_batch_dict(wrapper_forward, AUDIT_KEYS),
            },
            "after_post_processing": {
                "original": {
                    "pred_boxes": summarize_batch_dict({"pred_boxes": original_pred[0]["pred_boxes"]}, ["pred_boxes"])["pred_boxes"],
                    "pred_scores": summarize_batch_dict({"pred_scores": original_pred[0]["pred_scores"]}, ["pred_scores"])["pred_scores"],
                    "pred_labels": summarize_batch_dict({"pred_labels": original_pred[0]["pred_labels"]}, ["pred_labels"])["pred_labels"],
                },
                "wrapper": {
                    "pred_boxes": summarize_batch_dict({"pred_boxes": wrapper_pred[0]["pred_boxes"]}, ["pred_boxes"])["pred_boxes"],
                    "pred_scores": summarize_batch_dict({"pred_scores": wrapper_pred[0]["pred_scores"]}, ["pred_scores"])["pred_scores"],
                    "pred_labels": summarize_batch_dict({"pred_labels": wrapper_pred[0]["pred_labels"]}, ["pred_labels"])["pred_labels"],
                },
            },
        }
        rows.append(
            {
                "frame_id": frame_id,
                "valid_pillar_count": valid_pillar_count,
                "bucket_size": args.bucket_size,
                "padding_strategy": args.padding_strategy,
                "zero_padded_pillars_after_vfe": args.zero_padded_pillars_after_vfe,
                "padded_pillar_count": pad_stats["padded_pillar_count"],
                "dir_cls_present_after_dense_head_original": bool(original_stages.get("after_dense_head", {}).get("dir_cls_preds") is not None),
                "dir_cls_present_after_dense_head_wrapper": bool(wrapper_stages.get("after_dense_head", {}).get("dir_cls_preds") is not None),
                "cls_preds_normalized_original": original_forward.get("cls_preds_normalized"),
                "cls_preds_normalized_wrapper": wrapper_forward.get("cls_preds_normalized"),
                "original_box_count": int(original_pred[0]["pred_boxes"].shape[0]),
                "wrapper_box_count": int(wrapper_pred[0]["pred_boxes"].shape[0]),
                "stage_report": stage_report,
            }
        )

    payload = {
        "status": "completed",
        "frame_count": len(rows),
        "padding_strategy": args.padding_strategy,
        "zero_padded_pillars_after_vfe": args.zero_padded_pillars_after_vfe,
        "rows": rows,
        "global_findings": {
            "dir_cls_present_in_dense_head_stage_original": any(row["dir_cls_present_after_dense_head_original"] for row in rows),
            "dir_cls_present_in_dense_head_stage_wrapper": any(row["dir_cls_present_after_dense_head_wrapper"] for row in rows),
            "cls_preds_normalized_values": sorted({str(row["cls_preds_normalized_wrapper"]) for row in rows}),
            "wrapper_uses_native_post_processing": True,
            "suspected_contract_issue": (
                "padded pillars enter VFE/scatter as valid pillars unless explicitly zeroed after VFE"
                if not args.zero_padded_pillars_after_vfe
                else "valid-mask style zeroing after VFE prevents padded pillars from polluting scatter"
            ),
        },
    }
    write_json(report_json, payload)
    write_markdown(
        report_md,
        "# Wrapper Batch Dict Audit\n\n"
        f"- Frame count: `{payload['frame_count']}`\n"
        f"- Padding strategy: `{args.padding_strategy}`\n"
        f"- Zero padded pillars after VFE: `{args.zero_padded_pillars_after_vfe}`\n"
        f"- dir_cls present after dense head (original): `{payload['global_findings']['dir_cls_present_in_dense_head_stage_original']}`\n"
        f"- dir_cls present after dense head (wrapper): `{payload['global_findings']['dir_cls_present_in_dense_head_stage_wrapper']}`\n"
        f"- Native post_processing reused: `{payload['global_findings']['wrapper_uses_native_post_processing']}`\n"
        f"- Contract finding: `{payload['global_findings']['suspected_contract_issue']}`\n"
    )
    print(json.dumps({"status": "completed", "report": str(report_json)}, indent=2))


if __name__ == "__main__":
    main()
