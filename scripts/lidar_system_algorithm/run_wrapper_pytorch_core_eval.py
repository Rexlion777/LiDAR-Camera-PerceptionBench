from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.kitti_io import list_frame_ids, resolve_frame_assets
from runtime.lidar_system_algorithm.pointpillars_wrapper_runtime import run_pointpillars_modules
from runtime.lidar_system_algorithm.report_schema import write_json, write_markdown
from runtime.lidar_system_algorithm.tensorrt_bucketed_wrapper import parse_bucket_sizes, select_bucket_size
from runtime.lidar_system_algorithm.tensorrt_debug_utils import pad_prepared_inputs, prediction_dir_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run wrapper_pytorch_core official eval with bucketed padded inputs.")
    parser.add_argument("--kitti-root", default="data/kitti_object_raw/extracted")
    parser.add_argument("--openpcdet-root", default="external/OpenPCDet")
    parser.add_argument("--cfg-file", default="external/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml")
    parser.add_argument("--ckpt", default="checkpoints/pointpillar_kitti.pth")
    parser.add_argument("--bucket-sizes", default="4096,8192,12000,16000,20000")
    parser.add_argument("--padding-strategy", default="repeat_first_valid", choices=["repeat_first_valid", "duplicate_zero_coord_padding", "unique_dummy_coord_padding"])
    parser.add_argument("--zero-padded-pillars-after-vfe", action="store_true", help="Zero out padded pillar features after VFE to keep padded cells from polluting scatter.")
    parser.add_argument("--frames", type=int, default=20, help="Frames for local sanity profiling rows.")
    parser.add_argument("--eval-max-frames", type=int, default=200, help="Frames for official eval slice. 0 means full split.")
    parser.add_argument("--split-file", default="external/OpenPCDet/data/kitti/ImageSets/val.txt")
    parser.add_argument("--pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt_wrapper_pytorch_core")
    parser.add_argument("--baseline-pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm")
    parser.add_argument("--report-suffix", default="", help="Optional suffix for report filenames, for example _v2.")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _read_split_ids(path: Path, max_frames: int) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids[:max_frames] if max_frames > 0 else ids


def _read_image_shape(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return np.array(image.shape[:2], dtype=np.int32)


def main() -> None:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    pred_dir = _resolve(args.pred_dir)
    baseline_pred_dir = _resolve(args.baseline_pred_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.report_suffix
    report_json = output_dir / f"wrapper_pytorch_core_eval{suffix}.json"
    report_md = output_dir / f"wrapper_pytorch_core_eval{suffix}.md"
    audit_json = output_dir / f"wrapper_pytorch_core_prediction_audit{suffix}.json"

    try:
        import torch
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.datasets.kitti.kitti_dataset import KittiDataset
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import calibration_kitti, common_utils
    except Exception as exc:
        payload = {"status": "skipped", "reason": f"runtime import failed: {type(exc).__name__}: {exc}"}
        write_json(report_json, payload)
        write_markdown(report_md, f"# Wrapper PyTorch Core Eval\n\n- Status: `skipped`\n- Reason: `{payload['reason']}`\n")
        return

    kitti_root = _resolve(args.kitti_root)
    openpcdet_root = _resolve(args.openpcdet_root)
    cfg_file = _resolve(args.cfg_file)
    split_file = _resolve(args.split_file)
    bucket_sizes = parse_bucket_sizes(args.bucket_sizes)
    ckpt_path = Path(args.ckpt).expanduser()

    os.chdir(openpcdet_root / "tools")
    cfg_from_yaml_file(str(cfg_file), cfg)

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
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=True)
    model.cuda().eval()
    modules_for_export = torch.nn.ModuleList(list(model.module_list)).cuda().eval()

    sample_ids = _read_split_ids(split_file, args.eval_max_frames)
    temp_split = output_dir / f"wrapper_pytorch_core_eval_split_{len(sample_ids)}.txt"
    temp_split.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
    frame_rows = []
    with torch.no_grad():
        for frame_id in sample_ids:
            assets = resolve_frame_assets(kitti_root, frame_id)
            points = np.fromfile(str(assets.lidar_path), dtype=np.float32).reshape(-1, 4)
            prepared = dataset.prepare_runtime(points, frame_id)
            full_pillar_count = int(prepared["voxels"].shape[0])
            bucket_size = select_bucket_size(full_pillar_count, bucket_sizes)
            padded, pad_stats = pad_prepared_inputs(prepared, bucket_size, args.padding_strategy)
            batch_dict = dataset.collate_batch([padded])
            load_data_to_gpu(batch_dict)

            py_batch, _ = run_pointpillars_modules(
                modules_for_export,
                batch_dict,
                valid_pillar_count=full_pillar_count,
                zero_padded_pillars_after_vfe=args.zero_padded_pillars_after_vfe,
                capture_stages=False,
            )
            py_batch["cls_preds_normalized"] = False
            pred_dicts, _ = model.post_processing(py_batch)

            calib = calibration_kitti.Calibration(assets.calib_path)
            image_shape = _read_image_shape(assets.image_path)
            prediction_batch = {
                "frame_id": [frame_id],
                "calib": [calib],
                "image_shape": torch.from_numpy(np.expand_dims(image_shape, axis=0)),
            }
            annos = KittiDataset.generate_prediction_dicts(
                batch_dict=prediction_batch,
                pred_dicts=pred_dicts,
                class_names=cfg.CLASS_NAMES,
                output_path=pred_dir,
            )
            score_values = annos[0]["score"] if annos else []
            frame_rows.append(
                {
                    "frame_id": frame_id,
                    "full_pillar_count": full_pillar_count,
                    "bucket_size": bucket_size,
                    "padding_strategy": args.padding_strategy,
                    "zero_padded_pillars_after_vfe": args.zero_padded_pillars_after_vfe,
                    "padded_pillar_count": pad_stats["padded_pillar_count"],
                    "box_count": int(len(annos[0]["name"])) if annos else 0,
                    "score_mean": float(np.mean(score_values)) if len(score_values) else None,
                    "score_max": float(np.max(score_values)) if len(score_values) else None,
                }
            )

    audit_payload = prediction_dir_stats(pred_dir)
    write_json(audit_json, audit_payload)

    runner = PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "wsl_kitti_eval_runner.py"
    raw_eval_json = output_dir / "wrapper_pytorch_core_eval_raw.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--openpcdet-root",
            str(openpcdet_root),
            "--label-dir",
            str((kitti_root / "training" / "label_2").resolve()),
            "--pred-dir",
            str(pred_dir),
            "--split-file",
            str(temp_split),
            "--output-json",
            str(raw_eval_json),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(PROJECT_ROOT),
    )

    baseline_slice_json = output_dir / "wrapper_pytorch_core_eval_baseline_slice.json"
    baseline_completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--openpcdet-root",
            str(openpcdet_root),
            "--label-dir",
            str((kitti_root / "training" / "label_2").resolve()),
            "--pred-dir",
            str(baseline_pred_dir),
            "--split-file",
            str(temp_split),
            "--output-json",
            str(baseline_slice_json),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(PROJECT_ROOT),
    )
    baseline = json.loads(baseline_slice_json.read_text(encoding="utf-8")) if (baseline_completed.returncode == 0 and baseline_slice_json.exists()) else {}
    baseline_dict = baseline.get("official_result_dict", {}) if isinstance(baseline, dict) else {}
    if completed.returncode == 0 and raw_eval_json.exists():
        raw_eval = json.loads(raw_eval_json.read_text(encoding="utf-8"))
        result_dict = raw_eval.get("official_result_dict", {})
        payload = {
            "status": "completed",
            "frame_count": len(sample_ids),
            "padding_strategy": args.padding_strategy,
            "zero_padded_pillars_after_vfe": args.zero_padded_pillars_after_vfe,
            "bucket_sizes": bucket_sizes,
            "prediction_dir": str(pred_dir),
            **audit_payload,
            "official_eval_status": raw_eval.get("status"),
            "evaluation_backend": raw_eval.get("evaluation_backend"),
            "official_result_dict": result_dict,
            "baseline_slice_official_result_dict": baseline_dict,
            "ap_delta_vs_openpcdet_original": {
                key: (result_dict.get(key) - baseline_dict.get(key)) if (result_dict.get(key) is not None and baseline_dict.get(key) is not None) else None
                for key in ["Car_3d/moderate_R40", "Pedestrian_3d/moderate_R40", "Cyclist_3d/moderate_R40"]
            },
            "frame_rows_preview": frame_rows[:20],
            "blocker": None,
        }
    else:
        payload = {
            "status": "skipped",
            "frame_count": len(sample_ids),
            "prediction_dir": str(pred_dir),
            **audit_payload,
            "official_eval_status": "failed",
            "blocker": f"wrapper_pytorch_core official eval failed: {completed.stderr.strip() or completed.stdout.strip()}",
        }

    write_json(report_json, payload)
    write_markdown(
        report_md,
        "# Wrapper PyTorch Core Eval\n\n"
        f"- Status: `{payload['status']}`\n"
        f"- Prediction files: `{payload.get('prediction_file_count')}`\n"
        f"- Empty prediction files: `{payload.get('empty_prediction_file_count')}`\n"
        f"- Total boxes: `{payload.get('total_box_count')}`\n"
        f"- Car moderate 3D AP_R40: `{payload.get('official_result_dict', {}).get('Car_3d/moderate_R40')}`\n"
        f"- Pedestrian moderate 3D AP_R40: `{payload.get('official_result_dict', {}).get('Pedestrian_3d/moderate_R40')}`\n"
        f"- Cyclist moderate 3D AP_R40: `{payload.get('official_result_dict', {}).get('Cyclist_3d/moderate_R40')}`\n"
        f"- Blocker: `{payload.get('blocker')}`\n",
    )
    print(json.dumps({"status": payload["status"], "report": str(report_json)}, indent=2))


if __name__ == "__main__":
    main()
