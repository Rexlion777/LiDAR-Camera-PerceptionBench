from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KITTI official eval with PointPillars in the OpenPCDet runtime.")
    parser.add_argument("--openpcdet-root", required=True, help="Path to the OpenPCDet source tree.")
    parser.add_argument("--cfg-file", required=True, help="PointPillars config path.")
    parser.add_argument("--ckpt", required=True, help="PointPillars checkpoint path.")
    parser.add_argument("--kitti-root", required=True, help="KITTI dataset root containing training/velodyne.")
    parser.add_argument("--split-file", required=True, help="KITTI split file such as val.txt.")
    parser.add_argument("--output-json", required=True, help="Output JSON path.")
    parser.add_argument("--pred-dir", required=True, help="Directory for KITTI prediction txt files.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for smoke tests. 0 means full split.")
    return parser.parse_args()


def read_image_shape(image_path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {image_path}")
    return np.array(image.shape[:2], dtype=np.int32)


def load_split_ids(split_file: Path, max_frames: int) -> list[str]:
    sample_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_frames > 0:
        sample_ids = sample_ids[:max_frames]
    return sample_ids


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    openpcdet_root = Path(args.openpcdet_root).resolve()
    cfg_file = Path(args.cfg_file).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    kitti_root = Path(args.kitti_root).resolve()
    split_file = Path(args.split_file).resolve()
    output_json = Path(args.output_json).resolve()
    pred_dir = Path(args.pred_dir).resolve()

    sys.path.insert(0, str(openpcdet_root))
    sys.path.insert(0, str(openpcdet_root / "tools"))
    os.chdir(str(openpcdet_root))

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import DatasetTemplate
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import calibration_kitti, common_utils

    cfg_from_yaml_file(str(cfg_file), cfg)

    class RuntimeDataset(DatasetTemplate):
        def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None):
            super().__init__(
                dataset_cfg=dataset_cfg,
                class_names=class_names,
                training=training,
                root_path=root_path,
                logger=logger,
            )

        def __len__(self):
            return 0

        def __getitem__(self, index):
            raise IndexError("RuntimeDataset does not support indexing")

    logger = common_utils.create_logger()
    dataset = RuntimeDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=openpcdet_root,
        logger=logger,
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    original_torch_load = torch.load

    def trusted_checkpoint_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = trusted_checkpoint_load
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=True)
    torch.load = original_torch_load
    model.cuda()
    model.eval()

    root_split_path = kitti_root / "training"
    label_dir = root_split_path / "label_2"
    pred_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = load_split_ids(split_file, args.max_frames)
    per_frame: list[dict] = []
    det_annos = []
    total_detections = 0
    inference_times_ms: list[float] = []
    total_start = time.perf_counter()

    with torch.no_grad():
        for sample_idx in sample_ids:
            lidar_path = root_split_path / "velodyne" / f"{sample_idx}.bin"
            image_path = root_split_path / "image_2" / f"{sample_idx}.png"
            calib_path = root_split_path / "calib" / f"{sample_idx}.txt"

            points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)
            input_dict = {
                "points": points,
                "frame_id": sample_idx,
            }
            data_dict = dataset.prepare_data(data_dict=input_dict)
            batch_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(batch_dict)

            torch.cuda.synchronize()
            start = time.perf_counter()
            pred_dicts, _ = model.forward(batch_dict)
            torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - start) * 1000.0
            inference_times_ms.append(inference_ms)

            calib = calibration_kitti.Calibration(calib_path)
            image_shape = read_image_shape(image_path)
            prediction_batch = {
                "frame_id": [sample_idx],
                "calib": [calib],
                "image_shape": torch.from_numpy(np.expand_dims(image_shape, axis=0)),
            }
            annos = KittiDataset.generate_prediction_dicts(
                batch_dict=prediction_batch,
                pred_dicts=pred_dicts,
                class_names=cfg.CLASS_NAMES,
                output_path=pred_dir,
            )
            det_annos.extend(annos)

            det_count = int(len(annos[0]["name"])) if annos else 0
            total_detections += det_count
            score_values = annos[0]["score"].tolist() if annos and "score" in annos[0] else []
            per_frame.append(
                {
                    "frame_id": sample_idx,
                    "inference_ms": inference_ms,
                    "detection_count": det_count,
                    "score_mean": float(np.mean(score_values)) if score_values else None,
                    "score_max": float(np.max(score_values)) if score_values else None,
                }
            )

    total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

    payload = {
        "status": "completed",
        "openpcdet_root": str(openpcdet_root),
        "cfg_file": str(cfg_file),
        "ckpt": str(ckpt_path),
        "kitti_root": str(kitti_root),
        "split_file": str(split_file),
        "prediction_dir": str(pred_dir),
        "frame_count": len(sample_ids),
        "total_detections": total_detections,
        "total_elapsed_ms": total_elapsed_ms,
        "mean_inference_ms": float(np.mean(inference_times_ms)) if inference_times_ms else None,
        "p50_inference_ms": float(np.percentile(inference_times_ms, 50)) if inference_times_ms else None,
        "p95_inference_ms": float(np.percentile(inference_times_ms, 95)) if inference_times_ms else None,
        "per_frame": per_frame,
        "label_dir": str(label_dir),
        "sample_ids": sample_ids,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "frame_count": len(sample_ids), "prediction_dir": str(pred_dir)}, indent=2))


if __name__ == "__main__":
    main()
