from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(fn, *args, **kwargs):
    cuda_sync()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    cuda_sync()
    return result, (time.perf_counter() - start) * 1000.0


class NmsTimer:
    def __init__(self) -> None:
        self.total_ms = 0.0

    def reset(self) -> None:
        self.total_ms = 0.0

    def wrap(self, fn):
        def wrapped(*args, **kwargs):
            cuda_sync()
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            cuda_sync()
            self.total_ms += (time.perf_counter() - start) * 1000.0
            return result

        return wrapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenPCDet PointPillars runtime and emit JSON timings.")
    parser.add_argument("--openpcdet-root", required=True, help="Path to OpenPCDet source tree.")
    parser.add_argument("--cfg-file", required=True, help="OpenPCDet config path.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--input-json", required=True, help="JSON file containing frame records.")
    parser.add_argument("--output-json", required=True, help="Output JSON path.")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup frame count.")
    return parser.parse_args()


def get_processor_name(processor) -> str:
    func = processor.func if isinstance(processor, partial) else processor
    return getattr(func, "__name__", str(func))


def load_frames(input_json: Path) -> list[dict]:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("input-json must contain a frames list")
    return frames


def main() -> None:
    args = parse_args()
    openpcdet_root = Path(args.openpcdet_root).resolve()
    sys.path.insert(0, str(openpcdet_root))
    sys.path.insert(0, str(openpcdet_root / "tools"))
    os.chdir(str(openpcdet_root / "tools"))

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.models.model_utils import model_nms_utils
    from pcdet.utils import common_utils

    cfg_from_yaml_file(str(Path(args.cfg_file).resolve()), cfg)

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

        def prepare_data_timed(self, points: np.ndarray, frame_id: str) -> tuple[dict, dict]:
            data_dict = {
                "points": points,
                "frame_id": frame_id,
            }
            point_preprocess_ms = 0.0
            voxelization_ms = 0.0

            data_dict = self.set_lidar_aug_matrix(data_dict)
            data_dict, elapsed = timed_call(self.point_feature_encoder.forward, data_dict)
            point_preprocess_ms += elapsed

            for processor in self.data_processor.data_processor_queue:
                processor_name = get_processor_name(processor)
                data_dict, elapsed = timed_call(processor, data_dict=data_dict)
                if processor_name == "transform_points_to_voxels":
                    voxelization_ms += elapsed
                else:
                    point_preprocess_ms += elapsed

            data_dict.pop("gt_names", None)
            return data_dict, {
                "point_preprocess_ms": point_preprocess_ms,
                "voxelization_or_pillarization_ms": voxelization_ms,
            }

    logger = common_utils.create_logger()
    frames = load_frames(Path(args.input_json).resolve())
    dataset = RuntimeDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=openpcdet_root,
        logger=logger,
    )

    model, model_build_ms = timed_call(build_network, model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    _, checkpoint_load_ms = timed_call(model.load_params_from_file, filename=str(Path(args.ckpt).resolve()), logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    nms_timer = NmsTimer()
    original_class_agnostic_nms = model_nms_utils.class_agnostic_nms
    original_multi_classes_nms = model_nms_utils.multi_classes_nms
    model_nms_utils.class_agnostic_nms = nms_timer.wrap(original_class_agnostic_nms)
    model_nms_utils.multi_classes_nms = nms_timer.wrap(original_multi_classes_nms)

    module_names = [
        ("vfe", getattr(model, "vfe", None)),
        ("backbone_3d", getattr(model, "backbone_3d", None)),
        ("map_to_bev_module", getattr(model, "map_to_bev_module", None)),
        ("pfe", getattr(model, "pfe", None)),
        ("backbone_2d", getattr(model, "backbone_2d", None)),
        ("dense_head", getattr(model, "dense_head", None)),
        ("point_head", getattr(model, "point_head", None)),
        ("roi_head", getattr(model, "roi_head", None)),
    ]
    module_names = [(name, module) for name, module in module_names if module is not None]

    measured_records: list[dict] = []
    warmup_records: list[dict] = []
    measured_detections: list[dict] = []

    with torch.no_grad():
        for run_index, frame in enumerate(frames):
            frame_id = str(frame["frame_id"])
            lidar_path = Path(frame["lidar_path"])

            raw_points, data_load_ms = timed_call(np.fromfile, str(lidar_path), np.float32)
            points = raw_points.reshape(-1, 4)
            prepared, prep_timings = dataset.prepare_data_timed(points, frame_id)
            pillar_count = int(prepared["voxels"].shape[0]) if "voxels" in prepared else 0
            detected_box_count = 0

            batch_dict, collate_ms = timed_call(dataset.collate_batch, [prepared])
            _, gpu_transfer_ms = timed_call(load_data_to_gpu, batch_dict)
            point_preprocess_ms = prep_timings["point_preprocess_ms"] + collate_ms + gpu_transfer_ms

            nms_timer.reset()
            module_times = {}
            for module_name, module in module_names:
                batch_dict, elapsed = timed_call(module, batch_dict)
                module_times[module_name] = elapsed

            (pred_dicts, _), postprocess_total_ms = timed_call(model.post_processing, batch_dict)
            nms_ms = nms_timer.total_ms
            postprocess_ms = max(0.0, postprocess_total_ms - nms_ms)

            model_forward_ms = float(sum(module_times.values()))
            backbone_ms = float(
                module_times.get("backbone_3d", 0.0)
                + module_times.get("map_to_bev_module", 0.0)
                + module_times.get("backbone_2d", 0.0)
            )
            head_ms = float(
                module_times.get("dense_head", 0.0)
                + module_times.get("point_head", 0.0)
                + module_times.get("roi_head", 0.0)
            )
            detections = []
            scores = pred_dicts[0]["pred_scores"].detach().cpu().tolist()
            boxes = pred_dicts[0]["pred_boxes"].detach().cpu().tolist()
            labels = pred_dicts[0]["pred_labels"].detach().cpu().tolist()
            detected_box_count = len(boxes)
            for index, box in enumerate(boxes):
                class_id = int(labels[index]) if index < len(labels) else 0
                class_name = cfg.CLASS_NAMES[class_id - 1] if 0 < class_id <= len(cfg.CLASS_NAMES) else f"class_{class_id}"
                score = float(scores[index]) if index < len(scores) else None
                detections.append(
                    {
                        "frame_id": frame_id,
                        "object_id": index,
                        "class_id": class_id,
                        "class_name": class_name,
                        "score": score,
                        "box_3d_lidar": {
                            "x": float(box[0]),
                            "y": float(box[1]),
                            "z": float(box[2]),
                            "dx": float(box[3]),
                            "dy": float(box[4]),
                            "dz": float(box[5]),
                            "heading": float(box[6]),
                        },
                    }
                )

            record = {
                "frame_id": frame_id,
                "lidar_path": str(lidar_path),
                "data_load_ms": data_load_ms,
                "point_preprocess_ms": point_preprocess_ms,
                "voxelization_or_pillarization_ms": prep_timings["voxelization_or_pillarization_ms"],
                "model_forward_ms": model_forward_ms,
                "backbone_ms": backbone_ms,
                "head_ms": head_ms,
                "nms_ms": nms_ms,
                "postprocess_ms": postprocess_ms,
                "pillar_count": pillar_count,
                "detected_box_count": detected_box_count,
                "score_distribution": [float(value) for value in scores],
                "module_times": module_times,
            }

            if run_index < args.warmup:
                warmup_records.append(record)
            else:
                measured_records.append(record)
                measured_detections.extend(detections)

    model_nms_utils.class_agnostic_nms = original_class_agnostic_nms
    model_nms_utils.multi_classes_nms = original_multi_classes_nms

    payload = {
        "openpcdet_root": str(openpcdet_root),
        "cfg_file": str(Path(args.cfg_file).resolve()),
        "ckpt": str(Path(args.ckpt).resolve()),
        "warmup": args.warmup,
        "model_build_ms": model_build_ms,
        "checkpoint_load_ms": checkpoint_load_ms,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "warmup_records": warmup_records,
        "measured_records": measured_records,
        "detections": measured_detections,
    }
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_json": str(output_path), "measured_frames": len(measured_records)}, indent=2))


if __name__ == "__main__":
    main()
