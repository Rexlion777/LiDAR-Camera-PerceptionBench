from __future__ import annotations

import argparse
import io as sysio
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KITTI official eval from WSL.")
    parser.add_argument("--openpcdet-root", required=True, help="OpenPCDet root path on Linux/WSL.")
    parser.add_argument("--label-dir", required=True, help="KITTI label_2 directory on Linux/WSL.")
    parser.add_argument("--pred-dir", required=True, help="Prediction txt directory on Linux/WSL.")
    parser.add_argument("--split-file", required=True, help="KITTI split file on Linux/WSL.")
    parser.add_argument("--output-json", required=True, help="Output JSON path on Linux/WSL.")
    return parser.parse_args()


def _read_split_ids(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _corners_from_rbbox(rbbox: np.ndarray) -> np.ndarray:
    angle = float(rbbox[4])
    center_x = float(rbbox[0])
    center_y = float(rbbox[1])
    x_d = float(rbbox[2])
    y_d = float(rbbox[3])
    a_cos = np.cos(angle)
    a_sin = np.sin(angle)
    corners_x = np.array([-x_d / 2, -x_d / 2, x_d / 2, x_d / 2], dtype=np.float64)
    corners_y = np.array([-y_d / 2, y_d / 2, y_d / 2, -y_d / 2], dtype=np.float64)
    rotated = []
    for x_val, y_val in zip(corners_x, corners_y):
        rotated_x = a_cos * x_val + a_sin * y_val + center_x
        rotated_y = -a_sin * x_val + a_cos * y_val + center_y
        rotated.append((rotated_x, rotated_y))
    return np.asarray(rotated, dtype=np.float64)


def rotate_iou_cpu_eval(boxes: np.ndarray, query_boxes: np.ndarray, criterion: int = -1, device_id: int = 0) -> np.ndarray:
    del device_id
    boxes = np.asarray(boxes, dtype=np.float64)
    query_boxes = np.asarray(query_boxes, dtype=np.float64)
    overlaps = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float64)
    if boxes.shape[0] == 0 or query_boxes.shape[0] == 0:
        return overlaps

    box_polygons = [Polygon(_corners_from_rbbox(box)).buffer(0) for box in boxes]
    query_polygons = [Polygon(_corners_from_rbbox(box)).buffer(0) for box in query_boxes]
    box_areas = [poly.area for poly in box_polygons]
    query_areas = [poly.area for poly in query_polygons]

    for box_index, box_poly in enumerate(box_polygons):
        for query_index, query_poly in enumerate(query_polygons):
            intersection_area = box_poly.intersection(query_poly).area
            if criterion == -1:
                denominator = box_areas[box_index] + query_areas[query_index] - intersection_area
            elif criterion == 0:
                denominator = box_areas[box_index]
            elif criterion == 1:
                denominator = query_areas[query_index]
            else:
                denominator = 1.0
            overlaps[box_index, query_index] = 0.0 if denominator <= 0 else intersection_area / denominator
    return overlaps


def _load_eval_namespace(eval_path: Path) -> dict:
    source = eval_path.read_text(encoding="utf-8")
    source = source.replace("from .rotate_iou import rotate_iou_gpu_eval\n", "")
    namespace = {
        "__name__": "cpu_kitti_eval",
        "np": np,
        "numba": __import__("numba"),
        "sysio": sysio,
        "rotate_iou_gpu_eval": rotate_iou_cpu_eval,
    }
    exec(compile(source, str(eval_path), "exec"), namespace)
    return namespace


def _load_native_eval(openpcdet_root: Path):
    kitti_dataset_dir = openpcdet_root / "pcdet" / "datasets" / "kitti"
    sys.path.insert(0, str(kitti_dataset_dir))
    from kitti_object_eval_python import eval as kitti_eval

    return kitti_eval.get_official_eval_result


def _empty_anno() -> dict:
    return {
        "name": np.array([], dtype=object),
        "truncated": np.array([], dtype=np.float64),
        "occluded": np.array([], dtype=np.int64),
        "alpha": np.array([], dtype=np.float64),
        "bbox": np.zeros((0, 4), dtype=np.float64),
        "dimensions": np.zeros((0, 3), dtype=np.float64),
        "location": np.zeros((0, 3), dtype=np.float64),
        "rotation_y": np.array([], dtype=np.float64),
        "score": np.array([], dtype=np.float64),
    }


def _parse_kitti_label_file(path: Path, include_score: bool) -> dict:
    if not path.exists():
        return _empty_anno()
    names = []
    truncated = []
    occluded = []
    alpha = []
    bbox = []
    dimensions = []
    location = []
    rotation_y = []
    score = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if len(fields) < 15:
            continue
        names.append(fields[0])
        truncated.append(float(fields[1]))
        occluded.append(int(float(fields[2])))
        alpha.append(float(fields[3]))
        bbox.append([float(value) for value in fields[4:8]])
        dimensions.append([float(value) for value in fields[8:11]])
        location.append([float(value) for value in fields[11:14]])
        rotation_y.append(float(fields[14]))
        score.append(float(fields[15]) if include_score and len(fields) > 15 else 0.0)
    if not names:
        return _empty_anno()
    return {
        "name": np.asarray(names),
        "truncated": np.asarray(truncated, dtype=np.float64),
        "occluded": np.asarray(occluded, dtype=np.int64),
        "alpha": np.asarray(alpha, dtype=np.float64),
        "bbox": np.asarray(bbox, dtype=np.float64),
        "dimensions": np.asarray(dimensions, dtype=np.float64),
        "location": np.asarray(location, dtype=np.float64),
        "rotation_y": np.asarray(rotation_y, dtype=np.float64),
        "score": np.asarray(score, dtype=np.float64),
    }


def _load_label_annos(label_dir: Path, sample_ids: list[int], include_score: bool) -> list[dict]:
    return [_parse_kitti_label_file(label_dir / f"{sample_id:06d}.txt", include_score=include_score) for sample_id in sample_ids]


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
    label_dir = Path(args.label_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()
    split_file = Path(args.split_file).resolve()
    output_json = Path(args.output_json).resolve()

    sample_ids = _read_split_ids(split_file)
    gt_annos = _load_label_annos(label_dir, sample_ids, include_score=False)
    dt_annos = _load_label_annos(pred_dir, sample_ids, include_score=True)
    evaluation_backend = "openpcdet_native_numba_cuda"
    fallback_reason = None
    try:
        get_official_eval_result = _load_native_eval(openpcdet_root)
        result_text, result_dict = get_official_eval_result(gt_annos, dt_annos, ["Car", "Pedestrian", "Cyclist"])
    except Exception as exc:
        evaluation_backend = "openpcdet_eval_cpu_polygon_fallback"
        fallback_reason = f"{type(exc).__name__}: {exc}"
        kitti_eval_dir = openpcdet_root / "pcdet" / "datasets" / "kitti" / "kitti_object_eval_python"
        eval_namespace = _load_eval_namespace(kitti_eval_dir / "eval.py")
        get_official_eval_result = eval_namespace["get_official_eval_result"]
        result_text, result_dict = get_official_eval_result(gt_annos, dt_annos, ["Car", "Pedestrian", "Cyclist"])

    payload = {
        "status": "completed",
        "label_dir": str(label_dir),
        "pred_dir": str(pred_dir),
        "split_file": str(split_file),
        "frame_count": len(sample_ids),
        "evaluation_backend": evaluation_backend,
        "fallback_reason": fallback_reason,
        "fallback_traceback": traceback.format_exc() if fallback_reason else None,
        "official_result_text": result_text,
        "official_result_dict": to_jsonable(result_dict),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "frame_count": len(sample_ids)}, indent=2))


if __name__ == "__main__":
    main()
