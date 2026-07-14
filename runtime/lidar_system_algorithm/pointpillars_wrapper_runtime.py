from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


POINTPILLARS_STAGE_NAMES = (
    "after_vfe",
    "after_scatter",
    "after_backbone",
    "after_dense_head",
)


def make_valid_pillar_mask(valid_count: int, total_count: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((int(total_count),), dtype=torch.bool, device=device)
    mask[: int(valid_count)] = True
    return mask


def zero_invalid_pillar_features(batch_dict: dict, valid_pillar_count: int) -> dict:
    pillar_features = batch_dict.get("pillar_features")
    if pillar_features is None:
        return batch_dict
    total_count = int(pillar_features.shape[0])
    if valid_pillar_count >= total_count:
        batch_dict["valid_pillar_mask"] = make_valid_pillar_mask(valid_pillar_count, total_count, pillar_features.device)
        return batch_dict
    features = pillar_features.clone()
    features[int(valid_pillar_count) :] = 0
    batch_dict["pillar_features"] = features
    batch_dict["valid_pillar_mask"] = make_valid_pillar_mask(valid_pillar_count, total_count, pillar_features.device)
    return batch_dict


def tensor_stat_summary(value) -> dict:
    if value is None:
        return {"present": False}
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        finite = np.isfinite(array)
        finite_values = array[finite]
        return {
            "present": True,
            "type": "tensor",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": float(finite_values.min()) if finite_values.size else None,
            "max": float(finite_values.max()) if finite_values.size else None,
            "mean": float(finite_values.mean()) if finite_values.size else None,
            "std": float(finite_values.std()) if finite_values.size else None,
            "nan_count": int(np.isnan(array).sum()),
            "inf_count": int(np.isinf(array).sum()),
            "zero_ratio": float(np.mean(array == 0)) if array.size else 0.0,
        }
    if isinstance(value, (list, tuple)):
        return {"present": True, "type": type(value).__name__, "length": len(value)}
    return {"present": True, "type": type(value).__name__, "value": str(value)}


def summarize_batch_dict(batch_dict: dict, keys: Iterable[str]) -> dict[str, dict]:
    return {key: tensor_stat_summary(batch_dict.get(key)) for key in keys}


def run_pointpillars_modules(
    modules: Iterable[torch.nn.Module],
    batch_dict: dict,
    *,
    valid_pillar_count: int | None = None,
    zero_padded_pillars_after_vfe: bool = False,
    capture_stages: bool = False,
) -> tuple[dict, dict[str, dict]]:
    stage_snapshots: dict[str, dict] = {}
    current = dict(batch_dict)
    for index, module in enumerate(modules):
        current = module(current)
        if zero_padded_pillars_after_vfe and index == 0 and valid_pillar_count is not None:
            current = zero_invalid_pillar_features(current, valid_pillar_count)
        if capture_stages and index < len(POINTPILLARS_STAGE_NAMES):
            stage_snapshots[POINTPILLARS_STAGE_NAMES[index]] = dict(current)
    return current, stage_snapshots
