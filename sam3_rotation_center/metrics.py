from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MaskMetrics:
    raw_count: int
    nms_count: int
    high_count: int
    median_score: float
    top10_mean_score: float
    min_area: int
    median_area: float
    max_area: int
    union_area: int
    summed_area: int
    overlap_excess_fraction: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys = np.flatnonzero(mask.any(axis=1))
    xs = np.flatnonzero(mask.any(axis=0))
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pair_iou(
    masks: np.ndarray,
    areas: np.ndarray,
    bounds: list[tuple[int, int, int, int] | None],
    first: int,
    second: int,
) -> float:
    a = bounds[first]
    b = bounds[second]
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    intersection = np.logical_and(
        masks[first, y0:y1, x0:x1],
        masks[second, y0:y1, x0:x1],
    ).sum()
    union = int(areas[first] + areas[second] - intersection)
    return float(intersection / max(union, 1))


def mask_nms(
    masks: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    min_area: int = 0,
    max_area: int | None = None,
) -> list[int]:
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape [N,H,W], got {masks.shape}")
    scores = np.asarray(scores, dtype=np.float32)
    if len(masks) != len(scores):
        raise ValueError("masks and scores must contain the same number of items")
    if len(masks) == 0:
        return []

    geometry = MaskGeometry(masks)
    return geometry.nms(scores, iou_threshold, min_area, max_area)


class MaskGeometry:
    """Reuse full-resolution geometry across confidence thresholds, exactly."""

    def __init__(self, masks: np.ndarray):
        self.masks = masks
        self.areas = masks.sum(axis=(1, 2), dtype=np.int64)
        self.bounds = [mask_bounds(mask) for mask in masks]
        self.overlaps = {}

    def nms(self, scores, iou_threshold=0.5, min_area=0, max_area=None, indices=None):
        scores = np.asarray(scores, dtype=np.float32)
        if indices is None:
            indices = np.arange(len(scores))
        # Match the legacy per-threshold sort, including equal-score ties.
        candidates = [int(indices[i]) for i in np.argsort(scores[indices])[::-1]
                      if self.areas[indices[i]] >= min_area
                      and (max_area is None or self.areas[indices[i]] <= max_area)]
        kept = []
        for index in candidates:
            suppress = False
            for other in kept:
                key = (min(index, other), max(index, other))
                if key not in self.overlaps:
                    self.overlaps[key] = pair_iou(self.masks, self.areas, self.bounds, index, other)
                if self.overlaps[key] >= iou_threshold:
                    suppress = True
                    break
            if not suppress:
                kept.append(index)
        return kept


def summarize_prepared(geometry, scores, kept, high_score_threshold=0.7):
    masks, areas = geometry.masks, geometry.areas
    scores = np.asarray(scores, dtype=np.float32)
    selected_scores = scores[kept] if kept else np.empty(0, dtype=np.float32)
    selected_areas = areas[kept] if kept else np.empty(0, dtype=np.int64)
    union = np.zeros(masks.shape[1:], dtype=bool)
    for index in kept:
        np.logical_or(union, masks[index], out=union)
    union_area = int(union.sum())
    summed_area = int(selected_areas.sum())
    top10 = np.sort(selected_scores)[-10:]
    return MaskMetrics(
        raw_count=len(masks), nms_count=len(kept),
        high_count=int((selected_scores >= high_score_threshold).sum()),
        median_score=float(np.median(selected_scores)) if selected_scores.size else 0.0,
        top10_mean_score=float(top10.mean()) if selected_scores.size else 0.0,
        min_area=int(selected_areas.min()) if selected_areas.size else 0,
        median_area=float(np.median(selected_areas)) if selected_areas.size else 0.0,
        max_area=int(selected_areas.max()) if selected_areas.size else 0,
        union_area=union_area, summed_area=summed_area,
        overlap_excess_fraction=float(1-union_area/summed_area) if summed_area else 0.0)


def summarize_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    nms_iou: float = 0.5,
    high_score_threshold: float = 0.7,
    min_area: int = 0,
    max_area: int | None = None,
) -> tuple[MaskMetrics, list[int]]:
    geometry = MaskGeometry(masks)
    kept = geometry.nms(scores, nms_iou, min_area, max_area)
    return summarize_prepared(geometry, scores, kept, high_score_threshold), kept


def centered_rolling_mean(values: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        result[index] = values[start:stop].mean()
    return result
