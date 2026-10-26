from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import normalize_reconstruction
from .inference import Sam3ParticleSegmenter
from .metrics import mask_bounds, mask_nms, pair_iou
from .visualization import save_overlay


def inspect_image(
    image_path: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    prompt: str = "particle",
    confidence_threshold: float = 0.2,
    nms_iou: float = 0.5,
    high_score_threshold: float = 0.7,
    min_area: int = 0,
    max_area: int | None = None,
    device: str = "cuda",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    image, normalization = normalize_reconstruction(image_path)
    segmenter = Sam3ParticleSegmenter(
        checkpoint,
        prompt=prompt,
        confidence_threshold=confidence_threshold,
        device=device,
    )
    prediction = segmenter.predict(image)
    masks = prediction.masks
    scores = prediction.scores

    if len(masks):
        areas = masks.reshape(len(masks), -1).sum(axis=1, dtype=np.int64)
        bounds = [mask_bounds(mask) for mask in masks]
        overlap_pairs = []
        for first in range(len(masks)):
            for second in range(first + 1, len(masks)):
                iou = pair_iou(masks, areas, bounds, first, second)
                if iou >= nms_iou:
                    overlap_pairs.append(
                        {
                            "first": first,
                            "second": second,
                            "iou": iou,
                            "first_score": float(scores[first]),
                            "second_score": float(scores[second]),
                        }
                    )
        overlap_pairs.sort(key=lambda item: item["iou"], reverse=True)
        q25, median, q75 = np.quantile(areas, [0.25, 0.5, 0.75])
        large_limit = float(q75 + 1.5 * (q75 - q25))
        large = np.flatnonzero(areas > large_limit)
    else:
        areas = np.empty(0, dtype=np.int64)
        overlap_pairs = []
        q25 = median = q75 = large_limit = 0.0
        large = np.empty(0, dtype=np.int64)

    kept = mask_nms(
        masks,
        scores,
        iou_threshold=nms_iou,
        min_area=min_area,
        max_area=max_area,
    )
    report = {
        "image": str(image_path),
        "checkpoint": str(checkpoint),
        "prompt": prompt,
        "confidence_threshold": confidence_threshold,
        "nms_iou": nms_iou,
        "normalization": normalization,
        "image_shape": [image.height, image.width],
        "raw_count": len(masks),
        "nms_count": len(kept),
        "high_count_after_nms": int(
            (scores[kept] >= high_score_threshold).sum()
        )
        if kept
        else 0,
        "area_pixels": {
            "min": int(areas.min()) if areas.size else 0,
            "q25": float(q25),
            "median": float(median),
            "q75": float(q75),
            "q95": float(np.quantile(areas, 0.95)) if areas.size else 0.0,
            "max": int(areas.max()) if areas.size else 0,
            "mean": float(areas.mean()) if areas.size else 0.0,
        },
        "max_image_fraction": float(
            areas.max() / (image.height * image.width)
        )
        if areas.size
        else 0.0,
        "iqr_large_threshold": large_limit,
        "large_masks": [
            {
                "index": int(index),
                "area": int(areas[index]),
                "score": float(scores[index]),
            }
            for index in large[np.argsort(areas[large])[::-1]]
        ],
        "overlap_pair_count": len(overlap_pairs),
        "top_overlap_pairs": overlap_pairs[:50],
    }
    image.save(output_dir / "input.png")
    save_overlay(image, masks, list(range(len(masks))), output_dir / "raw_overlay.png")
    save_overlay(image, masks, kept, output_dir / "nms_overlay.png")
    (output_dir / "mask_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
