from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import DEFAULT_CENTER_REGEX, discover_images, normalize_reconstruction
from .inference import Sam3ParticleSegmenter
from .metrics import centered_rolling_mean, mask_nms, summarize_masks
from .visualization import save_count_curve, save_overlay
from .scoring import score_prediction, assign_scores


@dataclass
class SearchConfig:
    dataset_dir: Path
    checkpoint: Path
    output_dir: Path
    prompt: str = "particle"
    confidence_threshold: float = 0.2
    high_score_threshold: float = 0.7
    nms_iou: float = 0.5
    min_area: int = 0
    max_area: int | None = None
    rolling_radius: int = 5
    glob_pattern: str = "*.png"
    center_regex: str = DEFAULT_CENTER_REGEX
    step: int = 1
    device: str = "cuda"
    thresholds: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6)


def run_search(config: SearchConfig) -> dict:
    if config.step < 1:
        raise ValueError("step must be at least 1")
    if config.rolling_radius < 0:
        raise ValueError("rolling_radius cannot be negative")
    if not 0.0 <= config.confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not 0.0 <= config.nms_iou <= 1.0:
        raise ValueError("nms_iou must be between 0 and 1")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    images = discover_images(
        config.dataset_dir,
        glob_pattern=config.glob_pattern,
        center_regex=config.center_regex,
    )[:: config.step]
    segmenter = Sam3ParticleSegmenter(
        config.checkpoint,
        prompt=config.prompt,
        confidence_threshold=config.confidence_threshold,
        device=config.device,
    )

    rows: list[dict] = []
    for index, (center, path) in enumerate(images, start=1):
        image, normalization = normalize_reconstruction(path)
        started = time.perf_counter()
        prediction = segmenter.predict(image)
        metrics, _ = score_prediction(
            prediction, config.thresholds, config.nms_iou,
            config.high_score_threshold, config.min_area, config.max_area)
        row = {
            "center": center, "file": str(path),
            "seconds": time.perf_counter() - started,
            **metrics, **segmenter.last_telemetry,
            "normalization_low": normalization["low"],
            "normalization_high": normalization["high"],
        }
        with (config.output_dir / "scores_progress.jsonl").open("a", encoding="utf-8") as progress:
            progress.write(json.dumps(row) + "\n")
        del prediction
        rows.append(row)
        print(
            f"{index:4d}/{len(images)} center={center:g} "
            f"raw={metrics['raw_count']} nms={metrics['nms_count']} "
            f"high={metrics['high_count']}",
            flush=True,
        )

    centers = np.array([row["center"] for row in rows], dtype=np.float64)
    raw_counts = np.array([row["raw_count"] for row in rows], dtype=np.float64)
    nms_counts = np.array([row["nms_count"] for row in rows], dtype=np.float64)
    rolling = centered_rolling_mean(nms_counts, config.rolling_radius)
    assign_scores(rows)
    for row, value in zip(rows, rolling):
        row["rolling_nms_count"] = float(value)

    raw_best = rows[int(np.argmax(raw_counts))]
    nms_best = rows[int(np.argmax(nms_counts))]
    stable_best = rows[int(np.argmax(rolling))]
    multithreshold_best = max(rows, key=lambda row: row["multithreshold_score"])
    confidence_reward_best = max(rows, key=lambda row: row["confidence_reward"])

    fieldnames = list(rows[0])
    with (config.output_dir / "scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "num_images": len(rows),
        "raw_best": raw_best,
        "nms_best": nms_best,
        "stable_best": stable_best,
        "multithreshold_best": multithreshold_best,
        "particle_count_best": nms_best,
        "confidence_reward_best": confidence_reward_best,
        "top20_by_multithreshold_score": sorted(
            rows, key=lambda row: row["multithreshold_score"], reverse=True
        )[:20],
        "top20_by_nms_count": sorted(
            rows,
            key=lambda row: (row["nms_count"], row["high_count"]),
            reverse=True,
        )[:20],
        "top20_by_stable_count": sorted(
            rows,
            key=lambda row: (row["rolling_nms_count"], row["nms_count"]),
            reverse=True,
        )[:20],
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_count_curve(
        centers,
        raw_counts,
        nms_counts,
        rolling,
        stable_best["center"],
        config.output_dir / "count_curve.png",
    )

    best_path = Path(stable_best["file"])
    best_image, _ = normalize_reconstruction(best_path)
    best_prediction = segmenter.predict(best_image)
    _, kept = summarize_masks(
        best_prediction.masks,
        best_prediction.scores,
        nms_iou=config.nms_iou,
        high_score_threshold=config.high_score_threshold,
        min_area=config.min_area,
        max_area=config.max_area,
    )
    best_image.save(config.output_dir / "best_input.png")
    save_overlay(
        best_image,
        best_prediction.masks,
        kept,
        config.output_dir / "best_overlay.png",
    )
    return summary
