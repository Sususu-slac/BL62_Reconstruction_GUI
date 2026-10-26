from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_overlay(
    image: Image.Image,
    masks: np.ndarray,
    indices: list[int],
    output_path: Path,
) -> None:
    base = np.asarray(image, dtype=np.float32)
    overlay = base.copy()
    rng = np.random.default_rng(0)
    for index in indices:
        color = rng.integers(32, 256, size=3)
        overlay[masks[index]] = 0.45 * overlay[masks[index]] + 0.55 * color
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)


def save_count_curve(
    centers: np.ndarray,
    raw_counts: np.ndarray,
    nms_counts: np.ndarray,
    rolling_counts: np.ndarray,
    best_center: float,
    output_path: Path,
) -> None:
    width, height, margin = 1600, 800, 90
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y_max = max(raw_counts.max(), nms_counts.max(), rolling_counts.max(), 1) * 1.05

    def xy(center: float, value: float) -> tuple[float, float]:
        x = margin + (center - centers.min()) / max(
            centers.max() - centers.min(), 1
        ) * (width - 2 * margin)
        y = height - margin - value / y_max * (height - 2 * margin)
        return x, y

    axes = [
        (margin, margin),
        (margin, height - margin),
        (width - margin, height - margin),
    ]
    draw.line(axes, fill="black", width=2)
    draw.line(
        [xy(center, value) for center, value in zip(centers, raw_counts)],
        fill=(190, 200, 225),
        width=1,
    )
    draw.line(
        [xy(center, value) for center, value in zip(centers, nms_counts)],
        fill=(70, 130, 220),
        width=1,
    )
    draw.line(
        [xy(center, value) for center, value in zip(centers, rolling_counts)],
        fill=(10, 70, 190),
        width=3,
    )
    best_index = int(np.argmin(np.abs(centers - best_center)))
    x, y = xy(best_center, rolling_counts[best_index])
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="red")
    draw.text(
        (margin, 25),
        "Particle count: raw (light), mask-NMS (blue), rolling mean (dark)",
        fill="black",
    )
    draw.text(
        (width - 420, 25),
        f"best stable center={best_center:g}",
        fill="red",
    )
    canvas.save(output_path)
