from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_CENTER_REGEX = r"_center(?P<center>-?\d+(?:\.\d+)?)\.(?:tiff?|png)$"


def parse_center(path: Path, pattern: str = DEFAULT_CENTER_REGEX) -> float:
    match = re.search(pattern, path.name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot parse rotation center from {path.name!r}")
    return float(match.group("center"))


def discover_images(
    dataset_dir: Path,
    glob_pattern: str = "*.png",
    center_regex: str = DEFAULT_CENTER_REGEX,
) -> list[tuple[float, Path]]:
    items = [
        (parse_center(path, center_regex), path)
        for path in dataset_dir.glob(glob_pattern)
        if path.is_file() and re.search(center_regex, path.name, flags=re.IGNORECASE)
    ]
    if not items:
        raise FileNotFoundError(
            f"No images matched {glob_pattern!r} in {dataset_dir}"
        )
    return sorted(items, key=lambda item: item[0])


def normalize_reconstruction(
    path: Path,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.9,
) -> tuple[Image.Image, dict[str, float]]:
    array = np.asarray(Image.open(path), dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError(f"Image contains no finite pixels: {path}")

    low, high = np.percentile(
        array[finite], [lower_percentile, upper_percentile]
    )
    if high <= low:
        raise ValueError(f"Invalid intensity range in {path}: {low} to {high}")

    normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
    normalized[~finite] = 0.0
    uint8 = np.round(normalized * 255.0).astype(np.uint8)
    return Image.fromarray(uint8, mode="L").convert("RGB"), {
        "low": float(low),
        "high": float(high),
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
    }
