"""Detector coordinates and reconstruction-output coordinates are distinct in lamino."""
from __future__ import annotations

import math


def output_row_limit(detector_rows: int, geometry="tomography", lamino_angle=0.0, binning=0) -> int:
    rows = int(detector_rows)
    exponent = int(binning)
    if rows < 1 or exponent not in (0, 1, 2, 3):
        raise ValueError("Invalid detector height or binning")
    if geometry != "laminography":
        return rows
    angle = float(lamino_angle)
    if not math.isfinite(angle) or abs(angle) >= 90:
        raise ValueError("Lamino tilt must be finite and below 90 degrees")
    factor = 2 ** exponent
    binned_rows = rows // factor
    # Same even-height convention as TomoCuPy Reader.init_sizes_lamino.
    height = 2 * math.ceil(binned_rows / math.cos(math.radians(angle)) / 2)
    return int(height * factor)


def output_slice_count(detector_rows: int, geometry="tomography", lamino_angle=0.0, binning=0) -> int:
    return output_row_limit(detector_rows, geometry, lamino_angle, binning) // (2 ** int(binning))
