from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


REQUIRED = ("exchange/data", "exchange/data_white", "exchange/data_dark", "exchange/theta")


@dataclass(slots=True)
class Hdf5Info:
    path: Path
    data_shape: tuple[int, ...]
    data_dtype: str
    white_shape: tuple[int, ...]
    dark_shape: tuple[int, ...]
    theta_shape: tuple[int, ...]
    theta_unit: str
    theta_min: float
    theta_max: float
    theta_uniform: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def suggested_center(self) -> float:
        return (self.data_shape[2] - 1) / 2.0

    @property
    def suggested_rows(self) -> tuple[int, int]:
        middle = self.data_shape[1] // 2
        return max(0, middle - 2), min(self.data_shape[1], middle + 3)


def inspect_hdf5(path: str | Path) -> Hdf5Info:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        missing = [key for key in REQUIRED if key not in handle]
        if missing:
            raise KeyError(f"Missing datasets: {', '.join('/' + key for key in missing)}")
        data = handle["exchange/data"]
        if data.ndim != 3:
            raise ValueError(f"/exchange/data must be 3D, got {data.shape}")
        theta = np.asarray(handle["exchange/theta"][:], dtype=np.float64)
        if len(theta) != data.shape[0]:
            raise ValueError("Theta count does not match projection count")
        finite = theta[np.isfinite(theta)]
        if finite.size != theta.size:
            raise ValueError("Theta contains NaN or Inf")
        unit = "degrees" if np.max(np.abs(theta)) > 2 * np.pi + 0.1 else "radians"
        diffs = np.diff(theta)
        uniform = bool(diffs.size == 0 or np.allclose(diffs, np.median(diffs), rtol=1e-4, atol=1e-6))
        warnings = []
        if not uniform:
            warnings.append("Theta is not strictly uniform; LpRec needs explicit uniform-angle approximation")
        return Hdf5Info(
            path=path,
            data_shape=tuple(data.shape),
            data_dtype=str(data.dtype),
            white_shape=tuple(handle["exchange/data_white"].shape),
            dark_shape=tuple(handle["exchange/data_dark"].shape),
            theta_shape=tuple(theta.shape),
            theta_unit=unit,
            theta_min=float(theta.min()),
            theta_max=float(theta.max()),
            theta_uniform=uniform,
            warnings=warnings,
        )


def read_projection(path: str | Path, index: int | None = None) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        data = handle["exchange/data"]
        selected = data.shape[0] // 2 if index is None else max(0, min(index, data.shape[0] - 1))
        return np.asarray(data[selected], dtype=np.float32)


FRAME_DATASETS = {
    "projection": "exchange/data",
    "flat": "exchange/data_white",
    "dark": "exchange/data_dark",
}


def frame_count(info: Hdf5Info, source: str) -> int:
    if source in ("corrected", "attenuation"):
        return info.data_shape[0]
    shape = {
        "projection": info.data_shape,
        "flat": info.white_shape,
        "dark": info.dark_shape,
    }[source]
    return shape[0] if len(shape) == 3 else 1


def read_input_frame(path: str | Path, source: str, index: int) -> tuple[np.ndarray, float | None]:
    """Read one preview frame; corrected views use mean flat and dark fields."""
    if source not in (*FRAME_DATASETS, "corrected", "attenuation"):
        raise ValueError(f"Unknown input source: {source}")
    with h5py.File(path, "r") as handle:
        if source in FRAME_DATASETS:
            dataset = handle[FRAME_DATASETS[source]]
            selected = max(0, min(index, (dataset.shape[0] - 1) if dataset.ndim == 3 else 0))
            image = dataset[selected] if dataset.ndim == 3 else dataset[:]
            angle = None
            if source == "projection":
                angle = float(handle["exchange/theta"][selected])
            return np.asarray(image, dtype=np.float32), angle

        data = handle["exchange/data"]
        selected = max(0, min(index, data.shape[0] - 1))
        projection = np.asarray(data[selected], dtype=np.float32)
        white_data = handle["exchange/data_white"]
        dark_data = handle["exchange/data_dark"]
        flat = np.mean(white_data[:], axis=0, dtype=np.float32) if white_data.ndim == 3 else np.asarray(white_data[:], np.float32)
        dark = np.mean(dark_data[:], axis=0, dtype=np.float32) if dark_data.ndim == 3 else np.asarray(dark_data[:], np.float32)
        corrected = (projection - dark) / np.maximum(flat - dark, 1.0)
        image = -np.log(np.maximum(corrected, 1e-6)) if source == "attenuation" else corrected
        return image.astype(np.float32, copy=False), float(handle["exchange/theta"][selected])
