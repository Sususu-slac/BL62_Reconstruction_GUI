from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tifffile
from PIL import Image
from PySide6.QtCore import QObject, Signal, Slot

from .hdf5_inspector import FRAME_DATASETS, read_input_frame


@dataclass(slots=True)
class ExportJob:
    source: str
    input_path: Path
    output_dir: Path
    indices: list[int]
    output_format: str
    data_mode: str
    reconstruction_files: list[Path] | None = None


def display_convert(array: np.ndarray, bits: int) -> tuple[np.ndarray, tuple[float, float]]:
    data = np.asarray(array, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if not finite.size:
        raise ValueError("Frame contains no finite pixels")
    low, high = np.percentile(finite, (1, 99))
    if high <= low:
        high = low + 1.0
    maximum = 255 if bits == 8 else 65535
    converted = np.clip((data - low) / (high - low), 0, 1) * maximum
    dtype = np.uint8 if bits == 8 else np.uint16
    return converted.astype(dtype), (float(low), float(high))


def read_raw_hdf5_frame(path: Path, source: str, index: int) -> np.ndarray:
    dataset_name = FRAME_DATASETS[source]
    with h5py.File(path, "r") as handle:
        dataset = handle[dataset_name]
        if dataset.ndim == 2:
            return np.asarray(dataset[:])
        selected = max(0, min(index, dataset.shape[0] - 1))
        return np.asarray(dataset[selected])


class ExportWorker(QObject):
    progress = Signal(int, int, str)
    log = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, job: ExportJob):
        super().__init__()
        self.job = job
        self.cancel_requested = False

    @Slot()
    def cancel(self) -> None:
        self.cancel_requested = True

    def _read(self, position: int, index: int) -> tuple[np.ndarray, str]:
        if self.job.source == "reconstruction":
            assert self.job.reconstruction_files is not None
            path = self.job.reconstruction_files[position]
            return np.asarray(tifffile.imread(path)), path.stem
        if self.job.source in FRAME_DATASETS:
            return read_raw_hdf5_frame(self.job.input_path, self.job.source, index), f"{self.job.source}_{index:05d}"
        image, _ = read_input_frame(self.job.input_path, self.job.source, index)
        return image, f"{self.job.source}_{index:05d}"

    def _convert(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        if self.job.data_mode == "preserve":
            return image, {"mapping": "preserve"}
        if self.job.data_mode == "float32":
            return np.asarray(image, dtype=np.float32), {"mapping": "float32"}
        bits = 8 if self.job.data_mode == "display8" else 16
        converted, window = display_convert(image, bits)
        return converted, {"mapping": f"percentile_1_99_to_uint{bits}", "window": list(window)}

    @Slot()
    def run(self) -> None:
        records = []
        try:
            self.job.output_dir.mkdir(parents=True, exist_ok=True)
            total = len(self.job.indices)
            for position, index in enumerate(self.job.indices):
                if self.cancel_requested:
                    self.finished.emit(False, f"Export cancelled after {len(records)} of {total} files")
                    return
                image, stem = self._read(position, index)
                output, mapping = self._convert(image)
                suffix = ".tiff" if self.job.output_format == "tiff" else ".png"
                destination = self.job.output_dir / f"{stem}{suffix}"
                if self.job.output_format == "tiff":
                    tifffile.imwrite(destination, output)
                else:
                    if output.dtype not in (np.uint8, np.uint16):
                        raise ValueError("PNG requires 8-bit or 16-bit display mode")
                    Image.fromarray(output).save(destination)
                records.append({"index": index, "output": str(destination), "dtype": str(output.dtype),
                                "shape": list(output.shape), **mapping})
                self.progress.emit(position + 1, total, destination.name)

            metadata = {
                "created_at": datetime.now().isoformat(),
                "source": self.job.source,
                "input": str(self.job.input_path),
                "output_dir": str(self.job.output_dir),
                "format": self.job.output_format,
                "data_mode": self.job.data_mode,
                "count": len(records),
                "records": records,
            }
            (self.job.output_dir / "export_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            self.finished.emit(True, f"Exported {len(records)} files to {self.job.output_dir}")
        except Exception as exc:
            self.finished.emit(False, f"Export failed: {type(exc).__name__}: {exc}")
