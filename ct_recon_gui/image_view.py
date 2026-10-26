from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if not finite.size:
        raise ValueError("Image has no finite pixels")
    low, high = np.percentile(finite, (1, 99))
    if high <= low:
        high = low + 1
    return (np.clip((image - low) / (high - low), 0, 1) * 255).astype(np.uint8)


class ImageView(QLabel):
    def __init__(self) -> None:
        super().__init__("Select an HDF5 file to preview data")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(520, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#090d14; color:#8e9bb0; border:1px solid #273043;")
        self._pixmap: QPixmap | None = None

    def show_array(self, array: np.ndarray) -> None:
        gray = np.ascontiguousarray(normalize_uint8(array))
        height, width = gray.shape
        image = QImage(gray.data, width, height, gray.strides[0], QImage.Format.Format_Grayscale8).copy()
        self._pixmap = QPixmap.fromImage(image)
        self._fit()

    def show_tiff(self, path: str | Path) -> None:
        self.show_array(tifffile.imread(path))
        self.setToolTip(str(path))

    def reset(self, message: str = "Waiting for reconstruction results") -> None:
        """Drop the old pixmap so a new run cannot show a stale result."""
        self._pixmap = None
        self.clear()
        self.setText(message)
        self.setToolTip("")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        if self._pixmap:
            self.setPixmap(self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
