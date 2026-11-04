from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PySide6.QtCore import Qt, QObject, QRunnable, QThreadPool, Signal
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


class PreviewSignals(QObject):
    ready = Signal(object)


class TiffPreview(QRunnable):
    def __init__(self, path, generation):
        super().__init__()
        self.path, self.generation = path, generation
        self.signals = PreviewSignals()

    def run(self):
        try:
            gray = np.ascontiguousarray(normalize_uint8(tifffile.imread(self.path)))
            result = (self.generation, self.path, gray, None)
        except Exception as error:
            result = (self.generation, self.path, None, str(error))
        self.signals.ready.emit(result)


class ImageView(QLabel):
    def __init__(self) -> None:
        super().__init__("Select an HDF5 file to preview data")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(520, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName('imageCanvas')
        self._pixmap: QPixmap | None = None
        self._generation = 0
        self._loading = False
        self._pending_preview = None

    def show_array(self, array: np.ndarray) -> None:
        self._generation += 1
        self._pending_preview = None
        gray = np.ascontiguousarray(normalize_uint8(array))
        self._show_gray(gray)

    def _show_gray(self, gray):
        height, width = gray.shape
        image = QImage(gray.data, width, height, gray.strides[0], QImage.Format.Format_Grayscale8).copy()
        self._pixmap = QPixmap.fromImage(image)
        self._fit()

    def show_tiff(self, path: str | Path) -> None:
        self._generation += 1
        self._pending_preview = (str(path), self._generation)
        self._start_preview()

    def _start_preview(self):
        if self._loading or self._pending_preview is None:return
        path, generation = self._pending_preview
        self._pending_preview = None
        task = TiffPreview(path, generation)
        task.signals.ready.connect(self._preview_ready)
        self._loading = True
        QThreadPool.globalInstance().start(task)

    def _preview_ready(self, result):
        self._loading = False
        generation, path, gray, error = result
        if generation == self._generation:
            if gray is not None:self._show_gray(gray)
            self.setToolTip(path if error is None else f'{path}\nPreview unavailable: {error}')
        self._start_preview()

    def reset(self, message: str = "Waiting for reconstruction results") -> None:
        """Drop the old pixmap so a new run cannot show a stale result."""
        self._generation += 1
        self._pending_preview = None
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
