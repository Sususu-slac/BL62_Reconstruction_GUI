"""Paint one fixed cross-image exemplar without loading SAM3 in the GUI."""
from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QDialog, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout)

from sam3_rotation_center.exemplar import InstanceExample
from .themes import colors_for


def pil_pixmap(image: Image.Image) -> QPixmap:
    array = np.ascontiguousarray(image.convert('RGB'))
    return QPixmap.fromImage(QImage(array.data, image.width, image.height,
                                    array.strides[0], QImage.Format.Format_RGB888).copy())


class MaskCanvas(QGraphicsView):
    history_changed = Signal()

    def __init__(self, image: Image.Image, parent=None):
        super().__init__(parent)
        self.reference = image.convert('RGB').copy()
        scene = QGraphicsScene(self)
        self.setScene(scene)
        scene.addPixmap(pil_pixmap(self.reference))
        self.overlay = QImage(image.width, image.height, QImage.Format.Format_ARGB32_Premultiplied)
        self.overlay.fill(Qt.GlobalColor.transparent)
        self.overlay_item = scene.addPixmap(QPixmap.fromImage(self.overlay))
        scene.setSceneRect(0, 0, image.width, image.height)
        self.setObjectName('maskCanvas');self.set_theme(getattr(parent,'theme_name','dark'))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMinimumSize(420, 320)
        self.tool = 'brush'
        self.brush_size = 12
        self.undo_stack, self.redo_stack = [], []
        self._last_point = None
        self._pan_point = None
        self._auto_fit = True

    def set_theme(self,theme):
        self.setBackgroundBrush(QColor(colors_for(theme)['canvas']))

    def mask(self) -> np.ndarray:
        rgba = self.overlay.convertToFormat(QImage.Format.Format_RGBA8888)
        pixels = np.frombuffer(rgba.constBits(), np.uint8).reshape(rgba.height(), rgba.bytesPerLine())
        return pixels[:, :rgba.width()*4].reshape(rgba.height(), rgba.width(), 4)[:, :, 3].copy() > 0

    def _snapshot(self):
        return np.packbits(self.mask(), axis=None)

    def _restore(self, packed):
        mask = np.unpackbits(packed, count=self.reference.width*self.reference.height).reshape(self.reference.height, self.reference.width).astype(bool)
        rgba = np.zeros((self.reference.height, self.reference.width, 4), np.uint8)
        rgba[mask] = (99, 244, 181, 110)
        self.overlay = QImage(rgba.data, self.reference.width, self.reference.height, rgba.strides[0],
                              QImage.Format.Format_RGBA8888).copy().convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
        self.overlay_item.setPixmap(QPixmap.fromImage(self.overlay))
        self.history_changed.emit()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self._snapshot())
            self._restore(self.undo_stack.pop())

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self._snapshot())
            self._restore(self.redo_stack.pop())

    def _begin_stroke(self, point):
        self.undo_stack.append(self._snapshot())
        # Binary snapshots, bounded both by count and total memory (64 MiB).
        limit = max(1, min(40, (64*2**20)//max(1, self.undo_stack[-1].nbytes)))
        self.undo_stack = self.undo_stack[-limit:]
        self.redo_stack.clear()
        self._last_point = point
        self._paint_to(point)
        self.history_changed.emit()

    def _paint_to(self, point):
        painter = QPainter(self.overlay)
        if self.tool == 'eraser':
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        else:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.setPen(QPen(QColor(99, 244, 181, 110), self.brush_size,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        if point == self._last_point:
            painter.setBrush(QColor(99, 244, 181, 110))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(point, self.brush_size/2, self.brush_size/2)
        else:
            painter.drawLine(self._last_point, point)
        painter.end()
        self._last_point = point
        self.overlay_item.setPixmap(QPixmap.fromImage(self.overlay))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_point = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        point = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and self.sceneRect().contains(point):
            self._begin_stroke(point)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_point is not None:
            delta = event.position()-self._pan_point
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value()-int(delta.y()))
            self._pan_point = event.position()
            event.accept()
        elif self._last_point is not None:
            self._paint_to(self.mapToScene(event.position().toPoint()))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_point = None
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_point = None
            self.setCursor(Qt.CursorShape.CrossCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        scale = self.transform().m11()*factor
        if .01 <= scale <= 64:
            self.scale(factor, factor)
            self._auto_fit = False
        event.accept()

    def fit_image(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._auto_fit = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit_image()


class InstanceDialog(QDialog):
    def __init__(self, image: Image.Image, source: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select Instance')
        self.resize(980, 780)
        self.source = source
        self.theme_name = getattr(parent,'theme_name','dark')
        self.example = None
        layout = QVBoxLayout(self)
        tools = QHBoxLayout()
        self.brush = QPushButton('Brush')
        self.eraser = QPushButton('Eraser')
        group = QButtonGroup(self)
        for button in (self.brush, self.eraser):
            button.setCheckable(True)
            group.addButton(button)
            tools.addWidget(button)
        self.brush.setChecked(True)
        tools.addWidget(QLabel('Size'))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 512)
        self.size_spin.setValue(12)
        self.size_spin.setSuffix(' px')
        tools.addWidget(self.size_spin)
        self.undo_button, self.redo_button = QPushButton('Undo'), QPushButton('Redo')
        tools.addWidget(self.undo_button)
        tools.addWidget(self.redo_button)
        tools.addStretch()
        fit = QPushButton('Fit')
        tools.addWidget(fit)
        layout.addLayout(tools)
        self.canvas = MaskCanvas(image, self)
        self.canvas.setToolTip('Wheel: zoom · Middle mouse: pan · B: brush · E: eraser · Ctrl+Z: undo · Ctrl+Y: redo')
        layout.addWidget(self.canvas, 1)
        self.brush.clicked.connect(lambda: setattr(self.canvas, 'tool', 'brush'))
        self.eraser.clicked.connect(lambda: setattr(self.canvas, 'tool', 'eraser'))
        self.size_spin.valueChanged.connect(lambda value: setattr(self.canvas, 'brush_size', value))
        self.undo_button.clicked.connect(self.canvas.undo)
        self.redo_button.clicked.connect(self.canvas.redo)
        self.canvas.history_changed.connect(self._history_changed)
        fit.clicked.connect(self.canvas.fit_image)
        for keys, callback in [('Ctrl+Z', self.canvas.undo), ('Ctrl+Y', self.canvas.redo),
                               ('Ctrl+Shift+Z', self.canvas.redo), ('B', self.brush.click), ('E', self.eraser.click)]:
            QShortcut(QKeySequence(keys), self, activated=callback)
        actions = QHBoxLayout()
        actions.addStretch()
        cancel, confirm = QPushButton('Cancel'), QPushButton('Confirm')
        confirm.setObjectName('primaryButton')
        cancel.clicked.connect(self.reject)
        confirm.clicked.connect(self._confirm)
        # Enter must not accidentally paint/activate an unrelated toolbar action.
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
        actions.addWidget(cancel)
        actions.addWidget(confirm)
        layout.addLayout(actions)
        self._history_changed()

    def _history_changed(self):
        self.undo_button.setEnabled(bool(self.canvas.undo_stack))
        self.redo_button.setEnabled(bool(self.canvas.redo_stack))

    def _confirm(self):
        try:
            self.example = InstanceExample(self.canvas.reference.copy(), self.canvas.mask(), self.source).validate()
        except ValueError as error:
            QMessageBox.warning(self, 'Select a feature', str(error))
            return
        self.accept()
