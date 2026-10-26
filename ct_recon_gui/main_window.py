from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QIcon, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from .analysis_ui import AnalysisUiMixin
from .analysis_runner import AnalysisRunner
from .gpu_monitor import GpuMonitor, GpuSelector
from .runtime import APP_ROOT, load_settings
from .commands import build_wsl_argv, display_command
from .exporter import ExportJob, ExportWorker
from .hdf5_inspector import Hdf5Info, frame_count, inspect_hdf5, read_input_frame
from .image_view import ImageView
from .job_runner import JobRunner
from .models import GeometryMode, ReconJob, ReconMethod, TOMO_METHODS


METHOD_LABELS = {
    ReconMethod.FOURIERREC: "FourierRec — TomocuPy",
    ReconMethod.LPREC: "LpRec — TomocuPy",
    ReconMethod.LINEREC: "LineRec — TomocuPy",
    ReconMethod.FBP_CUDA: "FBP_CUDA — ASTRA",
    ReconMethod.SIRT_CUDA: "SIRT_CUDA — ASTRA",
    ReconMethod.CGLS_CUDA: "CGLS_CUDA — ASTRA",
    ReconMethod.LAMINO_FOURIERREC: "FourierRec — TomocuPy Laminography",
    ReconMethod.LAMINO_LINEREC: "LineRec — TomocuPy Laminography",
}


class MainWindow(AnalysisUiMixin, QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        font_path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/'segoeui.ttf'
        if font_path.is_file():
            QFontDatabase.addApplicationFont(str(font_path))
            symbols=font_path.with_name('seguisym.ttf')
            if symbols.is_file():QFontDatabase.addApplicationFont(str(symbols))
        self.setFont(QFont('Segoe UI',10))
        self.setWindowTitle("BL62 Reconstruction")
        self.resize(1680, 980)
        self.info: Hdf5Info | None = None
        self.results: list[Path] = []
        self.center_candidates: list[tuple[float, Path]] = []
        self.search_candidate_kind = "center"
        self.search_initial_center = 0.0
        self.active_selected_type = "full"
        self.active_output_dir: Path | None = None
        self._run_serial = 0
        self.export_thread: QThread | None = None
        self.export_worker: ExportWorker | None = None
        self._preview_cache: dict[tuple[str, int], tuple] = {}
        self.preview_load_timer = QTimer(self)
        self.preview_load_timer.setSingleShot(True)
        self.preview_load_timer.setInterval(80)
        self.preview_load_timer.timeout.connect(self._load_input_frame)
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(180)
        self.play_timer.timeout.connect(self._next_input_frame)
        self.runner = JobRunner(self)
        self.runner.log.connect(self.append_log)
        self.runner.state_changed.connect(self._set_state)
        self.runner.result_found.connect(self._new_result)
        self.runner.finished.connect(self._job_finished)
        self.settings = load_settings()
        self.analysis_runner = AnalysisRunner(self)
        self.analysis_runner.log.connect(self.append_log)
        self.analysis_runner.updated.connect(self._analysis_updated)
        self.analysis_runner.finished.connect(self._analysis_finished)
        self.result_preview_timer=QTimer(self);self.result_preview_timer.setSingleShot(True);self.result_preview_timer.setInterval(80)
        self.result_preview_timer.timeout.connect(self._display_selected_result)
        self._batching_results=False
        self._build_ui()
        self._apply_style()
        self._mode_changed()
        self._environment_label()
        self.gpu_monitor = GpuMonitor(self)
        self.gpu_monitor.updated.connect(self._gpu_updated)

    def _build_ui(self) -> None:
        root = QWidget(); outer = QVBoxLayout(root)
        self.setWindowIcon(QIcon(str(APP_ROOT / 'BL62_Recon.ico')))
        header = QHBoxLayout(); title = QLabel('BL62 Reconstruction'); title.setObjectName('title')
        header.addWidget(title); header.addStretch()
        self.backend_status = QLabel(); self.backend_status.setObjectName('statusPill'); header.addWidget(self.backend_status)
        outer.addLayout(header)
        self.main_tabs = QTabWidget()
        horizontal = QSplitter(Qt.Orientation.Horizontal)
        left=self._left_panel();left.setMinimumWidth(370)
        side=self._right_panel();side.setMinimumWidth(260)
        horizontal.addWidget(left);horizontal.addWidget(self._center_panel());horizontal.addWidget(side)
        horizontal.setChildrenCollapsible(False)
        horizontal.setSizes([410,900,320]);horizontal.setStretchFactor(1,1)
        self.main_tabs.addTab(horizontal,'Reconstruction')
        self.main_tabs.addTab(self._build_analysis_page(),'Intelligent Data Analysis (Test)')
        outer.addWidget(self.main_tabs,1)
        self.log_toggle = QToolButton();self.log_toggle.setText('Log');self.log_toggle.setCheckable(True)
        self.log_toggle.setArrowType(Qt.ArrowType.RightArrow);self.log_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.log_box=QPlainTextEdit();self.log_box.setReadOnly(True);self.log_box.setMaximumBlockCount(10000);self.log_box.setFixedHeight(140);self.log_box.hide()
        def toggle(checked):
            self.log_box.setVisible(checked);self.log_toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self.log_toggle.toggled.connect(toggle);outer.addWidget(self.log_toggle);outer.addWidget(self.log_box)
        self.setCentralWidget(root)

    def _left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        files = QGroupBox("Data")
        form = QFormLayout(files)
        self.input_edit = QLineEdit()
        browse = QPushButton("Browse HDF5")
        browse.clicked.connect(self._choose_input)
        row = QHBoxLayout(); row.addWidget(self.input_edit); row.addWidget(browse)
        form.addRow("Input", row)
        self.output_edit = QLineEdit()
        out_btn = QPushButton("Browse")
        out_btn.clicked.connect(self._choose_output)
        row = QHBoxLayout(); row.addWidget(self.output_edit); row.addWidget(out_btn)
        form.addRow("Output", row)
        self.file_info = QLabel("No file selected")
        self.file_info.setWordWrap(True)
        form.addRow(self.file_info)
        layout.addWidget(files)

        geometry = QGroupBox("Geometry and method")
        form = QFormLayout(geometry)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Tomography", GeometryMode.TOMOGRAPHY)
        self.mode_combo.addItem("Laminography", GeometryMode.LAMINOGRAPHY)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        form.addRow("Mode", self.mode_combo)
        self.method_combo = QComboBox()
        self.method_combo.currentIndexChanged.connect(self._method_changed)
        form.addRow("Algorithm", self.method_combo)
        layout.addWidget(geometry)

        params = QGroupBox("Reconstruction parameters")
        form = QFormLayout(params)
        self.center_spin = QDoubleSpinBox(); self.center_spin.setRange(0, 100000); self.center_spin.setDecimals(3)
        self.start_spin = QSpinBox(); self.start_spin.setRange(0, 100000)
        self.end_spin = QSpinBox(); self.end_spin.setRange(1, 100000); self.end_spin.setValue(5)
        self.try_row_spin = QSpinBox(); self.try_row_spin.setRange(0, 100000)
        self.search_width_spin = QDoubleSpinBox(); self.search_width_spin.setRange(0.01, 10000); self.search_width_spin.setDecimals(3); self.search_width_spin.setValue(20)
        self.search_step_spin = QDoubleSpinBox(); self.search_step_spin.setRange(0.001, 1000); self.search_step_spin.setDecimals(3); self.search_step_spin.setValue(0.5)
        self.lamino_search_width_spin = QDoubleSpinBox(); self.lamino_search_width_spin.setRange(0.01, 89.9); self.lamino_search_width_spin.setDecimals(3); self.lamino_search_width_spin.setValue(5)
        self.lamino_search_step_spin = QDoubleSpinBox(); self.lamino_search_step_spin.setRange(0.001, 89.9); self.lamino_search_step_spin.setDecimals(3); self.lamino_search_step_spin.setValue(0.25)
        self.gpu_spin = GpuSelector(); self.gpu_spin.setValue(self.settings["gpu"]); self.gpu_spin.currentIndexChanged.connect(self._gpu_selection_changed)
        self.chunk_spin = QSpinBox(); self.chunk_spin.setRange(1, 256); self.chunk_spin.setValue(4)
        self.binning_combo = QComboBox()
        for label, exponent in (("1×1 — Original", 0), ("2×2", 1), ("4×4", 2), ("8×8", 3)):
            self.binning_combo.addItem(label, exponent)
        self.binning_combo.currentIndexChanged.connect(self._effective_geometry_changed)
        self.iter_spin = QSpinBox(); self.iter_spin.setRange(1, 10000); self.iter_spin.setValue(20)
        self.lamino_spin = QDoubleSpinBox(); self.lamino_spin.setRange(-89.9, 89.9); self.lamino_spin.setValue(20)
        self.type_combo = QComboBox()
        self.type_combo.addItem("Try Rotation Center", "center_try")
        self.type_combo.addItem("Try Lamino Angle", "angle_try")
        self.type_combo.addItem("Full Reconstruction", "full")
        self.type_combo.setCurrentIndex(self.type_combo.findData("full"))
        center_row = QWidget(); center_layout = QHBoxLayout(center_row); center_layout.setContentsMargins(0, 0, 0, 0)
        self.center_settings_btn = QPushButton("⚙"); self.center_settings_btn.setFixedWidth(34); self.center_settings_btn.clicked.connect(self._open_center_settings)
        self.find_center_btn = QPushButton("Find Center"); self.find_center_btn.clicked.connect(lambda: self._start_search("center"))
        self.find_center_btn.setFixedSize(92,34); self.center_settings_btn.setFixedSize(28,34); self.center_spin.setMinimumWidth(70); self.center_spin.setFixedHeight(34)
        center_layout.addWidget(self.center_spin, 1); center_layout.addWidget(self.center_settings_btn); center_layout.addWidget(self.find_center_btn)
        lamino_row = QWidget(); lamino_layout = QHBoxLayout(lamino_row); lamino_layout.setContentsMargins(0, 0, 0, 0)
        self.lamino_settings_btn = QPushButton("⚙"); self.lamino_settings_btn.setFixedWidth(34); self.lamino_settings_btn.clicked.connect(self._open_lamino_settings)
        self.find_angle_btn = QPushButton("Find Angle"); self.find_angle_btn.clicked.connect(lambda: self._start_search("angle"))
        self.find_angle_btn.setFixedSize(92,34); self.lamino_settings_btn.setFixedSize(28,34); self.lamino_spin.setMinimumWidth(70); self.lamino_spin.setFixedHeight(34)
        lamino_layout.addWidget(self.lamino_spin, 1); lamino_layout.addWidget(self.lamino_settings_btn); lamino_layout.addWidget(self.find_angle_btn)
        self.params_form = form; self.params_box = params; self.lamino_row = lamino_row
        for label, widget in (("Rotation center", center_row), ("Lamino angle", lamino_row),
                              ("Start row", self.start_spin), ("End row (exclusive)", self.end_spin),
                              ("Binning", self.binning_combo), ("GPU", self.gpu_spin), ("Chunk size", self.chunk_spin),
                              ("Iterations", self.iter_spin)):
            form.addRow(label, widget)
        self.effective_geometry_label = QLabel("Effective geometry: select an HDF5 file")
        self.effective_geometry_label.setWordWrap(True)
        form.addRow(self.effective_geometry_label)
        self.center_spin.valueChanged.connect(self._effective_geometry_changed)
        self.start_spin.valueChanged.connect(self._effective_geometry_changed)
        self.end_spin.valueChanged.connect(self._effective_geometry_changed)
        self._update_search_tooltips()
        layout.addWidget(params)

        self.run_btn = QPushButton("Run Reconstruction")
        self.run_btn.setObjectName("primaryButton"); self.run_btn.clicked.connect(self._start_job)
        layout.addWidget(self.run_btn); layout.addStretch()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _open_center_settings(self) -> None:
        dialog = QDialog(self); dialog.setWindowTitle("Rotation Center Search Settings")
        layout = QVBoxLayout(dialog); form = QFormLayout()
        row = QSpinBox(); row.setRange(self.try_row_spin.minimum(), self.try_row_spin.maximum()); row.setValue(self.try_row_spin.value())
        width = QDoubleSpinBox(); width.setRange(0.01, 10000); width.setDecimals(3); width.setValue(self.search_width_spin.value()); width.setSuffix(" px")
        step = QDoubleSpinBox(); step.setRange(0.001, 1000); step.setDecimals(3); step.setValue(self.search_step_spin.value()); step.setSuffix(" px")
        count = QLabel()
        update_count = lambda: count.setText(str(max(1, int(np.ceil(2 * width.value() / step.value())))))
        width.valueChanged.connect(update_count); step.valueChanged.connect(update_count); update_count()
        form.addRow("Try detector row", row); form.addRow("Search range ±", width)
        form.addRow("Search step", step); form.addRow("Candidate count", count); layout.addLayout(form)
        reset = QPushButton("Reset Defaults"); reset.clicked.connect(lambda: (width.setValue(20), step.setValue(0.5)))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        button_row = QHBoxLayout(); button_row.addWidget(reset); button_row.addStretch(); button_row.addWidget(buttons); layout.addLayout(button_row)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.try_row_spin.setValue(row.value()); self.search_width_spin.setValue(width.value()); self.search_step_spin.setValue(step.value())
            self._update_search_tooltips()

    def _open_lamino_settings(self) -> None:
        dialog = QDialog(self); dialog.setWindowTitle("Lamino Angle Search Settings")
        layout = QVBoxLayout(dialog); form = QFormLayout()
        row = QSpinBox(); row.setRange(self.try_row_spin.minimum(), self.try_row_spin.maximum()); row.setValue(self.try_row_spin.value())
        angle = QDoubleSpinBox(); angle.setRange(-89.9, 89.9); angle.setDecimals(3); angle.setValue(self.lamino_spin.value()); angle.setSuffix("°")
        width = QDoubleSpinBox(); width.setRange(0.01, 89.9); width.setDecimals(3); width.setValue(self.lamino_search_width_spin.value()); width.setSuffix("°")
        step = QDoubleSpinBox(); step.setRange(0.001, 89.9); step.setDecimals(3); step.setValue(self.lamino_search_step_spin.value()); step.setSuffix("°")
        count = QLabel()
        update_count = lambda: count.setText(str(max(1, int(np.ceil(2 * width.value() / step.value())))))
        width.valueChanged.connect(update_count); step.valueChanged.connect(update_count); update_count()
        form.addRow("Try detector row", row); form.addRow("Initial angle", angle); form.addRow("Search range ±", width)
        form.addRow("Search step", step); form.addRow("Candidate count", count); layout.addLayout(form)
        reset = QPushButton("Reset Defaults"); reset.clicked.connect(lambda: (width.setValue(5), step.setValue(0.25)))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        button_row = QHBoxLayout(); button_row.addWidget(reset); button_row.addStretch(); button_row.addWidget(buttons); layout.addLayout(button_row)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.try_row_spin.setValue(row.value()); self.lamino_spin.setValue(angle.value()); self.lamino_search_width_spin.setValue(width.value()); self.lamino_search_step_spin.setValue(step.value())
            self._update_search_tooltips()

    def _update_search_tooltips(self) -> None:
        self.center_settings_btn.setToolTip(
            f"Try row {self.try_row_spin.value()} | ±{self.search_width_spin.value():g} px | step {self.search_step_spin.value():g} px"
        )
        self.lamino_settings_btn.setToolTip(
            f"Try row {self.try_row_spin.value()} | ±{self.lamino_search_width_spin.value():g}° | step {self.lamino_search_step_spin.value():g}°"
        )

    def _export_panel(self) -> QWidget:
        box = QGroupBox("Export Tools")
        form = QFormLayout(box)
        self.export_source_combo = QComboBox()
        for label, value in (("Projection", "projection"), ("Flat", "flat"), ("Dark", "dark"),
                             ("Corrected", "corrected"), ("-Log Attenuation", "attenuation"),
                             ("Reconstruction results", "reconstruction")):
            self.export_source_combo.addItem(label, value)
        self.export_source_combo.currentIndexChanged.connect(self._export_source_changed)
        self.export_scope_combo = QComboBox()
        self.export_scope_combo.addItem("Current", "current")
        self.export_scope_combo.addItem("Range", "range")
        self.export_scope_combo.addItem("All", "all")
        self.export_scope_combo.currentIndexChanged.connect(self._export_scope_changed)
        range_row = QHBoxLayout()
        self.export_start_spin = QSpinBox(); self.export_start_spin.setRange(0, 0)
        self.export_end_spin = QSpinBox(); self.export_end_spin.setRange(1, 1)
        range_row.addWidget(self.export_start_spin); range_row.addWidget(QLabel("to")); range_row.addWidget(self.export_end_spin)
        self.export_range_widget = QWidget(); self.export_range_widget.setLayout(range_row)
        self.export_format_combo = QComboBox(); self.export_format_combo.addItem("TIFF", "tiff"); self.export_format_combo.addItem("PNG", "png")
        self.export_format_combo.currentIndexChanged.connect(self._export_format_changed)
        self.export_mode_combo = QComboBox()
        self.export_mode_combo.addItem("Preserve values", "preserve")
        self.export_mode_combo.addItem("Float32", "float32")
        self.export_mode_combo.addItem("Display 8-bit", "display8")
        self.export_mode_combo.addItem("Display 16-bit", "display16")
        self.export_output_edit = QLineEdit()
        export_browse = QPushButton("Browse")
        export_browse.clicked.connect(self._choose_export_output)
        output_row = QHBoxLayout(); output_row.addWidget(self.export_output_edit); output_row.addWidget(export_browse)
        self.export_progress = QProgressBar(); self.export_progress.setRange(0, 100); self.export_progress.setValue(0)
        self.export_btn = QPushButton("Export")
        self.export_btn.setObjectName("primaryButton")
        self.export_btn.clicked.connect(self._start_export)
        self.export_cancel_btn = QPushButton("Cancel"); self.export_cancel_btn.setEnabled(False)
        self.export_cancel_btn.clicked.connect(self._cancel_export)
        button_row = QHBoxLayout(); button_row.addWidget(self.export_btn); button_row.addWidget(self.export_cancel_btn)
        form.addRow("Source", self.export_source_combo); form.addRow("Scope", self.export_scope_combo)
        form.addRow("Range [start,end)", self.export_range_widget)
        form.addRow("Format", self.export_format_combo); form.addRow("Data mode", self.export_mode_combo)
        form.addRow("Output", output_row); form.addRow(self.export_progress); form.addRow(button_row)
        self._export_scope_changed()
        return box

    def _center_panel(self) -> QWidget:
        tabs = QTabWidget()
        self.input_view = ImageView()
        self.result_view = ImageView()
        browser = QWidget(); browser_layout = QVBoxLayout(browser)
        toolbar = QHBoxLayout()
        self.input_source_combo = QComboBox()
        for label, value in (("Projection", "projection"), ("Flat", "flat"), ("Dark", "dark"),
                             ("Flat/Dark Corrected", "corrected"), ("-Log Attenuation", "attenuation")):
            self.input_source_combo.addItem(label, value)
        self.input_source_combo.currentIndexChanged.connect(self._input_source_changed)
        self.preview_binning_check = QCheckBox("Preview current binning")
        self.preview_binning_check.setChecked(True)
        self.preview_binning_check.toggled.connect(self._preview_binning_changed)
        self.prev_frame_btn = QPushButton("◀")
        self.prev_frame_btn.clicked.connect(self._previous_input_frame)
        self.play_btn = QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._toggle_input_playback)
        self.next_frame_btn = QPushButton("▶")
        self.next_frame_btn.clicked.connect(self._next_input_frame)
        self.frame_spin = QSpinBox(); self.frame_spin.setRange(0, 0)
        self.frame_spin.valueChanged.connect(self._frame_spin_changed)
        toolbar.addWidget(QLabel("View")); toolbar.addWidget(self.input_source_combo); toolbar.addWidget(self.preview_binning_check)
        toolbar.addWidget(self.prev_frame_btn); toolbar.addWidget(self.play_btn); toolbar.addWidget(self.next_frame_btn)
        toolbar.addStretch(); toolbar.addWidget(QLabel("Index")); toolbar.addWidget(self.frame_spin)
        browser_layout.addLayout(toolbar)
        browser_layout.addWidget(self.input_view, 1)
        navigation = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal); self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._frame_slider_changed)
        self.frame_position_label = QLabel("0 / 0")
        navigation.addWidget(self.frame_slider, 1); navigation.addWidget(self.frame_position_label)
        browser_layout.addLayout(navigation)
        self.input_stats_label = QLabel("Select an HDF5 file to inspect Projection, Flat and Dark frames")
        self.input_stats_label.setWordWrap(True)
        browser_layout.addWidget(self.input_stats_label)
        tabs.addTab(browser, "Input data")
        reconstruction = QWidget(); reconstruction_layout = QVBoxLayout(reconstruction)
        reconstruction_layout.addWidget(self.result_view, 1)
        result_nav = QHBoxLayout()
        self.previous_result_btn = QPushButton("◀")
        self.previous_result_btn.clicked.connect(lambda: self.result_slider.setValue(max(0, self.result_slider.value() - 1)))
        self.next_result_btn = QPushButton("▶")
        self.next_result_btn.clicked.connect(lambda: self.result_slider.setValue(min(self.result_slider.maximum(), self.result_slider.value() + 1)))
        self.result_slider = QSlider(Qt.Orientation.Horizontal); self.result_slider.setRange(0, 0); self.result_slider.setEnabled(False)
        self.result_slider.valueChanged.connect(self._result_slider_changed)
        self.result_position_label = QLabel("0 / 0")
        result_nav.addWidget(self.previous_result_btn); result_nav.addWidget(self.result_slider, 1)
        result_nav.addWidget(self.next_result_btn); result_nav.addWidget(self.result_position_label)
        reconstruction_layout.addLayout(result_nav)
        tabs.addTab(reconstruction, "Reconstruction")
        self.reconstruction_tab = reconstruction
        center_search = QWidget(); search_layout = QVBoxLayout(center_search)
        self.center_search_view = ImageView(); search_layout.addWidget(self.center_search_view, 1)
        candidate_nav = QHBoxLayout()
        self.center_candidate_slider = QSlider(Qt.Orientation.Horizontal); self.center_candidate_slider.setRange(0, 0)
        self.center_candidate_slider.valueChanged.connect(self._center_candidate_changed)
        self.center_candidate_label = QLabel("No center candidates")
        candidate_nav.addWidget(self.center_candidate_slider, 1); candidate_nav.addWidget(self.center_candidate_label)
        search_layout.addLayout(candidate_nav)
        self.apply_center_btn = QPushButton("Apply Selected Center")
        self.apply_center_btn.setObjectName("primaryButton"); self.apply_center_btn.setEnabled(False)
        self.apply_center_btn.clicked.connect(self._apply_selected_center)
        search_layout.addWidget(self.apply_center_btn)
        tabs.addTab(center_search, "Rotation Center Search")
        self.center_search_tab = center_search
        self.center_tabs = tabs
        return tabs

    def _right_panel(self) -> QWidget:
        panel = QWidget(); layout = QVBoxLayout(panel)
        box = QGroupBox("Current job"); form = QVBoxLayout(box)
        self.state_label = QLabel("Ready"); self.state_label.setObjectName("jobState")
        self.progress = QProgressBar(); self.progress.setRange(0, 0); self.progress.hide()
        self.latest_label = QLabel("No reconstructed slice yet"); self.latest_label.setWordWrap(True)
        self.stop_btn = QPushButton("Safe stop"); self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self.runner.stop)
        form.addWidget(self.state_label); form.addWidget(self.progress); form.addWidget(self.latest_label)
        self.gpu_name=QLabel('Detecting GPUs…');self.gpu_memory=QLabel('VRAM: —')
        self.gpu_usage=QProgressBar();self.gpu_usage.setRange(0,100);self.gpu_usage.setValue(0)
        self.gpu_vram=QProgressBar();self.gpu_vram.setTextVisible(False);self.gpu_vram.setRange(0,1);self.gpu_vram.setValue(0)
        refresh=QPushButton('Refresh GPUs');refresh.clicked.connect(lambda:self.gpu_monitor.refresh())
        for widget in (self.gpu_name,self.gpu_usage,self.gpu_memory,self.gpu_vram,refresh,self.stop_btn):form.addWidget(widget)
        layout.addWidget(box)
        results = QGroupBox("Results"); rlayout = QVBoxLayout(results)
        self.result_combo = QComboBox(); self.result_combo.currentIndexChanged.connect(self._selected_result)
        rlayout.addWidget(self.result_combo)
        layout.addWidget(results)
        layout.addWidget(self._export_panel())
        layout.addStretch()
        return panel

    def _choose_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select exchange HDF5", "D:\\", "HDF5 (*.h5 *.hdf5)")
        if not path:
            return
        self.input_edit.setText(path)
        self.output_edit.setText(str(Path(path).parent / f"{Path(path).stem}_reconstruction"))
        try:
            self.info = inspect_hdf5(path)
            start, end = self.info.suggested_rows
            self.start_spin.setValue(start); self.end_spin.setValue(end); self.center_spin.setValue(self.info.suggested_center)
            self.try_row_spin.setRange(0, self.info.data_shape[1] - 1); self.try_row_spin.setValue((self.info.data_shape[1] - 1) // 2)
            self._update_search_tooltips()
            warning = "\n" + "\n".join("WARNING: " + w for w in self.info.warnings) if self.info.warnings else ""
            self.file_info.setText(
                f"data {self.info.data_shape} {self.info.data_dtype}\n"
                f"theta {self.info.theta_min:.3g}..{self.info.theta_max:.3g} {self.info.theta_unit}"
                f" | uniform={self.info.theta_uniform}{warning}"
            )
            self._preview_cache.clear()
            self.input_source_combo.setCurrentIndex(0)
            self._configure_input_navigation(self.info.data_shape[0] // 2)
            self._effective_geometry_changed()
            self._load_input_frame()
            self._export_source_changed()
            if not self.export_output_edit.text():
                self.export_output_edit.setText(str(Path(path).parent / f"{Path(path).stem}_exports"))
            self.append_log(f"Validated HDF5: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "HDF5 validation failed", str(exc))

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory", "D:\\")
        if path:
            self.output_edit.setText(path)

    def _configure_input_navigation(self, preferred_index: int = 0) -> None:
        if not self.info:
            return
        count = frame_count(self.info, self.input_source_combo.currentData())
        index = max(0, min(preferred_index, count - 1))
        for widget in (self.frame_slider, self.frame_spin):
            widget.blockSignals(True); widget.setRange(0, count - 1); widget.setValue(index); widget.blockSignals(False)
        self.frame_position_label.setText(f"{index + 1} / {count}")

    def _input_source_changed(self) -> None:
        self.play_btn.setChecked(False)
        self._configure_input_navigation(0)
        self._schedule_input_frame()

    def _preview_binning_changed(self) -> None:
        self._preview_cache.clear()
        self._schedule_input_frame()

    def _frame_slider_changed(self, value: int) -> None:
        self.frame_spin.blockSignals(True); self.frame_spin.setValue(value); self.frame_spin.blockSignals(False)
        self._update_frame_position(value); self._schedule_input_frame()

    def _frame_spin_changed(self, value: int) -> None:
        self.frame_slider.setValue(value)

    def _update_frame_position(self, value: int) -> None:
        self.frame_position_label.setText(f"{value + 1} / {self.frame_slider.maximum() + 1}")

    def _schedule_input_frame(self) -> None:
        if self.info:
            self.preview_load_timer.start()

    def _load_input_frame(self) -> None:
        if not self.info:
            return
        source, index = self.input_source_combo.currentData(), self.frame_slider.value()
        exponent = self.binning_combo.currentData() if self.preview_binning_check.isChecked() else 0
        key = (source, index, exponent)
        try:
            if key not in self._preview_cache:
                image, angle = read_input_frame(self.info.path, source, index)
                factor = 2 ** exponent
                if factor > 1:
                    height, width = image.shape
                    height, width = (height // factor) * factor, (width // factor) * factor
                    image = image[:height, :width].reshape(
                        height // factor, factor, width // factor, factor
                    ).mean(axis=(1, 3))
                self._preview_cache[key] = image, angle
                if len(self._preview_cache) > 24:
                    self._preview_cache.pop(next(iter(self._preview_cache)))
            image, angle = self._preview_cache[key]
            self.input_view.show_array(image)
            finite = image[np.isfinite(image)]
            angle_text = "" if angle is None else f" | angle {angle:.4g} {self.info.theta_unit}"
            self.input_stats_label.setText(
                f"{self.input_source_combo.currentText()} {index + 1}/{self.frame_slider.maximum() + 1}{angle_text}"
                f" | {image.shape[1]}×{image.shape[0]} | min {finite.min():.5g} | max {finite.max():.5g}"
                f" | mean {finite.mean():.5g} | std {finite.std():.5g}"
            )
        except Exception as exc:
            self.input_stats_label.setText(f"Preview error: {exc}")
            self.append_log(f"Input preview warning: {exc}")

    def _previous_input_frame(self) -> None:
        self.frame_slider.setValue(max(0, self.frame_slider.value() - 1))

    def _next_input_frame(self) -> None:
        value = self.frame_slider.value() + 1
        if value > self.frame_slider.maximum():
            value = 0
        self.frame_slider.setValue(value)

    def _toggle_input_playback(self, playing: bool) -> None:
        self.play_btn.setText("Pause" if playing else "Play")
        self.play_timer.start() if playing else self.play_timer.stop()

    def _export_count(self) -> int:
        source = self.export_source_combo.currentData()
        if source == "reconstruction":
            return len(self.results)
        return frame_count(self.info, source) if self.info else 0

    def _export_source_changed(self) -> None:
        source = self.export_source_combo.currentData()
        if source != "reconstruction" and hasattr(self, "input_source_combo"):
            index = self.input_source_combo.findData(source)
            if index >= 0:
                self.input_source_combo.setCurrentIndex(index)
        count = self._export_count()
        maximum = max(0, count - 1)
        self.export_start_spin.setRange(0, maximum)
        self.export_end_spin.setRange(1, max(1, count))
        self.export_start_spin.setValue(0); self.export_end_spin.setValue(max(1, count))
        if source == "reconstruction" and self.results:
            self.export_output_edit.setText(str(self.results[0].parent / "png_exports"))

    def _export_scope_changed(self) -> None:
        self.export_range_widget.setVisible(self.export_scope_combo.currentData() == "range")

    def _export_format_changed(self) -> None:
        if self.export_format_combo.currentData() == "png" and self.export_mode_combo.currentData() in ("preserve", "float32"):
            self.export_mode_combo.setCurrentIndex(self.export_mode_combo.findData("display8"))

    def _choose_export_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select export output directory", "D:\\")
        if path:
            self.export_output_edit.setText(path)

    def _export_indices(self, count: int) -> list[int]:
        scope = self.export_scope_combo.currentData()
        if scope == "all":
            return list(range(count))
        if scope == "range":
            start, end = self.export_start_spin.value(), self.export_end_spin.value()
            if not 0 <= start < end <= count:
                raise ValueError(f"Invalid export range [{start},{end}) for {count} frames")
            return list(range(start, end))
        source = self.export_source_combo.currentData()
        if source == "reconstruction":
            current = self.result_combo.currentIndex()
        else:
            current = self.frame_slider.value()
        return [max(0, min(current, count - 1))]

    def _start_export(self) -> None:
        try:
            if self.export_thread and self.export_thread.isRunning():
                raise RuntimeError("An export is already running")
            source = self.export_source_combo.currentData()
            count = self._export_count()
            if count < 1:
                raise ValueError("No frames are available for this export source")
            output_format = self.export_format_combo.currentData()
            data_mode = self.export_mode_combo.currentData()
            if output_format == "png" and data_mode in ("preserve", "float32"):
                raise ValueError("PNG requires Display 8-bit or Display 16-bit mode")
            output_text = self.export_output_edit.text().strip()
            if not output_text:
                raise ValueError("Select an export output directory")
            output_dir = Path(output_text)
            indices = self._export_indices(count)
            reconstruction_files = None
            input_path = self.info.path if self.info else Path()
            if source == "reconstruction":
                reconstruction_files = [self.results[i] for i in indices]
                input_path = reconstruction_files[0].parent
            existing = list(output_dir.glob("*.png")) + list(output_dir.glob("*.tif")) + list(output_dir.glob("*.tiff")) if output_dir.exists() else []
            if existing and QMessageBox.question(
                self, "Existing exports", f"{output_dir} already contains image files. Overwrite matching names?"
            ) != QMessageBox.StandardButton.Yes:
                return
            job = ExportJob(source, input_path, output_dir, indices, output_format, data_mode, reconstruction_files)
            self.export_thread = QThread(self)
            self.export_worker = ExportWorker(job)
            self.export_worker.moveToThread(self.export_thread)
            self.export_thread.started.connect(self.export_worker.run)
            self.export_worker.progress.connect(self._export_progress_changed)
            self.export_worker.log.connect(self.append_log)
            self.export_worker.finished.connect(self._export_finished)
            self.export_worker.finished.connect(self.export_thread.quit)
            self.export_thread.finished.connect(self._export_thread_finished)
            self.export_btn.setEnabled(False); self.export_cancel_btn.setEnabled(True)
            self.export_progress.setRange(0, len(indices)); self.export_progress.setValue(0)
            self.append_log(f"Export started: {source}, {len(indices)} frames -> {output_dir}")
            self.export_thread.start()
        except Exception as exc:
            QMessageBox.critical(self, "Cannot start export", str(exc))

    def _export_progress_changed(self, done: int, total: int, name: str) -> None:
        self.export_progress.setRange(0, total); self.export_progress.setValue(done)
        self.export_progress.setFormat(f"{done}/{total}  {name}")

    def _cancel_export(self) -> None:
        if self.export_worker:
            self.export_worker.cancel_requested = True
            self.export_cancel_btn.setEnabled(False)
            self.append_log("Export cancellation requested...")

    def _export_finished(self, ok: bool, message: str) -> None:
        self.append_log(message)
        self.export_btn.setEnabled(True); self.export_cancel_btn.setEnabled(False)
        if ok:
            self.export_progress.setFormat("Export completed")

    def _export_thread_finished(self) -> None:
        if self.export_worker:
            self.export_worker.deleteLater()
        if self.export_thread:
            self.export_thread.deleteLater()
        self.export_worker = None; self.export_thread = None

    def _effective_geometry_changed(self) -> None:
        if not self.info:
            return
        factor = 2 ** self.binning_combo.currentData()
        effective_start = self.start_spin.value() // factor
        effective_end = effective_start + (self.end_spin.value() - self.start_spin.value()) // factor
        size = self.info.data_shape[2] // factor
        self.effective_geometry_label.setText(
            f"Effective: center {self.center_spin.value()/factor:.4g} | rows [{effective_start}, {effective_end})"
            f" | output {size}×{size}"
        )
        self._preview_cache.clear()
        self._schedule_input_frame()

    def _mode_changed(self) -> None:
        mode = self.mode_combo.currentData(); previous = self.method_combo.currentData()
        self.method_combo.blockSignals(True); self.method_combo.clear()
        methods = TOMO_METHODS if mode == GeometryMode.TOMOGRAPHY else (ReconMethod.LAMINO_FOURIERREC, ReconMethod.LAMINO_LINEREC)
        for method in methods:self.method_combo.addItem(METHOD_LABELS[method],method)
        index=self.method_combo.findData(previous);self.method_combo.setCurrentIndex(index if index>=0 else 0);self.method_combo.blockSignals(False)
        lamino=mode==GeometryMode.LAMINOGRAPHY
        self.params_box.setTitle('Laminography parameters' if lamino else 'Tomography parameters')
        self.params_form.setRowVisible(self.lamino_row,lamino)
        self.lamino_spin.setEnabled(lamino);self.lamino_settings_btn.setEnabled(lamino);self.find_angle_btn.setEnabled(lamino)
        self.center_settings_btn.setEnabled(True);self._method_changed()

    def _method_changed(self) -> None:
        method=self.method_combo.currentData();iterative=method in (ReconMethod.SIRT_CUDA,ReconMethod.CGLS_CUDA)
        self.iter_spin.setEnabled(iterative);self.params_form.setRowVisible(self.iter_spin,iterative)
        if method in (ReconMethod.LINEREC,ReconMethod.LAMINO_LINEREC) and self.chunk_spin.value()<4:self.chunk_spin.setValue(4)

    def _start_search(self, kind: str) -> None:
        index=self.type_combo.findData('angle_try' if kind=='angle' else 'center_try')
        self.type_combo.setCurrentIndex(index)
        try:self._start_job()
        finally:self.type_combo.setCurrentIndex(self.type_combo.findData('full'))

    def _job(self, run_type: str | None = None) -> ReconJob:
        selected_type = self.type_combo.currentData()
        backend_type = run_type or ("full" if selected_type == "full" else "try")
        method=ReconMethod(self.method_combo.currentData())
        if backend_type=='try':
            if self.mode_combo.currentData()==GeometryMode.LAMINOGRAPHY:method=ReconMethod.LAMINO_LINEREC
            elif method in (ReconMethod.FBP_CUDA,ReconMethod.SIRT_CUDA,ReconMethod.CGLS_CUDA):method=ReconMethod.FOURIERREC
        return ReconJob(
            input_path=Path(self.input_edit.text()), output_dir=Path(self.output_edit.text()),
            mode=GeometryMode(self.mode_combo.currentData()),
            method=method,
            reconstruction_type=backend_type,
            search_kind="angle" if selected_type == "angle_try" else "center",
            start_row=self.start_spin.value(), end_row=self.end_spin.value(), try_row=self.try_row_spin.value(), center=self.center_spin.value(),
            gpu=self.gpu_spin.value(), chunk_size=self.chunk_spin.value(), iterations=self.iter_spin.value(),
            lamino_angle=self.lamino_spin.value(), binning=self.binning_combo.currentData(),
            center_search_width=self.search_width_spin.value(), center_search_step=self.search_step_spin.value(),
            lamino_search_width=self.lamino_search_width_spin.value(), lamino_search_step=self.lamino_search_step_spin.value(),
        )

    def _preview_command(self) -> None:
        try:
            self.append_log(display_command(build_wsl_argv(self._job())))
        except Exception as exc:
            QMessageBox.warning(self, "Invalid parameters", str(exc))

    def _start_job(self) -> None:
        if self.runner.running or self.analysis_runner.running:
            QMessageBox.warning(self,'Job running','Wait for the current job or use Safe Stop.');return
        try:
            selected_type = self.type_combo.currentData()
            self.log_box.clear()
            run_type = "full" if selected_type == "full" else "try"
            job = self._job(run_type); job.validate()
            self.active_selected_type = selected_type
            # Re-read the exact path for every run. A manually changed input
            # must never reuse dimensions or theta metadata from the last HDF5.
            current_info = inspect_hdf5(job.input_path)
            input_changed = self.info is None or self.info.path.resolve() != current_info.path.resolve()
            self.info = current_info
            nrows = self.info.data_shape[1]
            if not 0 <= job.start_row < job.end_row <= nrows:
                raise ValueError(f"Row range [{job.start_row}, {job.end_row}) is outside this input (0..{nrows})")
            if run_type == "try" and not 0 <= job.try_row < nrows:
                raise ValueError(f"Try detector row {job.try_row} is outside this input (0..{nrows - 1})")
            if input_changed:
                self._preview_cache.clear()
                self._configure_input_navigation(min(self.frame_slider.value(), self.info.data_shape[0] - 1))
                self._schedule_input_frame()
                self.append_log(f"Input changed; revalidated HDF5 metadata: {job.input_path}")
            if selected_type == "center_try" and job.method in (ReconMethod.FBP_CUDA, ReconMethod.SIRT_CUDA, ReconMethod.CGLS_CUDA):
                raise ValueError("Rotation Center Search currently uses TomocuPy FourierRec, LpRec or LineRec")
            if job.method == ReconMethod.LPREC:
                job.uniform_angle_approximation = True
            # TomocuPy was previously called with --clear-folder False. Reusing
            # one directory therefore mixed old TIFFs with the new run. Keep
            # the user-selected folder as a root and isolate every execution.
            base_output = job.output_dir
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._run_serial += 1
            job.output_dir = base_output / f"run_{stamp}_{self._run_serial:04d}_{selected_type}_{job.method.value}"
            self.active_output_dir = job.output_dir
            self.active_job = job
            if job.method == ReconMethod.LPREC:
                self.append_log(
                    "LpRec will automatically create an external-link HDF5 view with theta in [0,180) at "
                    f"{job.output_dir / 'equal_angle_view.h5'}. The original input remains unchanged."
                )
            self.results.clear(); self.result_combo.clear(); self.center_candidates.clear()
            self.result_slider.setRange(0, 0); self.result_slider.setEnabled(False)
            self.result_position_label.setText("0 / 0")
            waiting_for = "angle" if selected_type == "angle_try" else "center"
            self.result_view.reset()
            self.center_search_view.reset(f"Waiting for {waiting_for} candidates")
            self.center_candidate_slider.setRange(0, 0); self.center_candidate_label.setText(f"Waiting for {waiting_for} candidates...")
            self.apply_center_btn.setEnabled(False)
            if run_type == "try":
                self.search_initial_center = job.center
                self.search_candidate_kind = job.search_kind
                self.center_tabs.setCurrentWidget(self.center_search_tab)
            self.latest_label.setText(f"Output for this run: {job.output_dir}")
            self.append_log(f"Isolated run output: {job.output_dir}")
            self.runner.start(job)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot start reconstruction", str(exc))

    def append_log(self, text: str) -> None:
        self.log_box.appendPlainText(text)

    def _set_state(self, state: str) -> None:
        self.state_label.setText(state)
        running = state in ("Running", "Stopping")
        self.progress.setVisible(running); self.stop_btn.setEnabled(state == "Running")
        self._update_busy()

    def _new_result(self, path: str) -> None:
        result = Path(path)
        match = re.search(r"_center(-?\d+(?:\.\d+)?)", result.stem)
        angle_match = re.search(r"_angle(-?\d+(?:\.\d+)?)", result.stem)
        if not match and angle_match:
            match = angle_match; self.search_candidate_kind = "angle"
        if match:
            center = float(match.group(1))
            existing = next((i for i, (_, path) in enumerate(self.center_candidates) if path == result), None)
            if existing is None:
                self.center_candidates.append((center, result))
                self.center_candidates.sort(key=lambda item: item[0])
            index = next(i for i, item in enumerate(self.center_candidates) if item[1] == result)
            self.center_candidates[index] = (center, result)
            self.center_candidate_slider.setRange(0, len(self.center_candidates) - 1)
            self.center_candidate_slider.setValue(index); self._show_center_candidate(index)
            self.apply_center_btn.setEnabled(True); self.center_tabs.setCurrentWidget(self.center_search_tab)
            return
        if result not in self.results:
            self.results.append(result)
            self.results.sort(key=lambda item: item.name.lower())
        index = self.results.index(result)
        self.result_combo.blockSignals(True)
        self.result_combo.clear()
        for item in self.results:
            self.result_combo.addItem(item.name, item)
        self.result_combo.setCurrentIndex(index)
        self.result_combo.blockSignals(False)
        self.result_slider.setEnabled(True)
        self.result_slider.setRange(0, len(self.results) - 1)
        self._batching_results=True
        self.result_slider.setValue(index)
        self._selected_result(index)
        self._batching_results=False
        self.latest_label.setText(str(result))
        if self.export_source_combo.currentData() == "reconstruction":
            self._export_source_changed()
    def _selected_result(self, index: int) -> None:
        if index >= 0:
            path = self.result_combo.itemData(index)
            if path:
                self.result_slider.blockSignals(True); self.result_slider.setValue(index); self.result_slider.blockSignals(False)
                self.result_position_label.setText(f"{index + 1} / {len(self.results)}  |  {Path(path).name}")
                if self._batching_results or self.runner.running:self.result_preview_timer.start()
                else:self._display_selected_result()

    def _display_selected_result(self) -> None:
        path=self.result_combo.currentData()
        if not path:return
        try:self.result_view.show_tiff(path);self.center_tabs.setCurrentWidget(self.reconstruction_tab)
        except Exception as error:self.append_log(str(error))

    def _result_slider_changed(self, index: int) -> None:
        if not 0 <= index < self.result_combo.count():
            return
        self.result_combo.setCurrentIndex(index)

    def _center_candidate_changed(self, index: int) -> None:
        self._show_center_candidate(index)

    def _show_center_candidate(self, index: int) -> None:
        if not 0 <= index < len(self.center_candidates):
            return
        center, path = self.center_candidates[index]
        try:
            self.center_search_view.show_tiff(path)
            quantity = f"lamino angle {center:.3f}°" if self.search_candidate_kind == "angle" else f"center {center:.3f} px"
            self.center_candidate_label.setText(f"Candidate {index + 1}/{len(self.center_candidates)} | row {self.try_row_spin.value()} | {quantity}")
        except Exception as exc:
            self.append_log(f"Center candidate preview warning: {exc}")

    def _apply_selected_center(self) -> None:
        index = self.center_candidate_slider.value()
        if not 0 <= index < len(self.center_candidates):
            return
        center, path = self.center_candidates[index]
        if self.search_candidate_kind == "angle":
            self.lamino_spin.setValue(center)
        else:
            self.center_spin.setValue(center)
        record = {"input": self.input_edit.text(), "detector_row": self.try_row_spin.value(),
                  "method": getattr(self,"active_job",self._job()).method.value, "binning": 2 ** self.binning_combo.currentData(),
                  "search_center": self.search_initial_center, "search_width": self.search_width_spin.value(),
                  "search_step": self.search_step_spin.value(), "selected_center": center,
                  "candidate_file": str(path), "selected_at": datetime.now().isoformat()}
        if self.search_candidate_kind == "angle":
            record.update(lamino_search_width=self.lamino_search_width_spin.value(),
                          lamino_search_step=self.lamino_search_step_spin.value(), selected_lamino_angle=center)
        output_name = "selected_lamino_angle.json" if self.search_candidate_kind == "angle" else "selected_rotation_center.json"
        output = (self.active_output_dir or Path(self.output_edit.text())) / output_name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        label = "lamino angle" if self.search_candidate_kind == "angle" else "rotation center"
        self.append_log(f"Applied {label}: {center:.3f} from detector row {self.try_row_spin.value()}")
        self.latest_label.setText(f"Selected {label}: {center:.3f}")

    def _job_finished(self, ok: bool, message: str) -> None:
        self.append_log(message)
        if ok and self.active_selected_type in ("center_try", "angle_try") and self.center_candidates:
            index = min(max(self.center_candidate_slider.value(), 0), len(self.center_candidates) - 1)
            self.center_candidate_slider.setValue(index)
            self._show_center_candidate(index)
            self.center_tabs.setCurrentWidget(self.center_search_tab)
        elif ok and self.results:
            self.result_preview_timer.stop();self._display_selected_result()
            self.center_tabs.setCurrentWidget(self.reconstruction_tab)

    def closeEvent(self,event):
        if self.runner.running or self.analysis_runner.running or self.export_thread is not None:
            self.runner.stop();self.analysis_runner.stop();self._cancel_export();event.ignore();QTimer.singleShot(500,self.close);return
        if hasattr(self,'gpu_monitor'):
            self.gpu_monitor.timer.stop()
            if self.gpu_monitor.process.state():self.gpu_monitor.process.kill();self.gpu_monitor.process.waitForFinished(1000)
        event.accept()

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget { background:#0b1018; color:#eaf0fa; font-family:'Segoe UI'; font-size:13px; }
            QGroupBox { border:1px solid #293348; border-radius:8px; margin-top:12px; padding-top:10px; font-weight:600; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; }
            QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QPlainTextEdit { background:#121a28; border:1px solid #303c52; border-radius:5px; padding:6px; }
            QLineEdit:disabled,QComboBox:disabled,QSpinBox:disabled,QDoubleSpinBox:disabled,QPlainTextEdit:disabled {
                background:#0d131d; color:#627087; border-color:#202a3a;
            }
            QSpinBox,QDoubleSpinBox { padding-right:26px; }
            QSpinBox::up-button,QSpinBox::down-button,QDoubleSpinBox::up-button,QDoubleSpinBox::down-button { width:24px; }
            QPushButton { background:#1a2637; border:1px solid #34445e; border-radius:6px; padding:8px 12px; }
            QPushButton:hover { background:#23344c; }
            QPushButton:disabled { background:#0d131d; color:#5b687d; border-color:#202a3a; }
            QPushButton#primaryButton { background:#16c79a; color:#07120f; font-weight:700; }
            QPushButton#primaryButton:disabled { background:#0d131d; color:#5b687d; border-color:#202a3a; }
            QScrollBar:vertical { background:#0d131d; width:14px; margin:2px; border:none; border-radius:7px; }
            QScrollBar::handle:vertical { background:#34445e; min-height:34px; border-radius:5px; }
            QScrollBar::handle:vertical:hover { background:#16c79a; }
            QScrollBar::handle:vertical:pressed { background:#12a981; }
            QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0px; background:transparent; border:none; }
            QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical { background:transparent; }
            QLabel#title { font-size:24px; font-weight:700; }
            QLabel#statusPill { background:#15253a; color:#7de3c8; border-radius:10px; padding:6px 12px; }
            QLabel#jobState { font-size:22px; font-weight:700; color:#7de3c8; }
            QTabWidget::pane { border:1px solid #293348; }
        """)
