from __future__ import annotations
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt,QTimer,QUrl
from PySide6.QtGui import QDesktopServices,QImage,QPixmap
from PySide6.QtWidgets import (QCheckBox,QDialog,QFileDialog,QGroupBox,QHBoxLayout,QLabel,QLineEdit,
    QMessageBox,QProgressBar,QPushButton,QScrollArea,QSplitter,QVBoxLayout,QWidget)
from .analysis_settings import AnalysisSettings
from .hdf5_inspector import inspect_hdf5
from .image_view import ImageView
from .runtime import save_settings
from .models import GeometryMode

class AnalysisUiMixin:
    def _build_analysis_page(self):
        self.analysis_output=None;self.analysis_status={};self._analysis_image_key=None;self._curve_signature=None
        self.analysis_started=0.;self.gpus=[]
        page=QSplitter(Qt.Orientation.Horizontal)
        left=QWidget();layout=QVBoxLayout(left);layout.setContentsMargins(0,0,4,0)
        task=QGroupBox('Task');form=QVBoxLayout(task)
        form.addWidget(QLabel('Input HDF5'))
        row=QHBoxLayout();self.analysis_input=QLineEdit();self.analysis_browse=QPushButton('Browse')
        self.analysis_browse.clicked.connect(self._choose_analysis_input);row.addWidget(self.analysis_input,1);row.addWidget(self.analysis_browse);form.addLayout(row)
        form.addWidget(QLabel('Analysis Target'));self.analysis_prompt=QLineEdit('particle');form.addWidget(self.analysis_prompt)
        row=QHBoxLayout();self.segment_all=QCheckBox('Segment All');self.segment_all.setChecked(True);row.addWidget(self.segment_all);row.addStretch()
        self.analysis_settings_btn=QPushButton('⚙ Settings');self.analysis_settings_btn.clicked.connect(self._analysis_settings)
        row.addWidget(self.analysis_settings_btn);form.addLayout(row);layout.addWidget(task)
        row=QHBoxLayout();self.analysis_start=QPushButton('Start Analysis');self.analysis_start.setObjectName('primaryButton');self.analysis_start.clicked.connect(self._start_analysis)
        self.analysis_stop=QPushButton('Safe Stop');self.analysis_stop.setEnabled(False);self.analysis_stop.clicked.connect(self._stop_analysis)
        row.addWidget(self.analysis_start,1);row.addWidget(self.analysis_stop,1);layout.addLayout(row);layout.addStretch()
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(left);scroll.setMinimumWidth(305)
        page.addWidget(scroll)
        right=QWidget();r=QVBoxLayout(right);r.setContentsMargins(5,0,0,0)
        progress=QGroupBox();p=QVBoxLayout(progress)
        row=QHBoxLayout();self.analysis_stage=QLabel('Ready');self.analysis_stage.setStyleSheet('font-weight:600;')
        self.analysis_round_progress=QLabel();self.analysis_elapsed=QLabel('Elapsed 00:00')
        row.addWidget(self.analysis_stage);row.addStretch();row.addWidget(self.analysis_round_progress);row.addWidget(self.analysis_elapsed);p.addLayout(row)
        self.analysis_progress=QProgressBar();self.analysis_progress.setRange(0,1);self.analysis_progress.setValue(0);p.addWidget(self.analysis_progress);r.addWidget(progress)
        viewing=QGroupBox();v=QVBoxLayout(viewing)
        row=QHBoxLayout();row.addWidget(QLabel('Best Reconstruction'));row.addStretch()
        self.analysis_open=QPushButton('Open Output');self.analysis_open.clicked.connect(self._open_analysis_output);row.addWidget(self.analysis_open);v.addLayout(row)
        self.analysis_view=ImageView();self.analysis_view.setMinimumSize(320,220);self.analysis_view.reset('Start analysis to display the best reconstruction');v.addWidget(self.analysis_view,1)
        row=QHBoxLayout();self.analysis_center=QLabel('Rotation Center: —');self.analysis_center.setStyleSheet('font-size:19px;font-weight:600;color:#a6e8d7;')
        self.analysis_parameters=QLabel('Round: —     Step: —     Slice: —');row.addWidget(self.analysis_center);row.addStretch();row.addWidget(self.analysis_parameters);v.addLayout(row)
        r.addWidget(viewing,1)
        chart_box=QGroupBox('Rotation Center Score');chart_layout=QVBoxLayout(chart_box)
        self.analysis_plot=pg.PlotWidget(background='#0d1723');self.analysis_plot.setMinimumHeight(160);self.analysis_plot.setMaximumHeight(240)
        self.analysis_plot.setLabel('bottom','Rotation Center',units='px');self.analysis_plot.setLabel('left','Confidence-weighted Score');self.analysis_plot.showGrid(x=True,y=True,alpha=.15)
        self.raw_curve=self.analysis_plot.plot([],[],pen=None,symbol='o',symbolSize=3,symbolBrush='#71869b',symbolPen=None)
        self.filtered_curve=self.analysis_plot.plot([],[],pen=pg.mkPen('#eab675',width=2))
        self.best_marker=pg.InfiniteLine(angle=90,pen=pg.mkPen('#6fe1bd',style=Qt.PenStyle.DashLine));self.best_marker.hide();self.analysis_plot.addItem(self.best_marker)
        chart_layout.addWidget(self.analysis_plot);r.addWidget(chart_box);page.addWidget(right);page.setSizes([350,1200]);page.setStretchFactor(1,1)
        self.analysis_preview_timer=QTimer(self);self.analysis_preview_timer.setInterval(int(self.settings['refresh_seconds']*1000));self.analysis_preview_timer.timeout.connect(self._refresh_analysis_preview)
        self.analysis_preview_timer.start()
        self.analysis_clock=QTimer(self);self.analysis_clock.setInterval(1000);self.analysis_clock.timeout.connect(self._analysis_tick);self.analysis_clock.start()
        return page

    def _choose_analysis_input(self):
        path,_=QFileDialog.getOpenFileName(self,'Select exchange HDF5',self.analysis_input.text() or 'D:/','HDF5 (*.h5 *.hdf5)')
        if path:
            try:inspect_hdf5(path)
            except Exception as error:QMessageBox.critical(self,'Invalid HDF5',str(error));return
            self.analysis_input.setText(path)

    def _analysis_settings(self):
        dialog=AnalysisSettings(self.settings,self.gpus,self)
        if dialog.exec()==QDialog.DialogCode.Accepted:
            self.settings=dialog.result_settings();save_settings(self.settings)
            self.analysis_preview_timer.setInterval(int(self.settings['refresh_seconds']*1000))
            self.gpu_spin.setValue(self.settings['gpu']);self._gpu_selection_changed();self._environment_label()

    def _environment_label(self):
        py=Path(self.settings['sam3_python'])
        env=py.parent.name if py.name.lower().endswith('.exe') else py.parent.parent.name
        self.backend_status.setText(f'Environment: tomocupy_astra / {env or "SAM3 not configured"}\nRepo: {self.settings["repo"]}')

    def _start_analysis(self):
        if self.runner.running or self.analysis_runner.running:
            QMessageBox.warning(self,'Job running','Wait for the current job or use Safe Stop.');return
        try:
            source=Path(self.analysis_input.text().strip())
            if not self.analysis_input.text().strip():raise ValueError('Select an input HDF5 file')
            info=inspect_hdf5(source)
            if not self.analysis_prompt.text().strip():raise ValueError('Enter an analysis target')
            for key in ('sam3_python','recon_python','checkpoint'):
                if not Path(self.settings[key]).is_file():raise ValueError(f'Configure {key} in Settings: {self.settings[key]}')
            if not (Path(self.settings['repo'])/'sam3').is_dir():raise ValueError('Select the SAM3 repository in Settings')
            if self.settings['gpu']<0:raise ValueError('Select an available NVIDIA GPU')
            start=self.settings['start_row'];end=self.settings['end_row'];factor=2**self.settings['binning']
            if start is not None and not 0<=start<end<=info.data_shape[1]:raise ValueError('Invalid reconstruction range')
            if start is not None and end-start<factor:raise ValueError('Selected range is smaller than the binning factor')
            parent=Path(self.settings['output_root']) if self.settings['output_root'] else source.parent
            output=parent/f'{source.stem}_analysis_{datetime.now():%Y%m%d_%H%M%S_%f}'
            config={**self.settings,'input':str(source.resolve()),'output':str(output.resolve()),'prompt':self.analysis_prompt.text().strip(),'segment_all':self.segment_all.isChecked()}
            self.log_box.clear();self.analysis_status={};self._analysis_image_key=None;self._curve_signature=None
            self.raw_curve.setData([],[]);self.filtered_curve.setData([],[]);self.best_marker.hide()
            self.analysis_center.setText('Rotation Center: —');self.analysis_parameters.setText('Round: —     Step: —     Slice: —')
            self.analysis_view.reset('Finding the best reconstruction…');self.analysis_output=output;self.analysis_started=time.monotonic()
            self.analysis_stage.setText('Starting');self.analysis_progress.setRange(0,0);self.analysis_elapsed.setText('Elapsed 00:00')
            self.append_log(f'Automatic analysis output: {output}')
            self.analysis_runner.start(config);self._update_busy()
        except Exception as error:
            self.analysis_stage.setText('Failed');self.analysis_progress.setRange(0,1);self.analysis_progress.setValue(0)
            self._update_busy();QMessageBox.critical(self,'Cannot start analysis',str(error))

    def _analysis_updated(self,status):
        self.analysis_status=status;self.analysis_stage.setText(status.get('stage','Running'))
        if self.analysis_runner.stopping:self.analysis_stage.setText('Stopping safely…')
        total=int(status.get('total',0));done=int(status.get('completed',0));round_number=status.get('round')
        self.analysis_round_progress.setText((f'Round {round_number} · ' if round_number else '')+f'{done} / {total}')
        if status.get('indeterminate'):self.analysis_progress.setRange(0,0)
        else:self.analysis_progress.setRange(0,max(1,total));self.analysis_progress.setValue(min(done,total))
        if status.get('error'):self.append_log(status['error'])

    def _refresh_analysis_preview(self):
        status=self.analysis_status;path=status.get('best_image');center=status.get('best_center')
        if path and center is not None:
            try:
                signature=(path,Path(path).stat().st_mtime_ns)
                if signature!=self._analysis_image_key:
                    image=QImage(path)
                    if image.isNull():return
                    self.analysis_view._pixmap=QPixmap.fromImage(image);self.analysis_view._fit()
                    self._analysis_image_key=signature
                # The displayed value always comes from the same status as the image.
                self.analysis_center.setText(f'Rotation Center: {center:.4f} px')
                self.analysis_parameters.setText(f'Round: {status.get("round","—")}     Step: {status.get("step","—")} px     Slice: {status.get("slice","—")}')
            except OSError:pass
        curve=status.get('curve_file')
        if curve:
            try:
                signature=Path(curve).stat().st_mtime_ns
                if signature!=self._curve_signature:
                    graph=json.loads(Path(curve).read_text(encoding='utf-8'));self._curve_signature=signature
                    self.raw_curve.setData(graph['centers'],graph['scores'])
                    values=graph['curve'];x=values['start']+np.arange(len(values['scores']))*values['step']
                    self.filtered_curve.setData(x,values['scores']);self.best_marker.setPos(graph['best_center']);self.best_marker.show()
            except (OSError,ValueError,KeyError):pass

    def _analysis_tick(self):
        if self.analysis_runner.running and self.analysis_started:
            seconds=int(time.monotonic()-self.analysis_started);self.analysis_elapsed.setText(f'Elapsed {seconds//60:02d}:{seconds%60:02d}')

    def _stop_analysis(self):
        self.analysis_runner.stop();self.analysis_stop.setEnabled(False);self.analysis_stage.setText('Stopping safely…')

    def _analysis_finished(self,ok,message):
        self.analysis_runner.poll();self._refresh_analysis_preview();self._analysis_tick()
        self.analysis_stage.setText(message);self.append_log(f'Analysis {message}: {self.analysis_output}');self._update_busy()
        if ok:
            try:
                summary=json.loads((self.analysis_output/'analysis_summary.json').read_text())
                if self.input_edit.text().strip() and Path(self.input_edit.text()).resolve()==Path(self.analysis_input.text()).resolve():
                    self.center_spin.setValue(summary['center_search']['recommended_center'])
            except (OSError,ValueError,KeyError):pass

    def _open_analysis_output(self):
        if self.analysis_output and self.analysis_output.exists():QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.analysis_output)))

    def _gpu_updated(self,gpus):
        self.gpus=gpus;self.gpu_spin.update_gpus(gpus);self._gpu_selection_changed()

    def _gpu_selection_changed(self):
        index=self.gpu_spin.value();self.settings['gpu']=index
        gpu=next((g for g in self.gpus if g['index']==index),None)
        if not gpu:
            self.gpu_name.setText('GPU unavailable');self.gpu_memory.setText('VRAM: —');return
        self.gpu_name.setText(f'GPU {index} · {gpu["name"].replace("NVIDIA ","").replace(" Generation","")}')
        self.gpu_usage.setValue(int(gpu['usage']));self.gpu_usage.setFormat('GPU: %p%')
        self.gpu_vram.setRange(0,int(gpu['total']));self.gpu_vram.setValue(int(gpu['used']))
        self.gpu_memory.setText(f'VRAM: {gpu["used"]/1024:.2f} / {gpu["total"]/1024:.0f} GB')
        self.statusBar().showMessage(f'{self.gpu_name.text()}   |   {gpu["usage"]:.0f}%   |   {self.gpu_memory.text()}')

    def _update_busy(self):
        manual=self.runner.running;automatic=self.analysis_runner.running;busy=manual or automatic
        for button in (self.run_btn,self.find_center_btn,self.analysis_start,self.analysis_settings_btn):button.setEnabled(not busy)
        self.find_angle_btn.setEnabled(not busy and self.mode_combo.currentData()==GeometryMode.LAMINOGRAPHY)
        self.gpu_spin.setEnabled(not busy);self.mode_combo.setEnabled(not busy)
        self.analysis_input.setEnabled(not automatic);self.analysis_browse.setEnabled(not automatic);self.analysis_prompt.setEnabled(not automatic);self.segment_all.setEnabled(not automatic)
        self.analysis_stop.setEnabled(automatic and not self.analysis_runner.stopping)
