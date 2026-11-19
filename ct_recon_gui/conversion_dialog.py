from pathlib import Path
import threading
import numpy as np
from PySide6.QtCore import QThread,Signal
from PySide6.QtWidgets import QDialog,QFormLayout,QHBoxLayout,QVBoxLayout,QLineEdit,QPushButton,QComboBox,QDoubleSpinBox,QSpinBox,QProgressBar,QLabel,QFileDialog,QMessageBox
from . import conversion as engine

class ConversionTask(QThread):
    progress=Signal(int,int,str)
    outcome=Signal(bool,str)
    def __init__(self,function,arguments,parent):
        super().__init__(parent);self.function=function;self.arguments=arguments;self.cancel_event=threading.Event()
    def run(self):
        try:
            result=self.function(**self.arguments,cancel=self.cancel_event.is_set,progress=self.progress.emit);self.outcome.emit(True,str(result))
        except Exception as error:self.outcome.emit(False,str(error))

class ConversionDialog(QDialog):
    def __init__(self,direction,parent=None,source=''):
        super().__init__(parent);self.direction=direction;self.task=None;self.closing=False;self.path_buttons={}
        self.setWindowTitle('Images → H5' if direction=='import' else 'H5 → Images');self.resize(650,540)
        layout=QVBoxLayout(self);self.form=QFormLayout();layout.addLayout(self.form)
        self.source=self.path_row('Image folder' if direction=='import' else 'HDF5',direction=='import',initial=source)
        self.source.editingFinished.connect(self.inspect)
        self.kind=QComboBox();self.kind.addItem('Projections','projections');self.kind.addItem('Reconstructed slices','reconstruction')
        self.state=QComboBox()
        from .input_state import LABELS
        self.state.addItem('Choose preparation…',None)
        for value,label in LABELS.items():self.state.addItem(label,value)
        self.theta_file=None
        if direction=='import':
            self.form.addRow('Image kind',self.kind);self.form.addRow('Data preparation',self.state)
            self.theta_mode=QComboBox();self.theta_mode.addItem('Generate: start + step','generate');self.theta_mode.addItem('Import CSV / TXT','file');self.form.addRow('Theta',self.theta_mode)
            self.theta_file=self.path_row('Theta file',False,theta=True)
            self.theta_start=QDoubleSpinBox();self.theta_start.setRange(-100000,100000);self.theta_start.setDecimals(8)
            self.theta_step=QDoubleSpinBox();self.theta_step.setRange(-100000,100000);self.theta_step.setDecimals(8);self.theta_step.setValue(1)
            self.form.addRow('Start angle',self.theta_start);self.form.addRow('Angle step',self.theta_step)
            self.units=QComboBox();self.units.addItems(['degrees','radians']);self.form.addRow('Theta units',self.units)
            self.flat=self.path_row('Flat images',True);self.dark=self.path_row('Dark images',True)
            self.kind.currentIndexChanged.connect(self.update_fields);self.state.currentIndexChanged.connect(self.update_fields);self.theta_mode.currentIndexChanged.connect(self.update_fields)
        else:
            self.dataset=QComboBox();self.form.addRow('Dataset',self.dataset);self.dataset.currentIndexChanged.connect(self.dataset_changed)
            self.start=QSpinBox();self.start.setRange(0,2147483646);self.end=QSpinBox();self.end.setRange(1,2147483647);self.stride=QSpinBox();self.stride.setRange(1,1000000)
            self.form.addRow('Start (included)',self.start);self.form.addRow('End (excluded)',self.end);self.form.addRow('Stride',self.stride)
            self.format=QComboBox();self.format.addItem('TIFF — preserve values','tiff');self.format.addItem('PNG — shared display range','png');self.form.addRow('Format',self.format)
            self.bits=QComboBox();self.bits.addItem('16 bit',16);self.bits.addItem('8 bit',8);self.form.addRow('PNG depth',self.bits)
            self.format.currentIndexChanged.connect(lambda:self.bits.setEnabled(self.format.currentData()=='png'));self.bits.setEnabled(False)
        self.output=self.path_row('Output H5' if direction=='import' else 'New output folder',direction!='import',save=True)
        self.info=QLabel('');self.info.setWordWrap(True);layout.addWidget(self.info)
        self.progress=QProgressBar();layout.addWidget(self.progress)
        row=QHBoxLayout();self.run_button=QPushButton('Convert');self.cancel_button=QPushButton('Close');row.addStretch();row.addWidget(self.run_button);row.addWidget(self.cancel_button);layout.addLayout(row)
        self.run_button.clicked.connect(self.start_task);self.cancel_button.clicked.connect(self.close)
        self.inspect();self.update_fields()
    def path_row(self,label,folder,save=False,theta=False,initial=''):
        edit=QLineEdit(initial);row=QHBoxLayout();row.addWidget(edit);button=QPushButton('Browse');row.addWidget(button);self.form.addRow(label,row)
        def choose():
            if folder:
                if save:
                    name,_=QFileDialog.getSaveFileName(self,'Choose a new output folder name',edit.text(),'All files (*)')
                else:name=QFileDialog.getExistingDirectory(self,label,edit.text())
            elif save:name,_=QFileDialog.getSaveFileName(self,label,edit.text(),'HDF5 (*.h5 *.hdf5)')
            else:name,_=QFileDialog.getOpenFileName(self,label,edit.text(),'Theta (*.csv *.txt)' if theta else 'HDF5 (*.h5 *.hdf5)')
            if name:edit.setText(name);self.inspect()
        button.clicked.connect(choose);self.path_buttons[edit]=button;return edit
    def inspect(self):
        if not hasattr(self,'info') or not self.source.text():return
        try:
            if self.direction=='import':
                self.files=engine.image_files(self.source.text());a=engine.read_image(self.files[0]) if self.files else None
                self.info.setText(f'{len(self.files)} images'+(f' · {a.shape} · {a.dtype}' if a is not None else ''))
            else:
                self.dataset.clear()
                for key,shape,dtype in engine.datasets(self.source.text()):self.dataset.addItem(f'/{key} · {shape} · {dtype}',(key,shape))
                self.info.setText('TIFF preserves pixel values. PNG uses one shared range for the selected stack.')
        except Exception as error:self.info.setText(str(error))
    def dataset_changed(self):
        data=self.dataset.currentData()
        if data:self.start.setValue(0);self.end.setValue(data[1][0] if len(data[1])==3 else 1)
    def update_fields(self):
        if self.direction!='import':return
        projections=self.kind.currentData()=='projections';generated=self.theta_mode.currentData()=='generate'
        for widget in (self.state,self.theta_mode,self.units):widget.setEnabled(projections)
        self.theta_file.setEnabled(projections and not generated)
        self.theta_start.setEnabled(projections and generated);self.theta_step.setEnabled(projections and generated)
        self.flat.setEnabled(projections and self.state.currentData()=='raw');self.dark.setEnabled(self.flat.isEnabled())
        for edit,button in self.path_buttons.items():button.setEnabled(edit.isEnabled())
    def start_task(self):
        if self.task:return
        try:
            if not self.output.text().strip():raise ValueError('Choose an output path')
            if self.direction=='import':
                files=engine.image_files(self.source.text());theta=None
                if self.kind.currentData()=='projections':
                    if self.theta_mode.currentData()=='generate':
                        if self.theta_step.value()==0:raise ValueError('Angle step cannot be zero')
                        theta=self.theta_start.value()+np.arange(len(files))*self.theta_step.value()
                    else:
                        header=Path(self.theta_file.text()).read_text(encoding='utf-8-sig').splitlines()[0].lower()
                        if header.startswith('#') and ('theta_degrees' in header or 'theta_radians' in header) and ('theta_'+self.units.currentText()) not in header:raise ValueError('Theta file units differ from the selected units')
                        theta=np.loadtxt(self.theta_file.text(),delimiter=',' if self.theta_file.text().lower().endswith('.csv') else None,ndmin=1)
                function=engine.images_to_h5;arguments=dict(files=files,destination=self.output.text(),kind=self.kind.currentData(),state=self.state.currentData() or ('attenuation' if self.kind.currentData()=='reconstruction' else ''),theta=theta,units=self.units.currentText(),flat=self.flat.text(),dark=self.dark.text())
            else:
                if not self.dataset.currentData():raise ValueError('Choose a dataset')
                function=engine.h5_to_images;arguments=dict(source=self.source.text(),key=self.dataset.currentData()[0],destination=self.output.text(),start=self.start.value(),end=self.end.value(),stride=self.stride.value(),format=self.format.currentData(),bits=self.bits.currentData())
            self.task=ConversionTask(function,arguments,self);self.task.progress.connect(self.update_progress);self.task.outcome.connect(self.outcome);self.task.finished.connect(self.task_finished)
            self.set_form_enabled(False);self._last_ok=False
            self.run_button.setEnabled(False);self.cancel_button.setText('Cancel');self.task.start()
        except Exception as error:QMessageBox.warning(self,'Conversion',str(error))
    def set_form_enabled(self,enabled):
        for i in range(self.form.rowCount()):
            item=self.form.itemAt(i,QFormLayout.ItemRole.FieldRole)
            if not item:continue
            if item.widget():item.widget().setEnabled(enabled)
            elif item.layout():
                for j in range(item.layout().count()):
                    widget=item.layout().itemAt(j).widget()
                    if widget:widget.setEnabled(enabled)
        if enabled:
            self.update_fields()
            if self.direction=='export':self.bits.setEnabled(self.format.currentData()=='png')
    def update_progress(self,done,total,stage):self.progress.setValue(int(100*done/max(1,total)));self.info.setText(f'{stage}: {done}/{total}')
    def outcome(self,ok,message):self._last_ok=ok;self.info.setText(('Completed: ' if ok else '')+message)
    def task_finished(self):
        self.task.deleteLater();self.task=None;self.cancel_button.setText('Close')
        if self.closing:self.accept()
        elif not self._last_ok:self.set_form_enabled(True);self.run_button.setEnabled(True)
    def closeEvent(self,event):
        if self.task:
            self.closing=True;self.task.cancel_event.set();self.info.setText('Cancelling…');event.ignore()
        else:super().closeEvent(event)
    def reject(self):
        if self.task:self.closing=True;self.task.cancel_event.set()
        else:super().reject()
