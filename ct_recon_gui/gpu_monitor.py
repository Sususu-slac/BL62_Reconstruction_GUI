from __future__ import annotations
import csv
from PySide6.QtCore import QObject,QProcess,QTimer,Signal
from PySide6.QtWidgets import QComboBox

class GpuSelector(QComboBox):
    def __init__(self):
        super().__init__();self.addItem('Detecting GPUs…',0)
    def value(self):return int(self.currentData() or 0)
    def setValue(self,value):
        index=self.findData(int(value))
        if index<0:self.addItem(f'GPU {value}',int(value));index=self.count()-1
        self.setCurrentIndex(index)
    def update_gpus(self,gpus):
        selected=self.value();self.blockSignals(True);self.clear()
        for g in gpus:self.addItem(f'{g["index"]} · {g["name"].replace("NVIDIA ","").replace(" Generation","")} · {round(g["total"]/1024)} GB',g['index'])
        if not gpus:self.addItem('No GPU detected',-1)
        index=self.findData(selected);self.setCurrentIndex(index if index>=0 else 0);self.blockSignals(False)

class GpuMonitor(QObject):
    updated=Signal(list)
    def __init__(self,parent=None):
        super().__init__(parent);self.process=QProcess(self)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(lambda _error:self.updated.emit([]))
        self.timer=QTimer(self);self.timer.setInterval(2500);self.timer.timeout.connect(self.refresh)
        self.timer.start();QTimer.singleShot(100,self.refresh)
    def refresh(self):
        if self.process.state()!=QProcess.ProcessState.NotRunning:return
        self.process.start('nvidia-smi',['--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
    def _finished(self,code,_status):
        gpus=[]
        if code==0:
            text=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
            for row in csv.reader(text.splitlines()):
                try:gpus.append(dict(index=int(row[0]),name=row[1].strip(),total=float(row[2]),used=float(row[3]),usage=float(row[4])))
                except (ValueError,IndexError):pass
        self.updated.emit(gpus)
