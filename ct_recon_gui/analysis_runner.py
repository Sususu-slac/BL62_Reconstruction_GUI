from __future__ import annotations
import json
import time
from pathlib import Path
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal
from .runtime import WORKER_ROOT
from .process_utils import terminate_tree,clean_environment,external_runtime

class AnalysisRunner(QObject):
    log = Signal(str)
    updated = Signal(dict)
    finished = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process=QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.timer=QTimer(self); self.timer.setInterval(500); self.timer.timeout.connect(self.poll)
        self.output=None; self._signature=None; self._buffer=''; self.stopping=False

    @property
    def running(self): return self.process.state()!=QProcess.ProcessState.NotRunning

    def start(self, config):
        if self.running: raise RuntimeError('An analysis job is already running')
        self.output=Path(config['output']); self.output.mkdir(parents=True,exist_ok=False)
        self.stopping=False; self._signature=None; self._buffer=''
        path=self.output/'analysis_job.json'; path.write_text(json.dumps(config,indent=2),encoding='utf-8')
        env=QProcessEnvironment()
        for key,value in clean_environment().items():env.insert(key,value)
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(WORKER_ROOT))
        with external_runtime():
            self.process.start(config['sam3_python'],['-u',str(WORKER_ROOT/'analysis_worker.py'),'--config',str(path)])
            if not self.process.waitForStarted(5000): raise RuntimeError(self.process.errorString())
        self.timer.start()

    def stop(self):
        if not self.running or self.stopping: return
        self.stopping=True
        (self.output/'CANCEL').touch()
        self.log.emit('Safe stop requested; waiting for the current GPU operation.')
        pid=int(self.process.processId())
        QTimer.singleShot(15000,lambda:self._force_stop(pid))

    def _force_stop(self, pid):
        if self.running and int(self.process.processId())==pid and self.stopping:
            import threading
            threading.Thread(target=terminate_tree,args=(pid,),daemon=True).start()

    def poll(self):
        if not self.output:return
        path=self.output/'status.json'
        try:
            stat=path.stat(); signature=(stat.st_mtime_ns,stat.st_size)
            if signature==self._signature:return
            status=json.loads(path.read_text(encoding='utf-8')); self._signature=signature
            self.updated.emit(status)
        except (OSError,ValueError):pass

    def _read(self):
        text=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
        if text and self.output:
            with (self.output/'analysis.log').open('a',encoding='utf-8') as handle:handle.write(text)
        self._buffer+=text
        lines=self._buffer.split('\n');self._buffer=lines.pop()
        for line in lines:self.log.emit(line.rstrip())

    def _finished(self, code, _status):
        self.timer.stop();self._read();self.poll()
        if self._buffer:self.log.emit(self._buffer);self._buffer=''
        ok=code==0 and (self.output/'analysis_summary.json').is_file() and not self.stopping
        message='Stopped' if self.stopping or code==2 else 'Completed' if ok else 'Failed (see Log)'
        self.finished.emit(ok,message)
