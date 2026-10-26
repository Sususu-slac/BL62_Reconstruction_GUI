from __future__ import annotations

import json
import subprocess
import zlib
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from .commands import build_wsl_argv, display_command
from .models import ReconJob
from .process_utils import terminate_tree,clean_environment,external_runtime


def file_signature(path: Path) -> tuple[int, int, int]:
    """Detect fast same-size TIFF rewrites even on coarse timestamp filesystems."""
    stat = path.stat()
    size = stat.st_size
    checksum = 0
    with path.open("rb") as handle:
        offsets = (0, max(0, size // 2 - 2048), max(0, size - 4096))
        for offset in dict.fromkeys(offsets):
            handle.seek(offset)
            checksum = zlib.crc32(handle.read(4096), checksum)
    return stat.st_mtime_ns, size, checksum


class JobRunner(QObject):
    log = Signal(str)
    state_changed = Signal(str)
    result_found = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(lambda error: self.log.emit(f"Process error: {error}"))
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._scan_results)
        self.stopping = False
        self.output_dir: Path | None = None
        self.signatures: dict[Path, tuple[int, int, int]] = {}
        self.pending_signatures: dict[Path, tuple[tuple[int, int, int], int]] = {}

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, job: ReconJob) -> None:
        if self.running:
            raise RuntimeError("A reconstruction job is already running")
        self.stopping = False
        argv = build_wsl_argv(job)
        self.output_dir = job.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.signatures = {
            path: file_signature(path)
            for path in (*self.output_dir.rglob("*.tif"), *self.output_dir.rglob("*.tiff"))
        }
        self.pending_signatures.clear()
        metadata = job.to_json_dict()
        metadata.update(created_at=datetime.now().isoformat(), command=argv)
        (self.output_dir / "gui_job.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        self.log.emit("$ " + display_command(argv))
        (self.output_dir/'gui_run.log').write_text('$ '+display_command(argv)+'\n',encoding='utf-8')
        environment=QProcessEnvironment()
        for key,value in clean_environment().items():environment.insert(key,value)
        self.process.setProcessEnvironment(environment)
        with external_runtime():
            self.process.start(argv[0], argv[1:])
            if not self.process.waitForStarted(5000):raise RuntimeError(self.process.errorString())
        self.state_changed.emit("Running")
        self.timer.start()

    def stop(self) -> None:
        if not self.running or self.stopping:return
        self.stopping=True;self.state_changed.emit('Stopping');self.log.emit('Safely stopping this job and its child processes…')
        pid=int(self.process.processId())
        import threading
        threading.Thread(target=terminate_tree,args=(pid,),daemon=True).start()

    def _read_output(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        if self.output_dir and text:
            with (self.output_dir/'gui_run.log').open('a',encoding='utf-8') as handle:handle.write(text)
        for line in text.splitlines():
            self.log.emit(line)

    def _scan_results(self, *, force: bool = False) -> None:
        if not self.output_dir or not self.output_dir.exists():
            return
        candidates = sorted((*self.output_dir.rglob("*.tif"), *self.output_dir.rglob("*.tiff")))
        for path in candidates:
            try:
                signature = file_signature(path)
            except OSError:
                self.pending_signatures.pop(path, None)
                continue
            if signature[1] == 0:
                continue
            if self.signatures.get(path) == signature:
                self.pending_signatures.pop(path, None)
                continue
            if not force:
                pending = self.pending_signatures.get(path)
                stable_count = pending[1] + 1 if pending and pending[0] == signature else 1
                if stable_count < 2:
                    self.pending_signatures[path] = (signature, stable_count)
                    continue
            self.pending_signatures.pop(path, None)
            self.signatures[path] = signature
            self.result_found.emit(str(path))

    def _finished(self, exit_code: int, _status) -> None:
        self.timer.stop()
        self._read_output()
        self._scan_results(force=True)
        ok = exit_code == 0 and not self.stopping
        state = "Stopped" if self.stopping else "Completed" if ok else "Failed"
        self.state_changed.emit(state)
        self.finished.emit(ok, f"{state} (exit code {exit_code})")
