from __future__ import annotations

import json
import subprocess
import struct
import zlib
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QThread, QTimer, Signal

from .commands import build_wsl_argv, display_command
from .models import ReconJob
from .process_utils import terminate_tree,clean_environment,external_runtime


def tiff_ready(path: Path, size: int) -> bool:
    """Reject headers and truncated pixel blocks while a writer is still saving."""
    import tifffile
    try:
        with path.open('rb') as handle:
            header = handle.read(16)
        if len(header) < 8 or header[:2] not in (b'II', b'MM'):
            return False
        endian = '<' if header[:2] == b'II' else '>'
        version = struct.unpack(endian + 'H', header[2:4])[0]
        if version == 42:
            offset = struct.unpack(endian + 'I', header[4:8])[0]
        elif version == 43 and len(header) == 16:
            offset = struct.unpack(endian + 'Q', header[8:16])[0]
        else:
            return False
        if offset <= 0 or offset + 2 > size:
            return False
        with tifffile.TiffFile(path) as image:
            if not image.pages:
                return False
            page = image.pages[0]
            return bool(page.dataoffsets) and all(
                start > 0 and count > 0 and start + count <= size
                for start, count in zip(page.dataoffsets, page.databytecounts)
            )
    except (OSError, ValueError, IndexError, struct.error):
        return False


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


class ResultScan(QThread):
    """Perform directory enumeration and TIFF signatures outside the GUI thread."""
    def __init__(self, runner, force):
        super().__init__(runner)
        self.output = runner.output_dir
        self.signatures = dict(runner.signatures)
        self.pending = dict(runner.pending_signatures)
        self.force = force
        self.paths = []
        self.error = None

    def run(self):
        try:
            for path in sorted((*self.output.rglob('*.tif'), *self.output.rglob('*.tiff'))):
                try:
                    signature = file_signature(path)
                except OSError:
                    self.pending.pop(path, None)
                    continue
                if not signature[1]:
                    continue
                if self.signatures.get(path) == signature:
                    self.pending.pop(path, None)
                    continue
                old = self.pending.get(path)
                stable = old[1] + 1 if old and old[0] == signature else 1
                if not self.force and stable < 2:
                    self.pending[path] = (signature, stable)
                    continue
                if not tiff_ready(path, signature[1]):
                    self.pending[path] = (signature, stable)
                    if self.force:
                        self.error = f'Incomplete TIFF was not added to the viewer: {path}'
                    continue
                self.pending.pop(path, None)
                self.signatures[path] = signature
                self.paths.append(str(path))
        except OSError as error:
            self.error = str(error)


class JobRunner(QObject):
    log = Signal(str)
    state_changed = Signal(str)
    result_found = Signal(str)
    results_found = Signal(list)
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
        self.timer.timeout.connect(self._request_scan)
        self.stopping = False
        self.output_dir: Path | None = None
        self.signatures: dict[Path, tuple[int, int, int]] = {}
        self.pending_signatures: dict[Path, tuple[tuple[int, int, int], int]] = {}
        self._scan_task = None
        self._completion = None

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning or self._completion is not None or self._scan_task is not None

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
        if self.process.state() == QProcess.ProcessState.NotRunning:return
        pid=int(self.process.processId())
        import threading
        threading.Thread(target=terminate_tree,args=(pid,),daemon=True).start()

    def _read_output(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        if self.output_dir and text:
            with (self.output_dir/'gui_run.log').open('a',encoding='utf-8') as handle:handle.write(text)
        if text:self.log.emit(text.replace("\r", "\n").rstrip("\n"))

    def _request_scan(self, *, force=False):
        if self._scan_task is not None or self.output_dir is None:return
        task = ResultScan(self, force)
        self._scan_task = task
        task.finished.connect(self._scan_completed)
        task.start()

    def _accept_scan(self, task):
        self.signatures = task.signatures
        self.pending_signatures = task.pending
        if task.error:self.log.emit('Result scan warning: ' + task.error)
        if task.paths:
            self.results_found.emit(task.paths)
            for path in task.paths:self.result_found.emit(path)

    def _scan_completed(self):
        task = self._scan_task
        self._scan_task = None
        self._accept_scan(task)
        final = task.force
        task.deleteLater()
        if self._completion is not None:
            if final:self._complete()
            else:self._request_scan(force=True)

    def _scan_results(self, *, force=False):
        # Synchronous helper for callers without an event loop; GUI uses _request_scan.
        if self.output_dir is None:return
        task = ResultScan(self, force)
        task.run()
        self._accept_scan(task)
        task.deleteLater()

    def _finished(self, exit_code: int, _status) -> None:
        self.timer.stop()
        self._read_output()
        self._completion = exit_code
        self.state_changed.emit('Finalizing results')
        self._request_scan(force=True)

    def _complete(self):
        exit_code = self._completion
        self._completion = None
        ok = exit_code == 0 and not self.stopping
        state = "Stopped" if self.stopping else "Completed" if ok else "Failed"
        self.state_changed.emit(state)
        self.finished.emit(ok, f"{state} (exit code {exit_code})")
