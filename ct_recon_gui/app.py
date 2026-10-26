from __future__ import annotations

import sys
import json
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("BL62 Reconstruction")
    from .runtime import VERSION
    app.setApplicationVersion(VERSION)
    window = MainWindow()
    window.show()
    if '--self-test' in sys.argv:
        from .runtime import APP_ROOT,WORKER_ROOT,reconstruction_python
        target=Path(sys.argv[sys.argv.index('--self-test')+1]);app.processEvents()
        report={'version':VERSION,'tabs':window.main_tabs.count(),'manual_methods':window.method_combo.count(),
            'log_collapsed':not window.log_toggle.isChecked(),'segment_all':window.segment_all.isChecked(),
            'reconstruction_python':reconstruction_python(),
            'backend_present':(WORKER_ROOT/'backend_runner.py').is_file(),
            'analysis_worker_present':(WORKER_ROOT/'analysis_worker.py').is_file(),
            'bundled_sam3':(APP_ROOT/'sam3_repo/sam3/model_builder.py').is_file()}
        target.write_text(json.dumps(report,indent=2));window.close();return 0
    if '--validation-config' in sys.argv:
        from .release_validation import validate_release
        report=validate_release(app,window,sys.argv[sys.argv.index('--validation-config')+1])
        app.exec();return 0 if report['manual'] and report['automatic'] else 1
    return app.exec()
