"""Headless GUI-to-background-worker export integration test."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ct_recon_gui.hdf5_inspector import inspect_hdf5
from ct_recon_gui.main_window import MainWindow


INPUT = Path(r"D:\Datasets\20260421_Bo\NMC811_Rod.hdf5")
OUTPUT = Path(r"D:\BL62_Reconstruction_GUI\test_outputs") / f"gui_export_{datetime.now():%Y%m%d_%H%M%S}"


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.info = inspect_hdf5(INPUT)
    window.input_edit.setText(str(INPUT))
    window._configure_input_navigation(450)
    window.export_source_combo.setCurrentIndex(window.export_source_combo.findData("projection"))
    window.export_scope_combo.setCurrentIndex(window.export_scope_combo.findData("current"))
    window.export_format_combo.setCurrentIndex(window.export_format_combo.findData("png"))
    window.export_mode_combo.setCurrentIndex(window.export_mode_combo.findData("display8"))
    window.export_output_edit.setText(str(OUTPUT))
    result = {"finished": False, "ok": False, "message": "timeout"}
    poll_timer = QTimer()

    def start() -> None:
        window._start_export()
        poll_timer.start(100)

    def poll() -> None:
        output = OUTPUT / "projection_00450.png"
        metadata = OUTPUT / "export_metadata.json"
        if output.exists() and metadata.exists() and window.export_thread is None:
            result.update(finished=True, ok=True, message="GUI export completed")
            poll_timer.stop(); app.quit()

    poll_timer.timeout.connect(poll)

    QTimer.singleShot(0, start)
    QTimer.singleShot(30000, app.quit)
    app.exec()
    output = OUTPUT / "projection_00450.png"
    checks = {**result, "output_exists": output.exists(), "metadata_exists": (OUTPUT / "export_metadata.json").exists(),
              "export_button_enabled": window.export_btn.isEnabled(), "output": str(output)}
    print(checks)
    return 0 if all((checks["finished"], checks["ok"], checks["output_exists"], checks["metadata_exists"],
                     checks["export_button_enabled"])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
