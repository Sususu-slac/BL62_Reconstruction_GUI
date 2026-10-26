"""Headless end-to-end GUI test on one real D-drive HDF5 slice."""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ct_recon_gui.hdf5_inspector import inspect_hdf5
from ct_recon_gui.main_window import MainWindow
from ct_recon_gui.models import ReconMethod


INPUT = Path(r"D:\Datasets\20260421_Bo\NMC811_Rod.hdf5")
OUTPUT = Path(r"D:\BL62_Reconstruction_GUI\test_outputs\gui_e2e_fbp_cuda_binning2x2")


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.input_edit.setText(str(INPUT))
    window.output_edit.setText(str(OUTPUT))
    window.info = inspect_hdf5(INPUT)
    window.center_spin.setValue(1028.5)
    window.start_spin.setValue(1012)
    window.end_spin.setValue(1016)
    window.binning_combo.setCurrentIndex(1)
    for index in range(window.method_combo.count()):
        if window.method_combo.itemData(index) == ReconMethod.FBP_CUDA:
            window.method_combo.setCurrentIndex(index)
            break

    result = {"finished": False, "ok": False, "message": "timeout"}

    def finished(ok: bool, message: str) -> None:
        result.update(finished=True, ok=ok, message=message)
        QTimer.singleShot(1500, app.quit)

    window.runner.finished.connect(finished)
    QTimer.singleShot(0, lambda: window._start_job("try"))
    QTimer.singleShot(60000, app.quit)
    app.exec()

    checks = {
        "job_finished": result["finished"],
        "job_ok": result["ok"],
        "result_count": window.result_combo.count(),
        "result_pixmap_visible": window.result_view.pixmap() is not None,
        "state": window.state_label.text(),
        "message": result["message"],
    }
    print(checks)
    return 0 if all((checks["job_finished"], checks["job_ok"], checks["result_count"] > 0,
                     checks["result_pixmap_visible"], checks["state"] == "Completed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
