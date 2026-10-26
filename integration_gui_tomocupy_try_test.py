"""Verify GUI Recon Try reconstructs one row at multiple rotation centers."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ct_recon_gui.hdf5_inspector import inspect_hdf5
from ct_recon_gui.main_window import MainWindow
from ct_recon_gui.models import ReconMethod


INPUT = Path(r"D:\Datasets\hdf5_collection\NMC622_Tri_7.hdf5")
OUTPUT = Path(r"D:\BL62_Reconstruction_GUI\test_outputs") / f"NMC622_Tri_7_center_search_{datetime.now():%Y%m%d_%H%M%S}"


def main() -> int:
    app = QApplication(sys.argv); window = MainWindow()
    window.input_edit.setText(str(INPUT)); window.output_edit.setText(str(OUTPUT)); window.info = inspect_hdf5(INPUT)
    window.center_spin.setValue(1023.5); window.start_spin.setValue(1022); window.end_spin.setValue(1027)
    window.try_row_spin.setValue(1024); window.search_width_spin.setValue(1.0); window.search_step_spin.setValue(0.5)
    window.binning_combo.setCurrentIndex(2)
    window.method_combo.setCurrentIndex(window.method_combo.findData(ReconMethod.FOURIERREC))
    result = {"finished": False, "ok": False, "message": "timeout"}

    def finished(ok: bool, message: str) -> None:
        result.update(finished=True, ok=ok, message=message); QTimer.singleShot(500, app.quit)

    window.runner.finished.connect(finished)
    QTimer.singleShot(0, lambda: window._start_job("try")); QTimer.singleShot(90000, app.quit)
    app.exec()
    tiffs = sorted(OUTPUT.rglob("*.tif*")); center_sweeps = [p for p in tiffs if "center" in p.name]
    image = tifffile.imread(tiffs[0]) if tiffs else np.empty((0, 0))
    if window.center_candidates:
        window.center_candidate_slider.setValue(len(window.center_candidates) // 2); window._apply_selected_center()
    checks = {**result, "tiff_count": len(tiffs), "center_sweep_count": len(center_sweeps),
              "gui_candidate_count": len(window.center_candidates), "normal_result_count": len(window.results),
              "shape": list(image.shape), "finite": bool(np.isfinite(image).all()),
              "std": float(np.std(image)) if image.size else 0.0,
              "selection_saved": (OUTPUT / "selected_rotation_center.json").exists(), "output": str(OUTPUT)}
    print(checks)
    passed = (checks["finished"] and checks["ok"] and checks["tiff_count"] == 4
              and checks["center_sweep_count"] == 4 and checks["gui_candidate_count"] == 4
              and checks["normal_result_count"] == 0 and checks["shape"] == [512, 512]
              and checks["finite"] and checks["std"] > 0 and checks["selection_saved"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
