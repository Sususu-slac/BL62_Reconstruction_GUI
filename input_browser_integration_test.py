"""Exercise every input-browser source on a real D-drive HDF5 file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from ct_recon_gui.hdf5_inspector import inspect_hdf5
from ct_recon_gui.main_window import MainWindow


INPUT = Path(r"D:\Datasets\20260421_Bo\NMC811_Rod.hdf5")


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.input_edit.setText(str(INPUT))
    window.info = inspect_hdf5(INPUT)
    checks = []
    for combo_index in range(window.input_source_combo.count()):
        window.input_source_combo.setCurrentIndex(combo_index)
        window._configure_input_navigation(0)
        window.frame_slider.setValue(window.frame_slider.maximum() // 2)
        window._load_input_frame()
        text = window.input_stats_label.text()
        checks.append({
            "source": window.input_source_combo.currentData(),
            "count": window.frame_slider.maximum() + 1,
            "pixmap": window.input_view.pixmap() is not None,
            "stats": text,
            "preview_error": text.startswith("Preview error:"),
        })
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    window.close()
    app.processEvents()
    return 0 if all(c["pixmap"] and not c["preview_error"] and c["count"] > 0 for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
