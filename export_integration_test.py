"""Real-data integration test for every export source and output mode."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from ct_recon_gui.exporter import ExportJob, ExportWorker


HDF5 = Path(r"D:\Datasets\20260421_Bo\NMC811_Rod.hdf5")
RECON = Path(r"D:\BL62_Reconstruction_GUI\test_outputs\NMC811_Rod_fbp_cuda\recon_01013.tiff")
ROOT = Path(r"D:\BL62_Reconstruction_GUI\test_outputs") / f"export_integration_{datetime.now():%Y%m%d_%H%M%S}"


def run(job: ExportJob) -> None:
    result = []
    worker = ExportWorker(job)
    worker.finished.connect(lambda ok, message: result.append((ok, message)))
    worker.run()
    if not result or not result[-1][0]:
        raise RuntimeError(result[-1][1] if result else "Worker emitted no completion status")


def main() -> int:
    jobs = [
        ExportJob("projection", HDF5, ROOT / "projection_raw_tiff", [450], "tiff", "preserve"),
        ExportJob("flat", HDF5, ROOT / "flat_png8", [0], "png", "display8"),
        ExportJob("dark", HDF5, ROOT / "dark_raw_tiff", [0], "tiff", "preserve"),
        ExportJob("corrected", HDF5, ROOT / "corrected_float_tiff", [450], "tiff", "float32"),
        ExportJob("attenuation", HDF5, ROOT / "attenuation_png16", [450], "png", "display16"),
        ExportJob("reconstruction", RECON.parent, ROOT / "recon_png8", [0], "png", "display8", [RECON]),
    ]
    for job in jobs:
        run(job)

    checks = {
        "projection_dtype": str(tifffile.imread(ROOT / "projection_raw_tiff" / "projection_00450.tiff").dtype),
        "dark_dtype": str(tifffile.imread(ROOT / "dark_raw_tiff" / "dark_00000.tiff").dtype),
        "corrected_dtype": str(tifffile.imread(ROOT / "corrected_float_tiff" / "corrected_00450.tiff").dtype),
        "flat_png_mode": Image.open(ROOT / "flat_png8" / "flat_00000.png").mode,
        "attenuation_png_shape": list(np.asarray(Image.open(ROOT / "attenuation_png16" / "attenuation_00450.png")).shape),
        "recon_png_shape": list(np.asarray(Image.open(ROOT / "recon_png8" / "recon_01013.png")).shape),
        "metadata_files": len(list(ROOT.rglob("export_metadata.json"))),
        "output_root": str(ROOT),
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    passed = (checks["projection_dtype"] == "uint16" and checks["dark_dtype"] == "uint16"
              and checks["corrected_dtype"] == "float32" and checks["flat_png_mode"] == "L"
              and checks["attenuation_png_shape"] == [2048, 2048]
              and checks["recon_png_shape"] == [2048, 2048] and checks["metadata_files"] == 6)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
