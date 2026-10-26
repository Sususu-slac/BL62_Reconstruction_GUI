"""Verify the isolated Windows GPU reconstruction environment."""
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> int:
    expected_name = "tomocupy_astra"
    prefix = Path(sys.prefix)
    if prefix.name.lower() != expected_name:
        raise RuntimeError(f"Wrong environment: {prefix}. Expected {expected_name}")

    import astra
    import cupy as cp
    import h5py
    import numpy as np
    import PySide6
    import tifffile
    import tomocupy

    count = cp.cuda.runtime.getDeviceCount()
    if count < 1:
        raise RuntimeError("CuPy cannot find an NVIDIA GPU")
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode(errors="replace")
    value = float(cp.sum(cp.arange(1024, dtype=cp.float32)).get())
    if value != 523776.0:
        raise RuntimeError("CuPy GPU calculation returned an unexpected result")
    if not astra.use_cuda():
        raise RuntimeError("ASTRA CUDA backend is unavailable")

    cli = prefix / "Scripts" / "tomocupy.exe"
    if not cli.exists():
        raise FileNotFoundError(f"TomocuPy CLI is missing: {cli}")

    print("BL62 Reconstruction environment verification: PASS")
    print(f"Environment : {prefix}")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"GPU         : {gpu_name} (count={count})")
    print(f"CuPy        : {cp.__version__}")
    print(f"ASTRA       : {astra.__version__}, CUDA={astra.use_cuda()}")
    print(f"TomocuPy    : package {version('tomocupy')}, module {getattr(tomocupy, '__version__', 'unknown')}")
    print(f"NumPy       : {np.__version__}")
    print(f"h5py        : {h5py.__version__}")
    print(f"tifffile    : {tifffile.__version__}")
    print(f"PySide6     : {PySide6.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
