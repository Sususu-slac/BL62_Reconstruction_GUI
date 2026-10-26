"""Validate test reconstructions and print compact numeric diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-size", type=int, default=2048)
    args = parser.parse_args()
    records = []
    for path in sorted(args.root.rglob("*.tif*")):
        image = tifffile.imread(path)
        record = {
            "path": str(path),
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "finite": bool(np.isfinite(image).all()),
            "min": float(np.nanmin(image)),
            "max": float(np.nanmax(image)),
            "mean": float(np.nanmean(image)),
            "std": float(np.nanstd(image)),
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))
    if not records:
        raise SystemExit("No TIFF reconstruction found")
    if not all(
        r["finite"] and r["dtype"] == "float32"
        and r["shape"] == [args.expected_size, args.expected_size] and r["std"] > 0
        for r in records
    ):
        raise SystemExit("One or more reconstructions failed validation")
    print(f"VALIDATED {len(records)} TIFF files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
