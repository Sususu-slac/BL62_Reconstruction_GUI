"""Use TomoCuPy unchanged except for the explicitly requested manual center grid.

The stock center trial uses an end-exclusive shift array. This process-local
adapter includes an aligned endpoint, preserving the supplied center anchor and
all filtering/backprojection operations. Installed TomoCuPy files are untouched.
"""
from __future__ import annotations

import math
import numpy as np


def inclusive_shifts(half_width, original_step, factor):
    half_width, original_step = float(half_width), float(original_step)
    if half_width <= 0 or original_step <= 0:
        raise ValueError("Center range and step must be positive")
    count = int(math.floor((2 * half_width) / original_step + 1e-7)) + 1
    values = -half_width + np.arange(count, dtype=np.float64) * original_step
    return (values / factor).astype(np.float32)


def main():
    from tomocupy.dataio.reader import Reader
    from tomocupy.global_vars import args, params
    original = Reader.init_sizes_try

    def manual_grid(self):
        original(self)
        if args.reconstruction_type != "try":
            return
        factor = 2 ** args.binning
        shifts = inclusive_shifts(args.center_search_width, args.center_search_step * factor, factor)
        saved = (params.centeri - shifts) * factor + params.st_n
        if args.file_type == "double_fov" and params.centeri < params.ni // 2:
            shifts = -shifts
        chunks = int(math.ceil(len(shifts) / params.ncz))
        params.shift_array = shifts
        params.save_centers = saved
        params.nschunk = chunks
        params.lschunk = np.minimum(params.ncz, np.int32(len(shifts) - np.arange(chunks) * params.ncz))

    Reader.init_sizes_try = manual_grid
    from tomocupy.__main__ import main as run
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
