"""Native Windows TomocuPy adapter for explicit centers on one detector row.

Uses the installed reconstruction/preprocessing implementation unchanged; only
the candidate shifts and saved center coordinates are supplied explicitly.
"""
import argparse
import json
import os
import sys
from pathlib import Path

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--centers-file', required=True)
    p.add_argument('--row', type=int, required=True)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--method', choices=['fourierrec', 'linerec'], default='fourierrec')
    a = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['PYTHONNOUSERSITE'] = '1'
    import h5py
    import numpy as np
    with h5py.File(a.input, 'r') as f:
        _, height, width = f['exchange/data'].shape
    centers = np.array(json.loads(Path(a.centers_file).read_text()), dtype=np.float32)
    if not len(centers) or not np.all(np.isfinite(centers)):
        raise ValueError('No finite candidate centers')
    if len(set(centers.tolist())) != len(centers) or centers.min() < 0 or centers.max() > width-1:
        raise ValueError('Centers must be unique and within detector width')
    if not 0 <= a.row < height:
        raise ValueError('Invalid detector row')
    from tomocupy.dataio.reader import Reader
    from tomocupy.global_vars import params
    from tomocupy.__main__ import main as tomo_main
    original = Reader.init_sizes_try
    def exact_centers(self):
        original(self)
        params.id_slices = np.array([a.row], dtype=int)
        params.save_centers = centers.copy()
        params.shift_array = (params.centeri - centers).astype(np.float32)
        params.nschunk = int(np.ceil(len(centers)/params.ncz))
        params.lschunk = np.minimum(params.ncz, np.int32(len(centers)-np.arange(params.nschunk)*params.ncz))
    Reader.init_sizes_try = exact_centers
    sys.argv = ['tomocupy', 'recon', '--file-name', a.input, '--out-path-name', a.output,
                '--reconstruction-type', 'try', '--reconstruction-algorithm', a.method,
                '--rotation-axis', str(width/2), '--start-row', '0', '--end-row', str(height),
                '--nsino', str(a.row/(height-1)), '--dtype', 'float32', '--binning', '0',
                '--nsino-per-chunk', '4', '--nproj-per-chunk', '8', '--fbp-filter', 'parzen',
                '--remove-stripe-method', 'none', '--center-search-width', '1',
                '--center-search-step', '.25', '--save-format', 'tiff', '--clear-folder', 'False']
    tomo_main()
    files = sorted(Path(a.output).rglob('*.tiff'))
    if len(files) != len(centers):
        raise RuntimeError(f'Expected {len(centers)} outputs, found {len(files)}')
    print(json.dumps({'row':a.row, 'centers':centers.tolist(), 'files':len(files)}), flush=True)

if __name__ == '__main__':
    main()
