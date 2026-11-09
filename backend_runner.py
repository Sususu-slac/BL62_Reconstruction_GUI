"""Native backend executed only by the isolated tomocupy_astra environment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from ct_recon_gui.preprocessing import add_arguments as preprocessing_arguments, argv as preprocessing_argv
from ct_recon_gui.geometry import output_row_limit


TOMOCUPY_METHODS = {"fourierrec", "lprec", "linerec"}
ASTRA_METHODS = {"fbp_cuda", "sirt_cuda", "cgls_cuda"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("method", choices=sorted(TOMOCUPY_METHODS | ASTRA_METHODS | {"lamino_fourierrec", "lamino_linerec"}))
    p.add_argument("input", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--reconstruction-type", choices=("try", "full"), default="try")
    p.add_argument("--search-kind", choices=("center", "angle"), default="center")
    p.add_argument("--start-row", type=int, required=True); p.add_argument("--end-row", type=int, required=True)
    p.add_argument("--try-row", type=int)
    p.add_argument("--center", type=float, required=True); p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--chunk-size", "--nsino-per-chunk", dest="chunk_size", type=int, default=4)
    p.add_argument("--nproj-per-chunk", type=int, default=8); p.add_argument("--dtype", default="float32")
    p.add_argument("--iterations", type=int, default=20);
    p.add_argument("--binning", type=int, choices=(0, 1, 2, 3), default=0)
    p.add_argument("--lamino-angle", type=float, default=20); p.add_argument("--center-search-width", type=float, default=50)
    p.add_argument("--center-search-step", type=float, default=.5)
    p.add_argument("--inclusive-center-search", action="store_true")
    p.add_argument("--lamino-search-width", type=float, default=5); p.add_argument("--lamino-search-step", type=float, default=.25)
    p.add_argument("--accept-uniform-angle-approximation", action="store_true")
    preprocessing_arguments(p)
    return p.parse_args()


def validate(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.method in {"linerec", "lamino_linerec"} and args.chunk_size < 4:
        raise ValueError("LineRec requires chunk size >= 4 because smaller chunks can silently produce zero images")
    with h5py.File(args.input, "r") as f:
        required = ("exchange/data", "exchange/data_white", "exchange/data_dark", "exchange/theta")
        missing = [key for key in required if key not in f]
        if missing: raise KeyError(f"Missing HDF5 datasets: {missing}")
        nproj, nrow, width = f["exchange/data"].shape
        limit = output_row_limit(nrow, "laminography", args.lamino_angle, args.binning) if args.method.startswith("lamino_") and args.reconstruction_type == "full" else nrow
        if not 0 <= args.start_row < args.end_row <= limit: raise ValueError("Invalid reconstruction-output row range")
        if args.reconstruction_type == "full" and args.end_row - args.start_row < 2 ** args.binning:
            raise ValueError("Row range is smaller than the selected binning factor")
        if args.reconstruction_type == "try" and (args.try_row is None or not 0 <= args.try_row < nrow):
            raise ValueError(f"Try detector row must be in [0,{nrow})")
        if len(f["exchange/theta"]) != nproj: raise ValueError("Theta/projection mismatch")
    return nproj, nrow, width


def tomocupy_executable() -> Path:
    candidate = Path(sys.prefix) / "Scripts" / "tomocupy.exe"
    if not candidate.exists():
        candidate = Path(sys.prefix) / "Scripts" / "tomocupy-script.py"
    if not candidate.exists(): raise FileNotFoundError("TomocuPy CLI not found in current isolated environment")
    return candidate


def equal_angle_view(source: Path, output: Path) -> Path:
    """Create a tiny view that links raw arrays and replaces only theta."""
    source = source.resolve()
    output = output.resolve()
    with h5py.File(source, "r") as src:
        nproj = int(src["exchange/data"].shape[0])
        names = tuple(src["exchange"].keys())
    with h5py.File(output, "w") as dst:
        exchange = dst.create_group("exchange")
        for name in names:
            if name != "theta":
                exchange[name] = h5py.ExternalLink(str(source), f"/exchange/{name}")
        exchange.create_dataset("theta", data=np.linspace(0, 180, nproj, endpoint=False, dtype=np.float32))
        exchange.attrs["theta_substitution"] = "uniform [0,180) approximation for LpRec"
        exchange.attrs["original_file"] = str(source)
    return output


def run_logged_command(command, child_env):
    stage_patterns = (("Reading data.", "reading"), ("Processing by chunks in z.", "preprocessing_z"),
                      ("Processing by chunks in angles.", "preprocessing_angles"),
                      ("Filtered backprojection and writing by chunks.", "reconstruction_write"))
    stages = {}
    clock = time.perf_counter()
    stage, since = "startup", clock
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace", env=child_env) as child:
        for line in child.stdout:
            print(line, end="", flush=True)
            for marker, following in stage_patterns:
                if marker in line:
                    now = time.perf_counter()
                    stages[stage] = stages.get(stage, 0.0) + now - since
                    stage, since = following, now
                    break
        code = child.wait()
    stages[stage] = stages.get(stage, 0.0) + time.perf_counter() - since
    if code:
        raise subprocess.CalledProcessError(code, command)
    return stages


def run_tomocupy(args: argparse.Namespace) -> tuple[list[str], dict]:
    is_lamino = args.method in {"lamino_fourierrec", "lamino_linerec"}
    method = {"lamino_fourierrec": "fourierrec", "lamino_linerec": "linerec"}.get(args.method, args.method)
    with h5py.File(args.input, "r") as handle:
        _, nrow, width = handle["exchange/data"].shape
    effective_reconstruction_type = "try_lamino" if is_lamino and args.reconstruction_type == "try" and args.search_kind == "angle" else args.reconstruction_type
    source = args.input
    approximation = False
    if method == "lprec":
        if not args.accept_uniform_angle_approximation:
            raise ValueError("LpRec requires explicit acceptance of the uniform [0,180) theta approximation")
        source = equal_angle_view(args.input, args.output / "equal_angle_view.h5")
        approximation = True
    command_start_row = 0 if args.reconstruction_type == "try" or is_lamino else args.start_row
    command_end_row = nrow if args.reconstruction_type == "try" or is_lamino else args.end_row
    command = [str(tomocupy_executable()), "recon_steps" if is_lamino else "recon",
        "--file-name", str(source), "--out-path-name", str(args.output),
        "--reconstruction-type", effective_reconstruction_type, "--reconstruction-algorithm", method,
        "--rotation-axis", str(args.center), "--start-row", str(command_start_row), "--end-row", str(command_end_row),
        "--dtype", args.dtype, "--nsino-per-chunk", str(args.chunk_size), "--nproj-per-chunk", str(args.nproj_per_chunk),
        "--binning", str(args.binning),
        "--save-format", "tiff", "--clear-folder", "False", "--verbose"]
    if args.reconstruction_type == "try" and (not is_lamino or args.search_kind == "center"):
        factor = 2 ** args.binning
        command += ["--nsino", str(args.try_row / max(1, nrow - 1)),
                    "--center-search-width", str(args.center_search_width),
                    "--center-search-step", str(args.center_search_step / factor)]
    if is_lamino:
        command += ["--lamino-angle", str(args.lamino_angle),
                    "--lamino-search-width", str(args.lamino_search_width),
                    "--lamino-search-step", str(args.lamino_search_step)]
        if args.reconstruction_type == "full":
            command += ["--lamino-start-row", str(args.start_row), "--lamino-end-row", str(args.end_row)]
        if args.reconstruction_type == "try" and args.search_kind == "angle":
            command += ["--nsino", str(args.try_row / max(1, nrow - 1))]
    command += preprocessing_argv(vars(args),external=True)
    child_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    if args.reconstruction_type == "try" and args.search_kind == "center" and getattr(args, "inclusive_center_search", False):
        command[:1] = [sys.executable, str(Path(__file__).with_name("tomocupy_manual_entry.py"))]
    stages = run_logged_command(command, child_env)
    factor = 2 ** args.binning
    count = (args.end_row - args.start_row) // factor
    fixed = sum(v for k, v in stages.items() if k != "reconstruction_write")
    timing = {"stages_seconds": stages, "fixed_seconds": fixed,
              "reconstruction_write_seconds": stages.get("reconstruction_write"),
              "work_units": int(np.ceil(count / args.chunk_size)) if is_lamino and method == "linerec" else count,
              "unit": "output_chunks" if is_lamino and method == "linerec" else "output_slices"}
    return command, {"uniform_angle_approximation": approximation,
                     "timing_components": timing,
                     "inclusive_center_search": bool(getattr(args, "inclusive_center_search", False)),
                     "projection_rows": [0, nrow] if is_lamino else [command_start_row, command_end_row],
                     "effective_input": str(source), "original_input": str(args.input),
                     "requested_reconstruction_type": args.reconstruction_type,
                     "search_kind": args.search_kind,
                     "effective_reconstruction_type": effective_reconstruction_type,
                     "try_detector_row": args.try_row if args.reconstruction_type == "try" else None,
                     "try_nsino": args.try_row / max(1, nrow - 1) if args.reconstruction_type == "try" else None,
                     "center_search_width_original_pixels": args.center_search_width if args.reconstruction_type == "try" and args.search_kind == "center" else None,
                     "center_search_step_original_pixels": args.center_search_step if args.reconstruction_type == "try" and args.search_kind == "center" else None,
                     "lamino_search_width_degrees": args.lamino_search_width if is_lamino and args.search_kind == "angle" else None,
                     "lamino_search_step_degrees": args.lamino_search_step if is_lamino and args.search_kind == "angle" else None,
                     "effective_center": args.center / factor,
                     "effective_rows": [args.start_row // factor,
                                        args.start_row // factor + (args.end_row-args.start_row) // factor],
                     "effective_width": width // factor}


def remove_astra_stripes(norm,args):
    """Apply the same transmission-domain Vo correction used by TomocuPy."""
    mode=getattr(args,'remove_stripe_method','vo-all')
    if mode=='none':return norm
    if mode not in ('vo-all','fw','ti'):raise ValueError('Unknown stripe method')
    import cupy as cp
    from tomocupy.processing.remove_stripe import remove_all_stripe
    with cp.cuda.Device(args.gpu):
        for start in range(0,norm.shape[1],args.chunk_size):
            stop=min(start+args.chunk_size,norm.shape[1])
            data=cp.asarray(norm[:,start:stop],dtype=cp.float32)
            if mode=='vo-all':corrected=remove_all_stripe(data,snr=3,la_size=61,sm_size=21,dim=1)
            elif mode=='fw':
                from tomocupy.processing.remove_stripe import remove_stripe_fw
                corrected=remove_stripe_fw(data,1,'sym16',7)
            else:
                from tomocupy.processing.remove_stripe import remove_stripe_ti
                corrected=remove_stripe_ti(data,.022,1)
            norm[:,start:stop]=cp.asnumpy(corrected)
    return norm


def preprocess_astra(raw,flat,dark,args):
    import cupy as cp
    from types import SimpleNamespace
    from tomocupy.processing import proc_functions as proc
    if getattr(args,'flat_linear',False) and flat.shape[0]<3:raise ValueError('Flat-linear needs at least three flat frames')
    old_args,old_params=proc.args,proc.params
    proc.args=SimpleNamespace(dtype='float32',dezinger=getattr(args,'dezinger',0),dezinger_threshold=getattr(args,'dezinger_threshold',5000),bright_ratio=getattr(args,'bright_ratio',1.),flat_linear=str(getattr(args,'flat_linear',False)))
    proc.params=SimpleNamespace(nproj=raw.shape[0]);functions=proc.ProcFunctions.__new__(proc.ProcFunctions)
    result=np.empty_like(raw,dtype=np.float32)
    try:
        with cp.cuda.Device(args.gpu):
            for start in range(0,raw.shape[1],args.chunk_size):
                stop=min(start+args.chunk_size,raw.shape[1])
                data=functions.remove_outliers(cp.asarray(raw[:,start:stop],dtype=cp.float32))
                white=functions.remove_outliers(cp.asarray(flat[:,start:stop],dtype=cp.float32))
                black=functions.remove_outliers(cp.asarray(dark[:,start:stop],dtype=cp.float32))
                transmission=cp.asnumpy(functions.darkflat_correction(data,black,white))
                transmission=remove_astra_stripes(transmission,args)
                result[:,start:stop]=cp.asnumpy(functions.minus_log(cp.asarray(transmission))) if getattr(args,'minus_log',True) else transmission
    finally:proc.args,proc.params=old_args,old_params
    return result


def run_astra(args: argparse.Namespace, width: int) -> tuple[list[str], dict]:
    import astra
    import tifffile
    if not astra.use_cuda(): raise RuntimeError("ASTRA CUDA backend is unavailable")
    with h5py.File(args.input, "r") as f:
        theta0 = f["exchange/theta"][:].astype(np.float32)
        theta = np.deg2rad(theta0) if np.max(np.abs(theta0)) > 2*np.pi else theta0
        factor = 2 ** args.binning
        effective_width = width // factor
        cropped_width = effective_width * factor
        effective_center = args.center / factor
        offset = effective_center - (effective_width - 1) / 2
        vec = np.zeros((len(theta), 6), np.float32)
        vec[:,0], vec[:,1] = np.sin(theta), -np.cos(theta); vec[:,4], vec[:,5] = np.cos(theta), np.sin(theta)
        vec[:,2], vec[:,3] = -offset*vec[:,4], -offset*vec[:,5]
        pg = astra.create_proj_geom("parallel_vec", effective_width, vec)
        vg = astra.create_vol_geom(effective_width, effective_width)
        usable_rows = ((args.end_row - args.start_row) // factor) * factor
        raw_end = args.start_row + usable_rows
        raw = f["exchange/data"][:, args.start_row:raw_end, :cropped_width].astype(np.float32)
        raw = raw.reshape(len(theta), usable_rows // factor, factor, effective_width, factor).mean(axis=(2, 4))
        def fields(name):
            dataset=f[name]
            a=dataset[:,args.start_row:raw_end,:cropped_width] if dataset.ndim==3 else dataset[args.start_row:raw_end,:cropped_width][None]
            return a.astype(np.float32).reshape(a.shape[0],usable_rows//factor,factor,effective_width,factor).mean(axis=(2,4))
        flat_b=fields('exchange/data_white');dark_b=fields('exchange/data_dark')
        batch=preprocess_astra(raw,flat_b,dark_b,args)
        effective_start = args.start_row // factor
        for start in range(0, batch.shape[1], args.chunk_size):
            end = min(start + args.chunk_size, batch.shape[1])
            for local in range(start, end):
                row = effective_start + local
                sid = astra.data2d.create("-sino", pg, np.ascontiguousarray(batch[:,local])); rid = astra.data2d.create("-vol", vg, 0)
                name = {"fbp_cuda":"FBP_CUDA", "sirt_cuda":"SIRT_CUDA", "cgls_cuda":"CGLS_CUDA"}[args.method]
                cfg = astra.astra_dict(name); cfg.update(ProjectionDataId=sid, ReconstructionDataId=rid)
                cfg["option"] = {"GPUindex": args.gpu}
                if args.method == "fbp_cuda":
                    if args.filter=="cosine2":raise ValueError("Cosine2 filter is only supported by TomocuPy")
                    cfg["option"]["FilterType"] = {"ramp":"ram-lak","shepp":"shepp-logan"}.get(args.filter,args.filter)
                aid = astra.algorithm.create(cfg)
                try:
                    astra.algorithm.run(aid, 1 if args.method == "fbp_cuda" else args.iterations)
                    rec = astra.data2d.get(rid).astype(np.float32)
                finally:
                    astra.algorithm.delete(aid); astra.data2d.delete([sid, rid])
                tifffile.imwrite(args.output / f"recon_{row:05d}.tiff", rec)
                print(f"RESULT {row} {args.output / f'recon_{row:05d}.tiff'}", flush=True)
    return [sys.executable, __file__, *sys.argv[1:]], {
        "uniform_angle_approximation": False,
        "binning_factor": 2 ** args.binning,
        "effective_center": args.center / (2 ** args.binning),
        "effective_rows": [args.start_row // (2 ** args.binning),
                           args.start_row // (2 ** args.binning) + (args.end_row-args.start_row) // (2 ** args.binning)],
        "effective_width": width // (2 ** args.binning),
    }


def main() -> int:
    args = parse_args(); _, _, width = validate(args); args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command, method_metadata = run_astra(args, width) if args.method in ASTRA_METHODS else run_tomocupy(args)
    metadata = {"environment": "tomocupy_astra", "preprocessing":__import__("ct_recon_gui.preprocessing",fromlist=["values"]).values(vars(args)), "remove_stripe_method":args.remove_stripe_method, "method": args.method, "input": str(args.input),
                "output": str(args.output), "command": command, "elapsed_seconds": time.perf_counter()-started,
                "binning_exponent": args.binning, "binning_factor": 2 ** args.binning,
                "original_center": args.center, "original_rows": [args.start_row, args.end_row],
                **method_metadata}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
