from __future__ import annotations

import shlex
import sys
from pathlib import Path

from .models import GeometryMode, ReconJob, ReconMethod


from .runtime import APP_ROOT, WORKER_ROOT, reconstruction_python

PROJECT_ROOT = APP_ROOT
BACKEND_RUNNER = WORKER_ROOT / "backend_runner.py"


def build_wsl_argv(job: ReconJob) -> list[str]:
    """Build a native Windows command (kept name for API compatibility)."""
    job.validate()
    source = str(job.input_path)
    output = str(job.output_dir)
    if job.mode == GeometryMode.LAMINOGRAPHY:
        return [
            reconstruction_python(), str(BACKEND_RUNNER), job.method.value, source, output,
            "--reconstruction-type", job.reconstruction_type,
            "--search-kind", job.search_kind,
            "--start-row", str(job.start_row), "--end-row", str(job.end_row),
            "--center", str(job.center), "--gpu", str(job.gpu),
            "--nsino-per-chunk", str(job.chunk_size),
            "--nproj-per-chunk", str(job.nproj_chunk),
            "--center-search-width", str(job.center_search_width),
            "--center-search-step", str(job.center_search_step),
            "--lamino-angle", str(job.lamino_angle),
            "--lamino-search-width", str(job.lamino_search_width),
            "--lamino-search-step", str(job.lamino_search_step),
            "--binning", str(job.binning),
            "--remove-stripe-method", job.stripe_method,
        ] + (["--try-row", str(job.try_row)] if job.reconstruction_type == "try" else [])
    argv = [
        reconstruction_python(), str(BACKEND_RUNNER), job.method.value, source, output,
        "--reconstruction-type", job.reconstruction_type,
        "--start-row", str(job.start_row), "--end-row", str(job.end_row),
        "--center", str(job.center), "--gpu", str(job.gpu),
        "--chunk-size", str(job.chunk_size), "--dtype", job.dtype, "--binning", str(job.binning),
    ]
    if job.reconstruction_type == "try":
        argv += ["--try-row", str(job.try_row),
                 "--center-search-width", str(job.center_search_width),
                 "--center-search-step", str(job.center_search_step)]
    if job.method in (ReconMethod.SIRT_CUDA, ReconMethod.CGLS_CUDA):
        argv += ["--iterations", str(job.iterations)]
    if job.method == ReconMethod.LPREC and job.uniform_angle_approximation:
        argv += ["--accept-uniform-angle-approximation"]
    return argv


def display_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)
