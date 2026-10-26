from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class GeometryMode(str, Enum):
    TOMOGRAPHY = "tomography"
    LAMINOGRAPHY = "laminography"


class ReconMethod(str, Enum):
    FOURIERREC = "fourierrec"
    LPREC = "lprec"
    LINEREC = "linerec"
    FBP_CUDA = "fbp_cuda"
    SIRT_CUDA = "sirt_cuda"
    CGLS_CUDA = "cgls_cuda"
    LAMINO_FOURIERREC = "lamino_fourierrec"
    LAMINO_LINEREC = "lamino_linerec"


LAMINO_METHODS = (ReconMethod.LAMINO_FOURIERREC, ReconMethod.LAMINO_LINEREC)
TOMO_METHODS = tuple(m for m in ReconMethod if m not in LAMINO_METHODS)


@dataclass(slots=True)
class ReconJob:
    input_path: Path
    output_dir: Path
    mode: GeometryMode = GeometryMode.TOMOGRAPHY
    method: ReconMethod = ReconMethod.FOURIERREC
    reconstruction_type: str = "try"
    search_kind: str = "center"
    start_row: int = 0
    end_row: int = 5
    try_row: int = 0
    center: float = 0.0
    binning: int = 0
    gpu: int = 0
    chunk_size: int = 4
    nproj_chunk: int = 8
    iterations: int = 20
    dtype: str = "float32"
    filter_name: str = "parzen"
    stripe_method: str = "none"
    lamino_angle: float = 20.0
    center_search_width: float = 50.0
    center_search_step: float = 0.5
    lamino_search_width: float = 5.0
    lamino_search_step: float = 0.25
    uniform_angle_approximation: bool = False

    def validate(self) -> None:
        if not self.input_path:
            raise ValueError("Select an input HDF5 file")
        if not self.output_dir:
            raise ValueError("Select an output directory")
        if self.end_row <= self.start_row:
            raise ValueError("End row must be greater than start row")
        if self.center <= 0:
            raise ValueError("Rotation center must be greater than zero")
        if self.binning not in (0, 1, 2, 3):
            raise ValueError("Binning must be 1×1, 2×2, 4×4 or 8×8")
        if self.reconstruction_type == "full" and self.end_row - self.start_row < 2 ** self.binning:
            raise ValueError(f"Selected row range must contain at least {2 ** self.binning} original rows for this binning")
        if self.try_row < 0:
            raise ValueError("Try detector row must be zero or greater")
        if self.center_search_width <= 0 or self.center_search_step <= 0:
            raise ValueError("Center search range and step must be greater than zero")
        if self.lamino_search_width <= 0 or self.lamino_search_step <= 0:
            raise ValueError("Lamino search range and step must be greater than zero")
        if self.search_kind not in ("center", "angle"):
            raise ValueError("Search kind must be center or angle")
        if self.mode == GeometryMode.LAMINOGRAPHY and self.method not in LAMINO_METHODS:
            raise ValueError("Laminography supports TomocuPy FourierRec or LineRec")
        if self.mode == GeometryMode.TOMOGRAPHY and self.method not in TOMO_METHODS:
            raise ValueError("Select a tomography method")
        if self.reconstruction_type == "try" and self.mode == GeometryMode.LAMINOGRAPHY and self.method != ReconMethod.LAMINO_LINEREC:
            raise ValueError("Laminography center/angle searches use LineRec")
        if self.method in (ReconMethod.LINEREC, ReconMethod.LAMINO_LINEREC) and self.chunk_size < 4:
            raise ValueError("LineRec requires chunk size 4 or greater; smaller chunks can produce a zero image")

    def to_json_dict(self) -> dict:
        data = asdict(self)
        data["input_path"] = str(self.input_path)
        data["output_dir"] = str(self.output_dir)
        data["mode"] = self.mode.value
        data["method"] = self.method.value
        return data
