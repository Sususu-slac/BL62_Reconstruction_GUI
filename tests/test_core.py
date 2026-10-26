from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import h5py
import numpy as np
import tifffile

from ct_recon_gui.commands import build_wsl_argv
from ct_recon_gui.exporter import ExportJob, ExportWorker, display_convert
from ct_recon_gui.hdf5_inspector import frame_count, inspect_hdf5, read_input_frame
from ct_recon_gui.image_view import normalize_uint8
from ct_recon_gui.models import GeometryMode, ReconJob, ReconMethod
from ct_recon_gui.paths import windows_to_wsl, wsl_to_windows
from backend_runner import equal_angle_view, run_tomocupy


class CoreTests(unittest.TestCase):
    def _tomocupy_args(self, root: Path, **overrides):
        source = root / "source.h5"; output = root / "out"
        with h5py.File(source, "w") as handle:
            ex = handle.create_group("exchange")
            ex.create_dataset("data", data=np.ones((5, 8, 10), np.uint16))
            ex.create_dataset("data_white", data=np.ones((1, 8, 10), np.uint16))
            ex.create_dataset("data_dark", data=np.zeros((1, 8, 10), np.uint16))
            ex.create_dataset("theta", data=np.linspace(0, 180, 5, endpoint=False))
        output.mkdir()
        values = dict(method="fourierrec", reconstruction_type="try", search_kind="center",
                      input=source, output=output, center=4.5, start_row=2, end_row=4,
                      dtype="float32", chunk_size=4, nproj_per_chunk=8, binning=0,
                      filter="parzen", remove_stripe_method="none", gpu=0,
                      lamino_angle=20, center_search_width=50, center_search_step=.5,
                      lamino_search_width=5, lamino_search_step=.25,
                      accept_uniform_angle_approximation=False, try_row=4)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_path_roundtrip(self):
        self.assertEqual(windows_to_wsl(r"D:\Data\a.h5"), "/mnt/d/Data/a.h5")
        self.assertEqual(wsl_to_windows("/mnt/d/Data/a.h5"), r"D:\Data\a.h5")

    def test_hdf5_inspection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.h5"
            with h5py.File(path, "w") as handle:
                ex = handle.create_group("exchange")
                ex.create_dataset("data", data=np.ones((5, 8, 10), np.uint16))
                ex.create_dataset("data_white", data=np.ones((2, 8, 10), np.uint16))
                ex.create_dataset("data_dark", data=np.zeros((2, 8, 10), np.uint16))
                ex.create_dataset("theta", data=np.linspace(0, 180, 5, endpoint=False))
            info = inspect_hdf5(path)
            self.assertEqual(info.data_shape, (5, 8, 10))
            self.assertEqual(info.suggested_center, 4.5)
            self.assertTrue(info.theta_uniform)
            self.assertEqual(frame_count(info, "projection"), 5)
            projection, angle = read_input_frame(path, "projection", 2)
            corrected, _ = read_input_frame(path, "corrected", 2)
            attenuation, _ = read_input_frame(path, "attenuation", 2)
            self.assertEqual(projection.shape, (8, 10))
            self.assertEqual(angle, 72.0)
            np.testing.assert_allclose(corrected, 1.0)
            np.testing.assert_allclose(attenuation, 0.0)

    def test_native_command_is_current_python(self):
        job = ReconJob(Path(r"D:\Data\a.h5"), Path(r"D:\Out"), center=1024.5,
                       start_row=10, end_row=15, method=ReconMethod.FBP_CUDA)
        argv = build_wsl_argv(job)
        self.assertTrue(argv[0].lower().endswith("python.exe") or argv[0].endswith("python"))
        self.assertIn("backend_runner.py", argv[1])
        self.assertNotIn("wsl.exe", argv)

    def test_laminography_does_not_offer_astra(self):
        job = ReconJob(Path("a.h5"), Path("out"), mode=GeometryMode.LAMINOGRAPHY,
                       method=ReconMethod.FBP_CUDA, center=10, start_row=0, end_row=1)
        with self.assertRaises(ValueError):
            build_wsl_argv(job)

    def test_lamino_search_uses_independent_range_and_step(self):
        job = ReconJob(Path(r"D:\Data\a.h5"), Path(r"D:\Out"), mode=GeometryMode.LAMINOGRAPHY,
                       method=ReconMethod.LAMINO_LINEREC, center=100, start_row=0, end_row=8,
                       try_row=4, lamino_angle=20, lamino_search_width=3, lamino_search_step=.2,
                       search_kind="angle")
        argv = build_wsl_argv(job)
        self.assertEqual(argv[argv.index("--lamino-search-width") + 1], "3")
        self.assertEqual(argv[argv.index("--lamino-search-step") + 1], "0.2")
        self.assertEqual(argv[argv.index("--try-row") + 1], "4")
        self.assertEqual(argv[argv.index("--search-kind") + 1], "angle")

    def test_lamino_full_supports_fourierrec_and_linerec(self):
        for method in (ReconMethod.LAMINO_FOURIERREC, ReconMethod.LAMINO_LINEREC):
            job = ReconJob(Path("a.h5"), Path("out"), mode=GeometryMode.LAMINOGRAPHY,
                           method=method, center=10, start_row=2, end_row=8,
                           reconstruction_type="full")
            argv = build_wsl_argv(job)
            self.assertEqual(argv[2], method.value)

    def test_lamino_try_rejects_fourierrec(self):
        job = ReconJob(Path("a.h5"), Path("out"), mode=GeometryMode.LAMINOGRAPHY,
                       method=ReconMethod.LAMINO_FOURIERREC, center=10, start_row=0, end_row=8,
                       reconstruction_type="try", search_kind="center")
        with self.assertRaisesRegex(ValueError, "searches use LineRec"):
            build_wsl_argv(job)

    def test_lprec_requires_explicit_angle_approximation_flag(self):
        job = ReconJob(Path(r"D:\Data\a.h5"), Path(r"D:\Out"), center=1024.5,
                       start_row=10, end_row=11, method=ReconMethod.LPREC,
                       uniform_angle_approximation=True)
        self.assertIn("--accept-uniform-angle-approximation", build_wsl_argv(job))

    def test_equal_angle_view_links_raw_data_and_replaces_theta(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.h5"
            view = Path(temp) / "view.h5"
            with h5py.File(source, "w") as handle:
                ex = handle.create_group("exchange")
                ex.create_dataset("data", data=np.ones((5, 2, 3), np.uint16))
                ex.create_dataset("data_white", data=np.ones((1, 2, 3), np.uint16))
                ex.create_dataset("data_dark", data=np.zeros((1, 2, 3), np.uint16))
                ex.create_dataset("theta", data=np.array([0, 20, 70, 100, 150], np.float32))
            equal_angle_view(source, view)
            with h5py.File(view, "r") as handle:
                np.testing.assert_allclose(handle["exchange/theta"][:], [0, 36, 72, 108, 144])
                self.assertEqual(handle["exchange/data"].shape, (5, 2, 3))
                self.assertIn("uniform", handle["exchange"].attrs["theta_substitution"])

    def test_linerec_rejects_unsafe_chunk(self):
        job = ReconJob(Path(r"D:\Data\a.h5"), Path(r"D:\Out"), center=1024.5,
                       start_row=10, end_row=11, method=ReconMethod.LINEREC, chunk_size=1)
        with self.assertRaisesRegex(ValueError, "chunk size 4"):
            job.validate()

    def test_binning_is_passed_as_tomocupy_exponent(self):
        job = ReconJob(Path(r"D:\Data\a.h5"), Path(r"D:\Out"), center=1028.5,
                       start_row=1008, end_row=1016, method=ReconMethod.FOURIERREC, binning=2)
        argv = build_wsl_argv(job)
        self.assertEqual(argv[argv.index("--binning") + 1], "2")
        self.assertEqual(job.center, 1028.5)

    def test_binning_rejects_too_few_original_rows(self):
        job = ReconJob(Path(r"D:\Data\a.h5"), Path(r"D:\Out"), center=1028.5,
                       start_row=1011, end_row=1015, method=ReconMethod.FBP_CUDA, binning=3,
                       reconstruction_type="full")
        with self.assertRaisesRegex(ValueError, "at least 8"):
            job.validate()

    def test_gui_try_maps_absolute_row_to_native_tomocupy_center_sweep(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self._tomocupy_args(Path(temp))
            with patch("backend_runner.tomocupy_executable", return_value=Path("tomocupy.exe")), \
                 patch("backend_runner.subprocess.run"):
                command, metadata = run_tomocupy(args)
            self.assertEqual(command[command.index("--reconstruction-type") + 1], "try")
            self.assertEqual(command[command.index("--start-row") + 1], "0")
            self.assertEqual(command[command.index("--end-row") + 1], "8")
            self.assertAlmostEqual(float(command[command.index("--nsino") + 1]), 4 / 7)
            self.assertEqual(metadata["requested_reconstruction_type"], "try")
            self.assertEqual(metadata["effective_reconstruction_type"], "try")

    def test_lamino_center_search_uses_recon_steps_try_and_linerec(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self._tomocupy_args(Path(temp), method="lamino_linerec", search_kind="center")
            with patch("backend_runner.tomocupy_executable", return_value=Path("tomocupy.exe")), \
                 patch("backend_runner.subprocess.run"):
                command, metadata = run_tomocupy(args)
            self.assertEqual(command[1], "recon_steps")
            self.assertEqual(command[command.index("--reconstruction-type") + 1], "try")
            self.assertEqual(command[command.index("--reconstruction-algorithm") + 1], "linerec")
            self.assertIn("--center-search-width", command)
            self.assertNotIn("--lamino-start-row", command)
            self.assertEqual(metadata["search_kind"], "center")

    def test_lamino_angle_search_uses_try_lamino_without_center_sweep(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self._tomocupy_args(Path(temp), method="lamino_linerec", search_kind="angle")
            with patch("backend_runner.tomocupy_executable", return_value=Path("tomocupy.exe")), \
                 patch("backend_runner.subprocess.run"):
                command, metadata = run_tomocupy(args)
            self.assertEqual(command[1], "recon_steps")
            self.assertEqual(command[command.index("--reconstruction-type") + 1], "try_lamino")
            self.assertEqual(command[command.index("--reconstruction-algorithm") + 1], "linerec")
            self.assertNotIn("--center-search-width", command)
            self.assertIn("--lamino-search-width", command)
            self.assertEqual(metadata["effective_reconstruction_type"], "try_lamino")
            self.assertIsNone(metadata["center_search_width_original_pixels"])

    def test_lamino_full_uses_requested_algorithm_and_output_row_range(self):
        for method, algorithm in (("lamino_fourierrec", "fourierrec"), ("lamino_linerec", "linerec")):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                args = self._tomocupy_args(Path(temp), method=method, reconstruction_type="full")
                with patch("backend_runner.tomocupy_executable", return_value=Path("tomocupy.exe")), \
                     patch("backend_runner.subprocess.run"):
                    command, _ = run_tomocupy(args)
                self.assertEqual(command[1], "recon_steps")
                self.assertEqual(command[command.index("--reconstruction-algorithm") + 1], algorithm)
                self.assertEqual(command[command.index("--start-row") + 1], "0")
                self.assertEqual(command[command.index("--end-row") + 1], "8")
                self.assertEqual(command[command.index("--lamino-start-row") + 1], "2")
                self.assertEqual(command[command.index("--lamino-end-row") + 1], "4")

    def test_normalization_handles_outliers(self):
        image = np.arange(100, dtype=np.float32).reshape(10, 10)
        result = normalize_uint8(image)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, image.shape)

    def test_export_display_conversion(self):
        image = np.arange(100, dtype=np.float32).reshape(10, 10)
        result8, window8 = display_convert(image, 8)
        result16, window16 = display_convert(image, 16)
        self.assertEqual(result8.dtype, np.uint8)
        self.assertEqual(result16.dtype, np.uint16)
        self.assertEqual(window8, window16)

    def test_hdf5_export_preserves_raw_dtype_and_writes_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "source.h5"; output = root / "export"
            with h5py.File(source, "w") as handle:
                ex = handle.create_group("exchange")
                ex.create_dataset("data", data=np.arange(60, dtype=np.uint16).reshape(3, 4, 5))
                ex.create_dataset("data_white", data=np.ones((1, 4, 5), np.uint16))
                ex.create_dataset("data_dark", data=np.zeros((1, 4, 5), np.uint16))
                ex.create_dataset("theta", data=np.array([0, 60, 120], np.float32))
            worker = ExportWorker(ExportJob("projection", source, output, [1], "tiff", "preserve"))
            worker.run()
            exported = tifffile.imread(output / "projection_00001.tiff")
            self.assertEqual(exported.dtype, np.uint16)
            self.assertEqual(exported.shape, (4, 5))
            metadata = __import__("json").loads((output / "export_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["count"], 1)
            self.assertEqual(metadata["records"][0]["dtype"], "uint16")


if __name__ == "__main__":
    unittest.main()
