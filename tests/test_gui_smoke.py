from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import tifffile

from PySide6.QtWidgets import QApplication, QMessageBox

from ct_recon_gui.commands import build_wsl_argv
from ct_recon_gui.main_window import MainWindow
from ct_recon_gui.job_runner import JobRunner, file_signature
from ct_recon_gui.models import GeometryMode, ReconMethod


class GuiSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_constructs(self):
        window=MainWindow()
        self.assertEqual(window.windowTitle(),'BL62 Reconstruction')
        self.assertEqual(window.main_tabs.count(),2)
        self.assertEqual(window.state_label.text(),'Ready')
        self.assertEqual(window.method_combo.count(),6)
        self.assertEqual(window.input_source_combo.count(),5)
        self.assertEqual(window.binning_combo.count(),4)
        self.assertEqual(window.export_source_combo.count(),6)
        self.assertEqual(window.center_tabs.count(),3)
        self.assertFalse(window.type_combo.isVisible())
        self.assertFalse(window.log_toggle.isChecked())
        self.assertFalse(window.log_box.isVisible())
        self.assertEqual(window.run_btn.text(),'Run Reconstruction')
        self.assertTrue(window.segment_all.isChecked())
        window.mode_combo.setCurrentIndex(window.mode_combo.findData(GeometryMode.LAMINOGRAPHY))
        self.assertEqual(window.method_combo.count(),2)
        self.assertTrue(window.params_form.isRowVisible(window.lamino_row))
        self.assertTrue(window.find_angle_btn.isEnabled())
        self.assertEqual(window.find_center_btn.size(),window.find_angle_btn.size())
        window.mode_combo.setCurrentIndex(window.mode_combo.findData(GeometryMode.TOMOGRAPHY))
        self.assertFalse(window.params_form.isRowVisible(window.lamino_row))
        window.input_edit.setText(r'D:\Data\sample.h5');window.output_edit.setText(r'D:\Output');window.center_spin.setValue(100)
        job=window._job('full');self.assertIsInstance(job.mode,GeometryMode);self.assertIsInstance(job.method,ReconMethod)
        window.close()

    def test_segmentation_widget_identifies_reconstruction_as_completed(self):
        window=MainWindow()
        window._analysis_updated(dict(stage='Segmenting',round=7,completed=10,total=2048,
                                      detail='Image 11/2048 · Saving masks'))
        self.assertIn('Reconstruction completed — segmenting all data',window.analysis_stage.text())
        self.assertEqual(window.analysis_round_progress.text(),'10 / 2048 images')
        self.assertEqual(window.analysis_progress.value(),10)
        self.assertIn('Saving masks',window.analysis_detail.text());window.close()

    def test_forced_stop_does_not_reset_elapsed_to_old_worker_status(self):
        import time
        window=MainWindow();window.analysis_status={'stage':'Loading SAM3','elapsed_seconds':2}
        window.analysis_started=time.monotonic()-73
        window._analysis_finished(False,'Stopped')
        self.assertEqual(window.analysis_elapsed.text(),'Total elapsed 01:13');window.close()

    def test_curve_widget_zooms_to_search_but_retains_global_data(self):
        with tempfile.TemporaryDirectory() as temp:
            import json
            graph={'centers':[0,1020,1025,1030,2047],'scores':[1,100,104,101,1],'best_center':1025,
                   'curve':{'start':0,'step':1,'scores':[float(1+100*np.exp(-((x-1025)/100)**2)) for x in range(2048)]},
                   'filter_bounds':[0,2047],'view_intervals':[[1020,1030]]}
            path=Path(temp)/'plot.json';path.write_text(json.dumps(graph))
            window=MainWindow();window.analysis_status={'curve_file':str(path)};window._refresh_analysis_preview()
            low,high=window.analysis_plot.viewRange()[0]
            self.assertLess(high-low,20);self.assertLessEqual(low,1020);self.assertGreaterEqual(high,1030)
            self.assertEqual(len(window.filtered_curve.getData()[0]),2048)
            window._show_global_curve();low,high=window.analysis_plot.viewRange()[0]
            self.assertAlmostEqual(low,0);self.assertAlmostEqual(high,2047);window.close()

    def test_result_watcher_ignores_old_files_but_detects_updated_file(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp); path = output / "recon_00001.tiff"
            tifffile.imwrite(path, np.zeros((8, 8), np.float32))
            runner = JobRunner(); runner.output_dir = output
            runner.signatures = {path: file_signature(path)}
            found = []; runner.result_found.connect(found.append)
            runner._scan_results(); self.assertEqual(found, [])
            tifffile.imwrite(path, np.ones((8, 8), np.float32))
            runner._scan_results(); self.assertEqual(found, [])
            runner._scan_results(); self.assertEqual(found, [str(path)])
            tifffile.imwrite(path, np.full((8, 8), 2, np.float32))
            runner._scan_results(); self.assertEqual(found, [str(path)])
            runner._scan_results(force=True); self.assertEqual(found, [str(path), str(path)])

    def test_duplicate_center_result_path_is_listed_once(self):
        with tempfile.TemporaryDirectory() as temp:
            window = MainWindow()
            candidate = Path(temp) / "recon_slice0100_center1028.50.tiff"
            tifffile.imwrite(candidate, np.ones((8, 8), np.float32))
            window._new_result(str(candidate))
            window._new_result(str(candidate))
            self.assertEqual(window.center_candidates, [(1028.5, candidate)])
            window.close()

    def test_lprec_angle_conversion_starts_automatically(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "sample.h5"
            with h5py.File(source, "w") as handle:
                exchange = handle.create_group("exchange")
                exchange.create_dataset("data", data=np.ones((5, 8, 10), np.uint16))
                exchange.create_dataset("data_white", data=np.ones((1, 8, 10), np.uint16))
                exchange.create_dataset("data_dark", data=np.zeros((1, 8, 10), np.uint16))
                exchange.create_dataset("theta", data=np.linspace(0, 180, 5))

            window = MainWindow()
            window.input_edit.setText(str(source))
            window.output_edit.setText(str(root / "sample_reconstruction"))
            window.center_spin.setValue(4.5)
            window.method_combo.setCurrentIndex(window.method_combo.findData(ReconMethod.LPREC))
            jobs = []
            errors = []
            with patch.object(window.runner, "start", side_effect=jobs.append), \
                 patch.object(QMessageBox, "question", side_effect=AssertionError("Unexpected confirmation")), \
                 patch.object(QMessageBox, "critical", side_effect=lambda *args: errors.append(args[2])):
                window._start_job()

            self.assertEqual(errors, [])
            self.assertEqual(len(jobs), 1)
            self.assertTrue(jobs[0].uniform_angle_approximation)
            self.assertIn("--accept-uniform-angle-approximation", build_wsl_argv(jobs[0]))
            self.assertIn("automatically create an external-link HDF5 view", window.log_box.toPlainText())
            window.close()

    def test_apply_center_handles_qt_string_item_data(self):
        with tempfile.TemporaryDirectory() as temp:
            window = MainWindow(); window.output_edit.setText(temp); window.input_edit.setText(r"D:\Data\sample.h5")
            candidate = Path(temp) / "recon_slice0100_center1028.50.tiff"
            tifffile.imwrite(candidate, np.ones((8, 8), np.float32))
            window.center_candidates = [(1028.5, candidate)]
            window.center_candidate_slider.setRange(0, 0)
            window._apply_selected_center()
            self.assertEqual(window.center_spin.value(), 1028.5)
            self.assertTrue((Path(temp) / "selected_rotation_center.json").exists())
            window.close()

    def test_direct_center_search_preserves_selected_astra_method(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'sample.h5'
            with h5py.File(source,'w') as handle:
                ex=handle.create_group('exchange')
                ex.create_dataset('data',data=np.ones((5,8,10),np.uint16))
                ex.create_dataset('data_white',data=np.ones((1,8,10),np.uint16))
                ex.create_dataset('data_dark',data=np.zeros((1,8,10),np.uint16))
                ex.create_dataset('theta',data=np.linspace(0,180,5,endpoint=False))
            window=MainWindow();window.input_edit.setText(str(source));window.output_edit.setText(str(root/'output'))
            window.center_spin.setValue(4.5);window.method_combo.setCurrentIndex(window.method_combo.findData(ReconMethod.FBP_CUDA))
            jobs=[]
            with patch.object(window.runner,'start',side_effect=jobs.append):window.find_center_btn.click()
            self.assertEqual(len(jobs),1);self.assertEqual(jobs[0].method,ReconMethod.FOURIERREC)
            self.assertEqual(window.method_combo.currentData(),ReconMethod.FBP_CUDA)
            self.assertEqual(window.type_combo.currentData(),'full');window.close()

    def test_sequential_runs_refresh_input_and_use_separate_output_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = []
            for name, rows in (("first", 8), ("second", 12)):
                source = root / f"{name}.h5"
                with h5py.File(source, "w") as handle:
                    exchange = handle.create_group("exchange")
                    exchange.create_dataset("data", data=np.ones((5, rows, 10), np.uint16))
                    exchange.create_dataset("data_white", data=np.ones((1, rows, 10), np.uint16))
                    exchange.create_dataset("data_dark", data=np.zeros((1, rows, 10), np.uint16))
                    exchange.create_dataset("theta", data=np.linspace(0, 180, 5, endpoint=False))
                sources.append(source)

            window = MainWindow()
            window.center_spin.setValue(4.5)
            window.start_spin.setValue(0); window.end_spin.setValue(5); window.try_row_spin.setValue(3)
            jobs = []
            with patch.object(window.runner, "start", side_effect=jobs.append):
                window.input_edit.setText(str(sources[0]))
                window.output_edit.setText(str(root / "first_reconstruction"))
                window._start_job()
                window.center_spin.setValue(5.0)
                window._start_job()
                window.input_edit.setText(str(sources[1]))
                window.output_edit.setText(str(root / "second_reconstruction"))
                window._start_job()

            self.assertEqual(len(jobs), 3)
            self.assertEqual(len({job.output_dir for job in jobs}), 3)
            self.assertTrue(all(job.output_dir.name.startswith("run_") for job in jobs))
            self.assertEqual(jobs[1].center, 5.0)
            self.assertEqual(window.info.data_shape[1], 12)
            window.close()

    def test_try_completion_keeps_candidate_image_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            window = MainWindow()
            candidate = Path(temp) / "recon_slice0100_center1028.50.tiff"
            tifffile.imwrite(candidate, np.ones((8, 8), np.float32))
            normal = Path(temp) / "recon_00100.tiff"
            tifffile.imwrite(normal, np.ones((8, 8), np.float32))
            window.active_selected_type = "center_try"
            window.center_candidates = [(1028.5, candidate)]
            window.results = [normal]
            window.center_candidate_slider.setRange(0, 0)
            window.center_tabs.setCurrentWidget(window.result_view)
            window._job_finished(True, "Completed")
            self.assertIs(window.center_tabs.currentWidget(), window.center_search_tab)
            self.assertIn("center 1028.500", window.center_candidate_label.text())
            self.assertIsNotNone(window.center_search_view._pixmap)
            window.close()

    def test_full_results_can_be_browsed_with_slider(self):
        with tempfile.TemporaryDirectory() as temp:
            window = MainWindow()
            paths = []
            for row, value in ((12, 12.0), (10, 10.0), (11, 11.0)):
                path = Path(temp) / f"recon_{row:05d}.tiff"
                tifffile.imwrite(path, np.full((8, 8), value, np.float32))
                paths.append(path)
                window._new_result(str(path))
            self.assertTrue(window.result_slider.isEnabled())
            self.assertEqual(window.result_slider.maximum(), 2)
            self.assertEqual([path.name for path in window.results], ["recon_00010.tiff", "recon_00011.tiff", "recon_00012.tiff"])
            window.result_slider.setValue(0)
            self.assertEqual(Path(window.result_combo.currentData()).name, "recon_00010.tiff")
            self.assertIn("1 / 3", window.result_position_label.text())
            self.assertIs(window.center_tabs.currentWidget(), window.reconstruction_tab)
            window.next_result_btn.click()
            self.assertEqual(Path(window.result_combo.currentData()).name, "recon_00011.tiff")
            window.close()


if __name__ == "__main__":
    unittest.main()
