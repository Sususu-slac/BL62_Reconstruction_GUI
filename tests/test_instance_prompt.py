from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from ct_recon_gui.instance_dialog import InstanceDialog
from ct_recon_gui.main_window import MainWindow
from ct_recon_gui.analysis_runner import AnalysisRunner
from sam3_rotation_center.exemplar import (InstanceExample, example_preview,
    isolated_reference, load_example, mask_box, read_reference, save_example)


def example():
    y,x = np.mgrid[:128,:160]
    pixels = np.stack([(x*3+y)%256, (x+y*2)%256, (x+y)%256], axis=2).astype(np.uint8)
    mask = (x-80)**2+(y-64)**2 <= 30**2
    mask[(x-80)**2+(y-64)**2 < 8**2] = False
    return InstanceExample(Image.fromarray(pixels), mask, 'reference.tiff')


class ExampleTests(unittest.TestCase):
    def test_hole_survives_storage_and_job_copy_is_independent(self):
        source = example()
        original = np.asarray(source.image).copy()
        with tempfile.TemporaryDirectory() as temp:
            paths = save_example(source, Path(temp)/'reference')
            source.mask[:] = False
            restored = load_example(paths)
            self.assertFalse(restored.mask[64,80])
            self.assertTrue(restored.mask[64,100])
            np.testing.assert_array_equal(np.asarray(restored.image), original)
            metadata = json.loads(Path(paths['metadata']).read_text())
            self.assertEqual(metadata['encoding'], 'mask_isolated_reference_box_features')
            self.assertEqual(metadata['mask_pixels'], int(restored.mask.sum()))

    def test_mask_changes_actual_reference_even_with_identical_box(self):
        hollow = example()
        solid = InstanceExample(hollow.image, hollow.mask.copy())
        solid.mask[64-5:64+6,80-5:80+6] = True
        self.assertEqual(mask_box(hollow.mask), mask_box(solid.mask))
        image_hollow, fill_hollow = isolated_reference(hollow)
        image_solid, fill_solid = isolated_reference(solid)
        self.assertEqual(fill_hollow, fill_solid)
        self.assertNotEqual(image_hollow.tobytes(), image_solid.tobytes())
        np.testing.assert_array_equal(np.asarray(image_hollow)[64,80], fill_hollow)
        np.testing.assert_array_equal(np.asarray(image_solid)[64,80], np.asarray(hollow.image)[64,80])
        self.assertEqual(image_hollow.size, hollow.image.size)

    def test_preview_is_cropped_outline_including_hole(self):
        source = example()
        crop = example_preview(source)
        self.assertLess(crop.width, source.image.width)
        pixels = np.asarray(crop)
        self.assertTrue(np.any(np.all(pixels == [99,244,181], axis=2)))
        # Underlying source remains intact.
        self.assertFalse(np.array_equal(np.asarray(source.image)[64,80], [99,244,181]))

    def test_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = save_example(example(), Path(temp)/'reference')
            Image.new('L',(160,128)).save(paths['mask'])
            with self.assertRaisesRegex(ValueError,'has changed'):
                load_example(paths)

    def test_reference_16_and_32_bit_do_not_overwrite_original(self):
        with tempfile.TemporaryDirectory() as temp:
            for dtype, suffix in [(np.uint16,'16'),(np.float32,'32')]:
                path = Path(temp)/(suffix+'.tiff')
                array = np.arange(2048).reshape(32,64).astype(dtype)
                Image.fromarray(array).save(path)
                before = path.read_bytes()
                loaded = read_reference(path)
                self.assertEqual(loaded.mode,'RGB')
                self.assertEqual(loaded.size,(64,32))
                self.assertEqual(before,path.read_bytes())
                self.assertGreater(np.asarray(loaded).max(),200)


class InstanceGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_real_mouse_brush_eraser_undo_redo_and_confirm(self):
        dialog = InstanceDialog(example().image,'reference.tiff')
        dialog.show()
        self.app.processEvents()
        dialog.canvas.fit_image()
        center = dialog.canvas.mapFromScene(QPointF(80,64))
        dialog.size_spin.setValue(48)
        QTest.mouseClick(dialog.canvas.viewport(),Qt.MouseButton.LeftButton,pos=center)
        self.assertTrue(dialog.canvas.mask()[64,80])
        dialog.eraser.click()
        dialog.size_spin.setValue(12)
        QTest.mouseClick(dialog.canvas.viewport(),Qt.MouseButton.LeftButton,pos=center)
        self.assertFalse(dialog.canvas.mask()[64,80])
        self.assertTrue(dialog.canvas.mask()[64,96])
        dialog.undo_button.click()
        self.assertTrue(dialog.canvas.mask()[64,80])
        dialog.redo_button.click()
        self.assertFalse(dialog.canvas.mask()[64,80])
        dialog._confirm()
        self.assertEqual(dialog.result(),QDialog.DialogCode.Accepted)
        self.assertFalse(dialog.example.mask[64,80])
        self.assertFalse(dialog.isVisible())

    def test_empty_confirm_and_cancel(self):
        dialog = InstanceDialog(example().image,'reference.tiff')
        with patch.object(QMessageBox,'warning') as warning:
            dialog._confirm()
        warning.assert_called_once()
        self.assertIsNone(dialog.example)
        dialog.reject()
        self.assertEqual(dialog.result(),QDialog.DialogCode.Rejected)

    def test_mode_switch_preview_and_cancel_preserve_example(self):
        window = MainWindow()
        window.show()
        window.main_tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.assertEqual(window.analysis_prompt_mode.currentData(),'text')
        self.assertTrue(window.analysis_prompt.isVisible())
        window.analysis_prompt_mode.setCurrentIndex(1)
        self.assertFalse(window.analysis_prompt.isVisible())
        self.assertTrue(window.analysis_add_image.isVisible())
        self.assertFalse(window.analysis_instance_preview.isVisible())
        selected = example()
        window._set_instance_example(selected)
        self.assertTrue(window.analysis_instance_preview.isVisible())
        self.assertFalse(window.analysis_instance_preview.pixmap().isNull())
        with patch('ct_recon_gui.analysis_ui.QFileDialog.getOpenFileName', return_value=('','')):
            window.analysis_add_image.click()
        self.assertIs(window.analysis_example, selected)
        window.analysis_prompt_mode.setCurrentIndex(0)
        self.assertEqual(window.analysis_prompt.text(),'particle')
        self.assertTrue(window.analysis_prompt.isVisible())
        self.assertFalse(window.analysis_instance_panel.isVisible())
        self.assertTrue(window.segment_all.isChecked())
        window.close()

    def test_runner_saves_frozen_assets_before_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            runner=AnalysisRunner()
            config=dict(output=str(Path(temp)/'job'),prompt_mode='instance',prompt='visual',sam3_python='missing.exe')
            with patch.object(runner.process,'start'),patch.object(runner.process,'waitForStarted',return_value=True):
                runner.start(config,instance_example=example())
            runner.timer.stop()
            saved=json.loads((Path(config['output'])/'analysis_job.json').read_text())
            self.assertEqual(saved['prompt_mode'],'instance')
            self.assertFalse(load_example(saved['instance_reference']).mask[64,80])
            self.assertNotIn('instance_reference',config)

    def test_running_analysis_locks_reference_and_mode(self):
        window=MainWindow()
        with patch.object(AnalysisRunner,'running',new_callable=lambda:property(lambda self: True)):
            window._update_busy()
            self.assertFalse(window.analysis_prompt_mode.isEnabled())
            self.assertFalse(window.analysis_add_image.isEnabled())
        window.close()


if __name__=='__main__':
    unittest.main()
