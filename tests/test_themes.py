"""Theme preference and visible data must survive switching during a task."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from ct_recon_gui.main_window import MainWindow
from ct_recon_gui.analysis_runner import AnalysisRunner
from ct_recon_gui.analysis_settings import AnalysisSettings
from ct_recon_gui.instance_dialog import InstanceDialog
from ct_recon_gui.runtime import load_settings,save_theme_preference
from ct_recon_gui.themes import COLORS
from sam3_rotation_center.exemplar import InstanceExample


class ThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        cls.app.setStyle('Fusion')

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'settings.json'
        self.patch=patch('ct_recon_gui.runtime.settings_path',return_value=self.path)
        self.patch.start()

    def tearDown(self):
        self.patch.stop();self.temp.cleanup()

    def test_first_launch_dark_and_button_persists_light_on_reopen(self):
        window=MainWindow()
        self.assertEqual(window.theme_name,'dark')
        self.assertEqual(window.theme_button.text(),'Switch to Light')
        self.assertFalse(self.path.exists())
        window.theme_button.click()
        self.assertEqual(window.theme_button.text(),'Switch to Dark')
        self.assertEqual(json.loads(self.path.read_text()),{'theme':'light'})
        window.close()
        reopened=MainWindow()
        self.assertEqual(reopened.theme_name,'light')
        self.assertEqual(reopened.palette().color(QPalette.ColorRole.Window).name(),COLORS['light']['window'])
        reopened.close()

    def test_preference_preserves_existing_settings_and_does_not_save_unsaved_changes(self):
        self.path.write_text(json.dumps(dict(theme='dark',gpu=0,checkpoint='shared.pt',future_option=37)))
        window=MainWindow();window.settings['checkpoint']='unsaved.pt'
        window.theme_button.click()
        saved=json.loads(self.path.read_text())
        self.assertEqual(saved,dict(theme='light',gpu=0,checkpoint='shared.pt',future_option=37))
        self.assertEqual(window.settings['checkpoint'],'unsaved.pt')
        window.close()

    def test_toggle_preserves_images_masks_curve_bounds_and_progress(self):
        window=MainWindow()
        window.show();self.app.processEvents()
        self.assertEqual(window.analysis_plot.getAxis('bottom').labelText,'Rotation Center')
        self.assertEqual(window.analysis_plot.getAxis('bottom').labelUnits,'px')
        self.assertEqual(window.analysis_plot.getAxis('left').labelText,'Confidence-weighted Score')
        pixels=np.arange(120,dtype=np.uint8).reshape(10,12)
        window.analysis_view.show_array(pixels)
        image_key=window.analysis_view._pixmap.cacheKey()
        sample=InstanceExample(Image.fromarray(pixels).convert('RGB'),pixels>60,'reference.png')
        window._set_instance_example(sample)
        preview_key=window.analysis_instance_preview.pixmap().cacheKey()
        window.analysis_prompt_mode.setCurrentIndex(1)
        window.analysis_prompt.setText('preserve target')
        window.raw_curve.setData([10,11,12],[1,3,2])
        window.filtered_curve.setData([10,11,12],[1.2,2.6,2])
        window.analysis_plot.setXRange(10,12,padding=0)
        window.analysis_plot.setYRange(0,4,padding=0)
        window.best_marker.setPos(11);window.best_marker.show()
        window._analysis_updated(dict(stage='Finding Center',round=2,completed=3,total=8))
        dialog=InstanceDialog(sample.image,'reference.png',window)
        dialog.canvas._restore(np.packbits(sample.mask))
        for expected in ('light','dark','light','dark'):
            window.theme_button.click();self.app.processEvents()
            self.assertEqual(window.theme_name,expected)
            self.assertEqual(window.theme_button.text(),'Switch to '+('Dark' if expected=='light' else 'Light'))
            self.assertEqual(window.analysis_view._pixmap.cacheKey(),image_key)
            self.assertEqual(window.analysis_instance_preview.pixmap().cacheKey(),preview_key)
            np.testing.assert_array_equal(dialog.canvas.mask(),sample.mask)
            np.testing.assert_array_equal(window.raw_curve.getData()[1],[1,3,2])
            np.testing.assert_array_equal(window.filtered_curve.getData()[1],[1.2,2.6,2])
            np.testing.assert_allclose(window.analysis_plot.viewRange()[0],[10,12])
            self.assertEqual(window.analysis_progress.value(),3)
            self.assertEqual(window.analysis_prompt.text(),'preserve target')
            self.assertEqual(window.analysis_prompt_mode.currentData(),'instance')
            self.assertEqual(dialog.canvas.backgroundBrush().color().name(),COLORS[expected]['canvas'])
            self.assertEqual(window.best_marker.value(),11)
            self.assertEqual(window.analysis_plot.getAxis('bottom').labelText,'Rotation Center')
            self.assertEqual(window.analysis_plot.getAxis('bottom').labelUnits,'px')
            self.assertEqual(window.analysis_plot.getAxis('left').labelText,'Confidence-weighted Score')
        dialog.reject();window.close()

    def test_theme_stays_available_while_analysis_controls_are_locked(self):
        window=MainWindow()
        with patch.object(AnalysisRunner,'running',property(lambda self:True)):
            window._update_busy()
            self.assertFalse(window.analysis_start.isEnabled())
            self.assertFalse(window.analysis_add_image.isEnabled())
            self.assertTrue(window.theme_button.isEnabled())
            window.theme_button.click()
            self.assertFalse(window.analysis_start.isEnabled())
            self.assertTrue(window.theme_button.isEnabled())
        window.close()

    def test_dialogs_inherit_palette_and_file_picker_is_readable(self):
        window=MainWindow();window.theme_button.click()
        dialogs=[AnalysisSettings(window.settings,[],window),QMessageBox(window),QFileDialog(window)]
        dialogs[-1].setOption(QFileDialog.Option.DontUseNativeDialog,True)
        for dialog in dialogs:
            dialog.show();self.app.processEvents()
            self.assertEqual(dialog.palette().color(QPalette.ColorRole.Window).name(),COLORS['light']['window'])
            self.assertEqual(dialog.palette().color(QPalette.ColorRole.Text).name(),COLORS['light']['text'])
            dialog.reject()
        window.close()

    def test_bad_theme_preference_falls_back_and_settings_merge_handles_invalid_json(self):
        self.path.write_text('{"theme":"unknown"}')
        window=MainWindow();self.assertEqual(window.theme_name,'dark');window.close()
        self.path.write_text('invalid json')
        save_theme_preference('light')
        self.assertEqual(load_settings()['theme'],'light')


if __name__=='__main__':unittest.main()
