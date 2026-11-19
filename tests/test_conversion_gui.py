import os,time,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,h5py,tifffile
from PySide6.QtWidgets import QApplication,QPushButton
from PySide6.QtCore import QTimer
from ct_recon_gui.main_window import MainWindow
from ct_recon_gui.conversion_dialog import ConversionDialog
from ct_recon_gui.models import GeometryMode

class ConversionGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def test_state_selection_job_and_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'n.h5'
            with h5py.File(path,'w') as h:
                h.create_dataset('exchange/data',data=np.ones((3,8,16),np.float32));h.create_dataset('exchange/theta',data=[0,60,120]);h.attrs['data_preparation']='normalized'
            w=MainWindow();w.input_edit.setText(str(path));w._refresh_manual_defaults()
            self.assertEqual(w.data_state_combo.currentData(),'normalized');job=w._job();self.assertEqual(job.data_state,'normalized');self.assertTrue(job.minus_log);self.assertEqual(job.stripe_method,'none')
            w.data_state_combo.setCurrentIndex(w.data_state_combo.findData('attenuation'));self.assertFalse(w._job().minus_log)
            buttons={b.text() for b in w.findChildren(QPushButton)};self.assertTrue({'Images → H5','H5 → Images'}<=buttons)
            with patch('ct_recon_gui.input_state.PROCESSED_METHODS',{'linerec'}):w._mode_changed();self.assertEqual(w.method_combo.count(),1)
            w.close();self.app.processEvents()
    def test_conversion_worker_and_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);folder=r/'images';folder.mkdir()
            for i in range(10):tifffile.imwrite(folder/f'{i}.tiff',np.ones((32,32),np.uint16)*i)
            dialog=ConversionDialog('import',source=str(folder));dialog.state.setCurrentIndex(dialog.state.findData('attenuation'));dialog.output.setText(str(r/'converted.h5'));dialog.start_task()
            deadline=time.time()+10
            while dialog.task and time.time()<deadline:self.app.processEvents();time.sleep(.01)
            self.assertIsNone(dialog.task);self.assertTrue((r/'converted.h5').exists());dialog.close()
            export=ConversionDialog('export',source=str(r/'converted.h5'));export.output.setText(str(r/'exported'));export.start_task();export.close()
            while export.task and time.time()<deadline:self.app.processEvents();time.sleep(.01)
            self.assertIsNone(export.task);self.assertFalse(list(r.glob('*.partial-*')))
