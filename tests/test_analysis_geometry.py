"""Analysis geometry is explicit and reaches the worker job unchanged."""
import os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
import h5py,numpy as np
from PySide6.QtWidgets import QApplication
from ct_recon_gui.main_window import MainWindow

class AnalysisGeometryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.env=patch.dict(os.environ,BL62_RECON_SETTINGS=str(self.root/'settings.json'));self.env.start()
  self.window=MainWindow()
 def tearDown(self):self.window.close();self.app.processEvents();self.env.stop();self.tmp.cleanup()
 def test_default_tomography_accept_angle_and_cancel(self):
  w=self.window;self.assertFalse(w.analysis_laminography.isChecked())
  with patch('ct_recon_gui.analysis_ui.QInputDialog.getDouble',return_value=(27.5,True)) as dialog:
   w.analysis_laminography.setChecked(True)
  dialog.assert_called_once();self.assertEqual(w.analysis_lamino_angle,27.5)
  w.analysis_laminography.setChecked(False)
  with patch('ct_recon_gui.analysis_ui.QInputDialog.getDouble',return_value=(20.,False)):
   w.analysis_laminography.setChecked(True)
  self.assertFalse(w.analysis_laminography.isChecked())
 def test_geometry_and_angle_are_part_of_actual_runner_configuration(self):
  w=self.window;source=self.root/'input.h5'
  with h5py.File(source,'w') as h:
   h['exchange/data']=np.ones((3,8,8),np.uint16)
   h['exchange/data_white']=np.ones((1,8,8),np.uint16)
   h['exchange/data_dark']=np.zeros((1,8,8),np.uint16)
   h['exchange/theta']=np.array([0.,60.,120.])
  checkpoint=self.root/'model.pt';checkpoint.touch();repo=self.root/'repo';(repo/'sam3').mkdir(parents=True)
  w.settings.update(sam3_python=sys.executable,recon_python=sys.executable,checkpoint=str(checkpoint),repo=str(repo),gpu=0,output_root=str(self.root))
  w.analysis_input.setText(str(source));configs=[]
  with patch.object(w.analysis_runner,'start',side_effect=lambda cfg,**kw:configs.append(cfg)),patch('ct_recon_gui.analysis_ui.QMessageBox.critical') as error:
   w._start_analysis()
   self.assertEqual(configs[-1]['geometry'],'tomography');self.assertEqual(configs[-1]['method'],'fourierrec')
   with patch('ct_recon_gui.analysis_ui.QInputDialog.getDouble',return_value=(27.5,True)):w.analysis_laminography.setChecked(True)
   w._start_analysis();self.assertEqual(configs[-1]['geometry'],'laminography');self.assertEqual(configs[-1]['lamino_angle'],27.5)
   self.assertEqual(configs[-1]['search_method'],'linerec');self.assertEqual(configs[-1]['method'],'lamino_fourierrec')
   error.assert_not_called()
if __name__=='__main__':unittest.main()
