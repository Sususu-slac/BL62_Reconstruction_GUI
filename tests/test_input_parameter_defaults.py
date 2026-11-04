import os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np,h5py
from PySide6.QtWidgets import QApplication
from ct_recon_gui.main_window import MainWindow
class InputDefaultsTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.r=Path(self.temp.name);self.env=patch.dict(os.environ,BL62_RECON_SETTINGS=str(self.r/'settings.json'));self.env.start();self.w=MainWindow();self.sources=[]
  for rows,width in [(64,128),(15,80)]:
   path=self.r/f'data{rows}.h5'
   with h5py.File(path,'w') as h:
    h['exchange/data']=np.ones((7,rows,width),np.uint16);h['exchange/data_white']=np.ones((3,rows,width),np.uint16)*2;h['exchange/data_dark']=np.zeros((1,rows,width),np.uint16);h['exchange/theta']=np.arange(7)*30.
   self.sources.append(path)
 def tearDown(self):self.w.close();self.app.processEvents();self.env.stop();self.temp.cleanup()
 def test_paste_switch_resets_full_range_middle_and_limits(self):
  for path,rows in zip(self.sources,[64,15]):
   self.w.input_edit.setText(str(path));self.assertEqual(self.w.start_spin.value(),0);self.assertEqual(self.w.end_spin.value(),rows);self.assertEqual(self.w.try_row_spin.value(),rows//2);self.assertEqual(self.w.end_spin.maximum(),rows)
   self.w.start_spin.setValue(2);self.w.end_spin.setValue(8);self.w.try_row_spin.setValue(3)
 def test_same_input_keeps_user_customizations(self):
  self.w.input_edit.setText(str(self.sources[0]));self.w.start_spin.setValue(4);self.w.end_spin.setValue(12);self.w.try_row_spin.setValue(5);self.w._refresh_manual_defaults();self.assertEqual(self.w.start_spin.value(),4);self.assertEqual(self.w.end_spin.value(),12);self.assertEqual(self.w.try_row_spin.value(),5)
 def test_automatic_new_input_clears_previous_custom_range_and_row(self):
  self.w.analysis_input.setText(str(self.sources[0]));self.w.settings.update(start_row=2,end_row=8,search_row=3);self.w.analysis_input.setText(str(self.sources[1]));self.assertIsNone(self.w.settings['start_row']);self.assertIsNone(self.w.settings['end_row']);self.assertIsNone(self.w.settings['search_row']);self.assertEqual(self.w._analysis_dimensions[1],15)
  from ct_recon_gui.analysis_settings import AnalysisSettings
  dialog=AnalysisSettings(self.w.settings,[],self.w);self.assertTrue(dialog.all_rows.isChecked());self.assertEqual(dialog.end.value(),15);self.assertEqual(dialog.row.value(),7);dialog.close()
 def test_browse_also_uses_full_range_and_middle(self):
  with patch('ct_recon_gui.main_window.QFileDialog.getOpenFileName',return_value=(str(self.sources[0]),'')):self.w._choose_input()
  self.assertEqual((self.w.start_spin.value(),self.w.end_spin.value(),self.w.try_row_spin.value()),(0,64,32))
if __name__=='__main__':unittest.main()
