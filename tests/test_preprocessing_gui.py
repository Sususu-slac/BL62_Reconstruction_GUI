import os,tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
from PySide6.QtWidgets import QApplication,QDialog
from ct_recon_gui.main_window import MainWindow
from ct_recon_gui.preprocessing_dialog import PreprocessingWidget,PreprocessingDialog
from ct_recon_gui.analysis_settings import AnalysisSettings
from ct_recon_gui.commands import build_wsl_argv
from ct_recon_gui.models import GeometryMode,ReconMethod
class PreprocessingTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'settings.json';self.env=patch.dict(os.environ,BL62_RECON_SETTINGS=str(self.path));self.env.start();self.w=MainWindow()
 def tearDown(self):self.w.close();self.app.processEvents();self.env.stop();self.temp.cleanup()
 def test_defaults_and_dezinger_threshold_enable(self):
  editor=PreprocessingWidget(self.w.settings);self.assertEqual(editor.settings()['remove_stripe_method'],'vo-all');self.assertFalse(editor.threshold.isEnabled());editor.dezinger.setValue(5);self.assertTrue(editor.threshold.isEnabled());self.assertTrue(editor.settings()['minus_log']);editor.close()
 def test_manual_dialog_saves_and_routes_all_parameters_for_both_geometries(self):
  def accept(dialog):
   e=dialog.editor;e.dezinger.setValue(5);e.threshold.setValue(1234);e.ratio.setValue(1.2);e.linear.setChecked(True);e.log.setChecked(False);e.filter.setCurrentIndex(e.filter.findData('hann'));return QDialog.Accepted
  with patch.object(PreprocessingDialog,'exec',accept):self.w._open_preprocessing()
  self.assertEqual(json.loads(self.path.read_text())['dezinger'],5)
  self.w.input_edit.setText('input.h5');self.w.output_edit.setText('output');self.w.center_spin.setValue(32)
  for mode in GeometryMode:
   self.w.mode_combo.setCurrentIndex(self.w.mode_combo.findData(mode));job=self.w._job('full');args=build_wsl_argv(job)
   for key,value in [('--dezinger','5'),('--dezinger-threshold','1234'),('--bright-ratio','1.2'),('--minus-log','False'),('--flat-linear','True'),('--filter','hann')]:self.assertEqual(args[args.index(key)+1],value)
 def test_analysis_settings_use_same_preprocessing_contract(self):
  dialog=AnalysisSettings(self.w.settings,[],self.w);dialog.preprocessing.dezinger.setValue(5);dialog.preprocessing.stripe.setCurrentIndex(dialog.preprocessing.stripe.findData('ti'));result=dialog.result_settings();self.assertEqual(result['dezinger'],5);self.assertEqual(result['remove_stripe_method'],'ti');dialog.close()
 def test_cancel_does_not_change_saved_or_current_settings(self):
  before=dict(self.w.settings)
  with patch.object(PreprocessingDialog,'exec',return_value=QDialog.Rejected):self.w._open_preprocessing()
  self.assertEqual(self.w.settings,before);self.assertFalse(self.path.exists())
if __name__=='__main__':unittest.main()
