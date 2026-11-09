import tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import tifffile
from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from ct_recon_gui.job_runner import JobRunner,file_signature
from ct_recon_gui.image_view import ImageView
from ct_recon_gui.main_window import MainWindow

class ResponsivenessTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
 def wait(self,predicate,timeout=5000):
  end=time.monotonic()+timeout/1000
  while not predicate() and time.monotonic()<end:QTest.qWait(10)
  self.assertTrue(predicate())
 def test_final_scan_keeps_event_loop_alive_and_finishes_after_results(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'recon_00001.tiff';tifffile.imwrite(p,np.ones((8,8),np.float32))
   runner=JobRunner();runner.output_dir=Path(tmp);order=[];beats=[]
   runner.results_found.connect(lambda paths:order.append(('results',paths)))
   runner.finished.connect(lambda ok,msg:order.append(('finished',ok)))
   timer=QTimer();timer.setInterval(10);timer.timeout.connect(lambda:beats.append(time.monotonic()));timer.start()
   def slow(path):time.sleep(.25);return file_signature(path)
   with patch('ct_recon_gui.job_runner.file_signature',side_effect=slow):
    runner._request_scan();runner._finished(0,None)
    self.assertTrue(runner.running)
    self.wait(lambda:bool(order) and order[-1][0]=='finished')
   timer.stop();self.assertEqual(order,[('results',[str(p)]),('finished',True)])
   self.assertGreater(len(beats),10);self.assertFalse(runner.running)
 def test_batch_of_400_candidates_requests_one_preview(self):
  window=MainWindow()
  names=[f'C:/fake/recon_slice100_center{x*.5:.2f}.tiff' for x in range(400)]
  with patch.object(window.center_search_view,'show_tiff') as show:
   window._new_results(names);self.assertEqual(show.call_count,1)
  self.assertEqual(len(window.center_candidates),400)
  self.assertEqual(window.center_tabs.tabText(2),'Parameter Optimization')
  self.assertEqual(window.apply_center_btn.text(),'Apply Selected Parameter')
  self.assertEqual(window.preprocessing_btn.text(),'⚙');self.assertEqual(window.preprocessing_btn.width(),28)
  window.close()
 def test_partial_tiff_is_not_published_until_complete(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'recon_00001.tiff'
   runner=JobRunner();runner.output_dir=Path(tmp);found=[]
   runner.results_found.connect(found.extend)
   path.write_bytes(b'II*\x00\x08\x00\x00\x00')
   runner._scan_results();runner._scan_results();self.assertEqual(found,[])
   tifffile.imwrite(path,np.arange(64,dtype=np.float32).reshape(8,8))
   complete=path.read_bytes();path.write_bytes(complete[:-20])
   runner._scan_results();runner._scan_results();self.assertEqual(found,[])
   path.write_bytes(complete)
   runner._scan_results();runner._scan_results();self.assertEqual(found,[str(path)])
 def test_reset_does_not_restore_stale_background_preview(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'image.tiff';tifffile.imwrite(path,np.ones((64,64),np.float32))
   view=ImageView();view.show_tiff(path);view.reset('New job')
   self.wait(lambda:not view._loading)
   self.assertIsNone(view._pixmap);self.assertEqual(view.text(),'New job');view.close()
 def test_latest_selected_preview_wins(self):
  with tempfile.TemporaryDirectory() as tmp:
   paths=[]
   for i in range(3):
    p=Path(tmp)/f'{i}.tiff';tifffile.imwrite(p,np.arange(64*64,dtype=np.float32).reshape(64,64)+i);paths.append(p)
   view=ImageView()
   for p in paths:view.show_tiff(p)
   self.wait(lambda:not view._loading and view._pending_preview is None)
   self.assertEqual(view.toolTip(),str(paths[-1]));self.assertIsNotNone(view._pixmap);view.close()
