import tempfile,unittest
from pathlib import Path
import numpy as np,tifffile
from shiboken6 import delete,isValid
from ct_recon_gui.image_view import TiffPreview

class PreviewShutdownTests(unittest.TestCase):
    def test_decode_can_finish_after_qt_disposes_its_signal_object(self):
        with tempfile.TemporaryDirectory() as temp:
            image=Path(temp)/'frame.tiff';tifffile.imwrite(image,np.arange(64,dtype=np.float32).reshape(8,8))
            task=TiffPreview(str(image),1);delete(task.signals)
            self.assertFalse(isValid(task.signals))
            task.run()  # Reproduces the shutdown race without timing-dependent threads.

if __name__=='__main__':unittest.main()
