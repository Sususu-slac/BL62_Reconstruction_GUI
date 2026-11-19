import subprocess,sys,tempfile,time,unittest
from pathlib import Path
from tofu_prepare import run_child,Cancelled

class TofuStopTests(unittest.TestCase):
    def test_unresponsive_cancellation_helper_cannot_hold_the_native_worker_open(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);(output/'.stop_requested').touch()
            sleeping=[sys.executable,'-c','import time; time.sleep(5)']
            unrelated=subprocess.Popen(sleeping)
            started=time.monotonic()
            try:
                with self.assertRaises(Cancelled):
                    run_child(sleeping,output,sleeping,stop_grace=.05,cancel_timeout=.1)
                self.assertLess(time.monotonic()-started,2)
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.terminate();unrelated.wait(timeout=5)

if __name__=='__main__':unittest.main()
