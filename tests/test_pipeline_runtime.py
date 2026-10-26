import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from analysis_worker import Pipeline,atomic_json
from ct_recon_gui.runtime import defaults
from sam3_rotation_center.adaptive import plan

class PipelineRuntimeTests(unittest.TestCase):
    def test_progress_replacement_with_concurrent_windows_reader(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'status.json';atomic_json(path,{'number':0,'values':[0]*1000})
            stop=threading.Event();errors=[]
            def reader():
                while not stop.is_set():
                    try:
                        with path.open() as handle:
                            data=json.load(handle);time.sleep(.002)
                        if any(v!=data['number'] for v in data['values']):errors.append('Partial JSON')
                    except PermissionError:
                        # Readers retry transient Windows delete-sharing locks,
                        # just as AnalysisRunner.poll does.
                        pass
                    except Exception as error:errors.append(str(error))
                    time.sleep(.001)
            thread=threading.Thread(target=reader);thread.start()
            try:
                for number in range(1,31):atomic_json(path,{'number':number,'values':[number]*1000})
            finally:stop.set();thread.join()
            self.assertEqual(errors,[]);self.assertEqual(json.loads(path.read_text())['number'],30)

    def test_default_planner_preserves_validated_decision(self):
        with tempfile.TemporaryDirectory() as temp:
            pipe=Pipeline({**defaults(),'output':temp});pipe.low=0.;pipe.high=20.
            rows=[dict(center=float(x),mean_threshold_count=10.,confidence_reward=float(20-abs(x-10))) for x in range(21)]
            expected=plan(rows,1,20,.25,3,'confidence_reward')
            self.assertEqual(pipe.decision(rows,1),expected)

    def test_custom_range_never_samples_outside_its_bounds(self):
        with tempfile.TemporaryDirectory() as temp:
            pipe=Pipeline({**defaults(),'output':temp});pipe.low=1000.1;pipe.high=1020.1
            samples=pipe.sample_grid([[1000.1,1020.1]],2)
            self.assertGreaterEqual(min(samples),pipe.low);self.assertLessEqual(max(samples),pipe.high)
            rows=[dict(center=c,mean_threshold_count=10.,confidence_reward=20-abs(c-1010.1)) for c in samples]
            decision=pipe.decision(rows,2)
            for peak in decision['peaks']:self.assertTrue(pipe.low<=peak['center']<=pipe.high)
            self.assertAlmostEqual(decision['filter_curve']['start'],pipe.low)

if __name__=='__main__':unittest.main()
