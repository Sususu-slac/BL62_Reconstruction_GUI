import sys,unittest
from types import SimpleNamespace
from unittest.mock import patch,Mock
from pathlib import Path
import numpy as np
from ct_recon_gui.models import ReconJob,ReconMethod,GeometryMode
from ct_recon_gui.commands import build_wsl_argv
from backend_runner import parse_args,remove_astra_stripes
class StripeDefaultsTests(unittest.TestCase):
 def test_every_manual_method_passes_vo_all_to_external_worker(self):
  for method in ReconMethod:
   mode=GeometryMode.LAMINOGRAPHY if method.value.startswith('lamino_') else GeometryMode.TOMOGRAPHY
   job=ReconJob(input_path=Path('input.h5'),output_dir=Path('output'),center=32,method=method,mode=mode,reconstruction_type='full')
   argv=build_wsl_argv(job);self.assertEqual(argv[argv.index('--remove-stripe-method')+1],'vo-all',method)
 def test_omitted_backend_switch_defaults_on_but_explicit_none_is_respected(self):
  args=['backend_runner.py','fourierrec','input.h5','output','--start-row','0','--end-row','4','--center','32']
  with patch.object(sys,'argv',args):self.assertEqual(parse_args().remove_stripe_method,'vo-all')
  with patch.object(sys,'argv',args+['--remove-stripe-method','none']):self.assertEqual(parse_args().remove_stripe_method,'none')
 def test_astra_preprocessing_uses_gpu_chunks_and_returns_corrected_transmission(self):
  device=Mock();device.__enter__=Mock();device.__exit__=Mock(return_value=False)
  cp=SimpleNamespace(cuda=SimpleNamespace(Device=Mock(return_value=device)),float32=np.float32,asarray=np.asarray,asnumpy=np.asarray)
  correction=Mock(side_effect=lambda a,**kwargs:a+.05)
  with patch.dict(sys.modules,{'cupy':cp,'tomocupy.processing.remove_stripe':SimpleNamespace(remove_all_stripe=correction)}):
   result=remove_astra_stripes(np.ones((7,3,64),np.float32),SimpleNamespace(remove_stripe_method='vo-all',gpu=0,chunk_size=2))
  self.assertEqual(correction.call_count,2);np.testing.assert_allclose(result,1.05);self.assertEqual(correction.call_args.kwargs,dict(snr=3,la_size=61,sm_size=21,dim=1))
if __name__=='__main__':unittest.main()
