import argparse
import sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.repo))
import numpy,h5py,scipy,matplotlib,torch
from sam3.model_builder import build_sam3_image_model
assert numpy.__version__=='1.26.4','SAM3 requires the pinned NumPy 1.26.4'
assert torch.cuda.is_available(),'CUDA is unavailable'
print('SAM3 Python:',sys.executable)
print('SAM3 repo:',a.repo)
print('Torch:',torch.__version__,'CUDA GPUs:',torch.cuda.device_count())
print('NumPy:',numpy.__version__,'h5py:',h5py.__version__,'SciPy:',scipy.__version__)
