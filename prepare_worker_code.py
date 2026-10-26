"""Put external-interpreter source away from the frozen GUI's extension DLLs."""
import shutil
from pathlib import Path
root=Path(__file__).resolve().parent;destination=root/'worker_code'
if destination.exists():shutil.rmtree(destination)
destination.mkdir()
for name in ('backend_runner.py','analysis_worker.py','reconstruct_candidates.py'):
    shutil.copy2(root/name,destination/name)
for name in ('ct_recon_gui','sam3_rotation_center'):
    shutil.copytree(root/name,destination/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
