"""Discover isolated runtimes without storing another computer's user name."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

APP_ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
WORKER_ROOT = APP_ROOT/'worker_code' if getattr(sys,'frozen',False) else APP_ROOT
VERSION = '1.0.1'

def conda_python(name: str) -> str:
    roots = [Path(sys.prefix).parent.parent]
    home = Path.home()
    roots += [Path(os.environ.get('LOCALAPPDATA', home / 'AppData/Local')) / 'miniforge3',
              home / 'miniforge3', home / 'miniconda3', home / 'anaconda3']
    for root in roots:
        path = root / 'envs' / name / ('python.exe' if os.name == 'nt' else 'bin/python')
        if path.is_file():
            return str(path)
    return ''

def reconstruction_python() -> str:
    if not getattr(sys, 'frozen', False):
        return str(Path(sys.executable).with_name('python.exe')) if os.name == 'nt' else sys.executable
    return conda_python('tomocupy_astra')

def settings_path() -> Path:
    root = Path(os.environ.get('APPDATA', Path.home() / '.config')) / 'BL62_Reconstruction'
    return root / 'settings.json'

def defaults() -> dict:
    repo=next((p for p in [Path('D:/SAM3p1/sam3'),APP_ROOT/'sam3_repo',APP_ROOT/'runtime_sam3'] if (p/'sam3').is_dir()),APP_ROOT/'runtime_sam3')
    return dict(sam3_python=conda_python('SAM3p1'), recon_python=reconstruction_python(),
                repo=str(repo), checkpoint='D:/Chcekpoints/checkpoint_demo_05142026.pt',
                gpu=1, method='fourierrec', binning=0, chunk_size=4, iterations=20,
                output_root='', start_row=None, end_row=None, search_row=None,
                range_low=None, range_high=None, initial_step=None, target_step=.25,
                search_method='fourierrec', confidence=.2, nms_iou=.5, refresh_seconds=3)

def load_settings() -> dict:
    result = defaults()
    try:
        result.update({k:v for k,v in json.loads(settings_path().read_text()).items() if k in result})
    except (OSError, ValueError, TypeError):
        pass
    return result

def save_settings(value: dict) -> None:
    path = settings_path(); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2), encoding='utf-8'); tmp.replace(path)
