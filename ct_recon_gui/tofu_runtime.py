"""Machine-local runtime settings; no personal Linux paths in the application."""
from __future__ import annotations
import json, os
from pathlib import Path


def configuration_path() -> Path:
    if os.environ.get('BL62_TOFU_RUNTIME'):
        return Path(os.environ['BL62_TOFU_RUNTIME'])
    local = Path(os.environ.get('APPDATA', Path.home()/'.config'))/'BL62_Reconstruction/tofu_runtime.json'
    if local.is_file():
        return local
    return Path(os.environ.get('PROGRAMDATA', 'C:/ProgramData'))/'BL62/Tofu/runtime.json'


def configuration() -> dict:
    path = configuration_path()
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as error:
        raise RuntimeError(f'Tofu GPU runtime is not configured. Run Setup Tofu GPU first. Runtime configuration: {path}') from error
    required = ('python', 'ufo_prefix', 'plugin_root', 'cuda_prefix')
    if not isinstance(value, dict) or any(not isinstance(value.get(k), str) or not value[k] for k in required):
        raise RuntimeError(f'Incomplete Tofu runtime configuration: {path}')
    if os.name == 'nt' and not value.get('distribution'):
        raise RuntimeError('The Tofu runtime configuration must specify a WSL distribution')
    return value
