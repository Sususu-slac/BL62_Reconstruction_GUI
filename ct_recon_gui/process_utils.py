"""Terminate only the selected worker and its descendants, never other jobs."""
import subprocess
import psutil
import os
import sys
from contextlib import contextmanager

def clean_environment():
    env=dict(os.environ)
    for key in list(env):
        if key in ('PYTHONPATH','PYTHONHOME','CUDA_VISIBLE_DEVICES','_MEIPASS2') or key.startswith('_PYI_'):env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    bundle=getattr(sys,'_MEIPASS',None)
    if bundle:
        env['PATH']=os.pathsep.join(p for p in env.get('PATH','').split(os.pathsep) if not os.path.normcase(p).startswith(os.path.normcase(bundle)))
    return env

@contextmanager
def external_runtime():
    """Do not let a frozen GUI's NumPy/Qt DLL directory leak into Conda."""
    frozen=os.name=='nt' and getattr(sys,'frozen',False)
    if frozen:
        import ctypes
        ctypes.windll.kernel32.SetDllDirectoryW(None)
    try:yield
    finally:
        if frozen:ctypes.windll.kernel32.SetDllDirectoryW(sys._MEIPASS)

def terminate_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for process in reversed(children):
            try: process.terminate()
            except psutil.Error: pass
        try: parent.terminate()
        except psutil.Error: pass
        _, alive = psutil.wait_procs(children + [parent], timeout=2)
        for process in alive:
            try: process.kill()
            except psutil.Error: pass
    except psutil.NoSuchProcess:
        pass

def subprocess_flags() -> dict:
    import os
    return {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
