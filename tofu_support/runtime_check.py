"""Offline self-check in the clean, account-local WSL runtime."""
from pathlib import Path
import os,subprocess,sys

def main():
    base=Path('/opt/bl62-tofu');ufo=base/'ufo';python=base/'python'
    if '--child' not in sys.argv:
        env=dict(os.environ)
        env.update(PYTHONPATH=str(ufo/'usr/lib/python3/dist-packages'),
                   GI_TYPELIB_PATH=str(ufo/'usr/lib/x86_64-linux-gnu/girepository-1.0'),
                   LD_LIBRARY_PATH=':'.join(map(str,[python/'lib',ufo/'usr/lib/x86_64-linux-gnu',ufo/'usr/lib/x86_64-linux-gnu/blas',ufo/'usr/lib/x86_64-linux-gnu/lapack',Path('/usr/lib/wsl/lib')])),
                   UFO_PLUGIN_PATH=str(base/'plugin')+':'+str(ufo/'usr/lib/x86_64-linux-gnu/ufo'),
                   UFO_KERNEL_PATH=str(base/'plugin/kernels')+':'+str(ufo/'usr/share/ufo'),
                   OCL_ICD_VENDORS=str(python/'etc/OpenCL/vendors'),POCL_DEVICES='cuda',POCL_CUDA_TOOLKIT_PATH=str(python),
                   CUDA_VISIBLE_DEVICES='0',UFO_DEVICE_TYPE='gpu',UFO_DEVICES='0')
        return subprocess.call([sys.executable,__file__,'--child'],env=env)
    import gi
    gi.require_version('Ufo','1.0')
    from gi.repository import Ufo
    scheduler=Ufo.FixedScheduler()
    if not scheduler.get_resources().get_gpu_nodes():raise RuntimeError('No OpenCL GPU is available')
    if not (base/'plugin/libufofilterlamino-backproject.so').is_file():raise RuntimeError('GPU compatibility plugin is missing')
    Ufo.PluginManager().get_task('lamino-backproject')
    print('BL62 Tofu GPU runtime ready',flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
