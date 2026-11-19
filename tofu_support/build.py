"""Build the isolated GPU plugin against a supplied UFO development prefix."""
from pathlib import Path
import argparse, os, shlex, subprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=Path(__file__).resolve().parent;prefix=a.prefix.resolve();output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'PKG_CONFIG_LIBDIR':str(prefix/'usr/lib/x86_64-linux-gnu/pkgconfig')+':/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/share/pkgconfig','PKG_CONFIG_SYSROOT_DIR':str(prefix),'LD_LIBRARY_PATH':str(prefix/'usr/lib/x86_64-linux-gnu')}
    flags=shlex.split(subprocess.check_output([str(prefix/'usr/bin/pkgconf'),'--cflags','--libs','ufo'],env=env,text=True))
    dynamic={'-lufo':'-l:libufo.so.1','-lglib-2.0':'-l:libglib-2.0.so.0','-lgobject-2.0':'-l:libgobject-2.0.so.0','-lgmodule-2.0':'-l:libgmodule-2.0.so.0','-lgio-2.0':'-l:libgio-2.0.so.0','-ljson-glib-1.0':'-l:libjson-glib-1.0.so.0'}
    flags=[dynamic.get(x,x) for x in flags]
    command=['gcc','-shared','-fPIC','-O3','-Wno-deprecated-declarations','-I'+str(source),'-I'+str(prefix/'usr/include'),str(source/'ufo-lamino-backproject-task.c'),str(source/'lamino-roi.c'),*flags,'-L'+str(prefix/'usr/lib/x86_64-linux-gnu'),'-lOpenCL','-lm','-o',str(output/'libufofilterlamino-backproject.so')]
    subprocess.run(command,env=env,check=True)
    import shutil
    shutil.copytree(source/'kernels',output/'kernels',dirs_exist_ok=True)

if __name__=='__main__':main()
