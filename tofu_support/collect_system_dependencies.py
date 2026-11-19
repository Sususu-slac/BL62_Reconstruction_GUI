"""Collect the public OS-library closure omitted by an already-provisioned host."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,tarfile

def main():
    p=argparse.ArgumentParser();p.add_argument('--ufo',type=Path,required=True);p.add_argument('--python',type=Path,required=True);p.add_argument('--plugin',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    lib=a.ufo/'usr/lib/x86_64-linux-gnu';env={**os.environ,'LD_LIBRARY_PATH':str(a.python/'lib')+':'+str(lib)+':'+str(lib/'blas')+':'+str(lib/'lapack')+':/usr/lib/wsl/lib'}
    files=[lib/'libufo.so.1',a.plugin/'libufofilterlamino-backproject.so',a.python/'lib/libpocl.so.2',a.python/'lib/pocl/libpocl-devices-cuda.so',*a.ufo.glob('usr/lib/python3/dist-packages/gi/_gi*.so')]
    files += [lib/'ufo'/('libufofilter'+n+'.so') for n in ['read','write','slice']]
    skip={'libc.so.6','libm.so.6','libpthread.so.0','librt.so.1','libdl.so.2','libresolv.so.2','ld-linux-x86-64.so.2'}
    collected={}
    for file in files:
        if not file.exists():raise FileNotFoundError(file)
        text=subprocess.check_output(['ldd',str(file)],env=env,text=True)
        for name,path in re.findall(r'^\s*(\S+) => (/\S+)',text,re.M):
            source=Path(path)
            if name in skip or '/usr/lib/wsl/' in str(source) or str(source).startswith(str(a.python)) or str(source).startswith(str(a.ufo)):continue
            target=lib/name
            if not target.is_file():
                target.unlink(missing_ok=True);shutil.copy2(source,target)
                collected[str(target.relative_to(a.ufo))]=dict(source=str(source),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    for name in ['GLib','GObject','Gio','GModule']:
        source=Path('/usr/lib/x86_64-linux-gnu/girepository-1.0')/(name+'-2.0.typelib')
        if source.is_file():
            target=lib/'girepository-1.0'/source.name
            if not target.exists():shutil.copy2(source,target);collected[str(target.relative_to(a.ufo))]=dict(source=str(source))
    (a.ufo/'system_closure.json').write_text(json.dumps(collected,indent=2))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(a.output,'w') as archive:
        for relative in collected:archive.add(a.ufo/relative,arcname=relative)
        archive.add(a.ufo/'system_closure.json',arcname='system_closure.json')
    print(json.dumps(dict(files=len(collected),archive=str(a.output))),flush=True)

if __name__=='__main__':main()
