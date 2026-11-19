"""Build a clean WSL runtime image from an official base and dependency prefixes.

Never export a developer's personal WSL distribution or home directory.
The supplied Conda tar must be packed for /opt/bl62-tofu/python.
"""
from pathlib import Path
import argparse, gzip, hashlib, io, json, tarfile

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--base-sha256',required=True)
    p.add_argument('--python-tar',type=Path,required=True);p.add_argument('--ufo-prefix',type=Path,required=True)
    p.add_argument('--plugin',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert hashlib.sha256(a.base.read_bytes()).hexdigest()==a.base_sha256
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def owned(item):
        if '__pycache__' in Path(item.name).parts:return None
        item.uid=item.gid=0;item.uname=item.gname='root';return item
    def write_file(archive,name,text):
        data=text.encode();item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644
        archive.addfile(item,io.BytesIO(data))
    with a.output.open('wb') as file,gzip.GzipFile(fileobj=file,mode='wb',compresslevel=1,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w|') as out:
        with tarfile.open(a.base,'r:gz') as base:
            for item in base:
                if item.name.lstrip('./')=='etc/wsl.conf':continue
                out.addfile(item,base.extractfile(item) if item.isfile() else None)
        with tarfile.open(a.python_tar,'r:') as packed:
            for item in packed:
                item.name='opt/bl62-tofu/python/'+item.name.lstrip('./')
                if item.islnk():item.linkname='opt/bl62-tofu/python/'+item.linkname.lstrip('./')
                item=owned(item)
                if item is not None:item.pax_headers={}
                if item is not None:out.addfile(item,packed.extractfile(item) if item.isfile() else None)
        out.add(a.ufo_prefix,arcname='opt/bl62-tofu/ufo',filter=owned)
        out.add(a.plugin,arcname='opt/bl62-tofu/plugin',filter=owned)
        out.add(Path(__file__).resolve().parent,arcname='opt/bl62-tofu/source',filter=owned)
        out.add(Path(__file__).with_name('runtime_check.py'),arcname='opt/bl62-tofu/check_runtime.py',filter=owned)
        write_file(out,'etc/wsl.conf','[boot]\nsystemd=false\n[automount]\nenabled=true\nmountFsTab=false\n[interop]\nappendWindowsPath=false\n[user]\ndefault=root\n')
        write_file(out,'opt/bl62-tofu/README.txt','BL62 Tofu GPU runtime 1.0.0. Dependencies and independent UFO buffer compatibility plugin only. No developer home, credentials, datasets, GUI conversations or checkpoints are included. GPU work is selected explicitly by the Windows job.\n')
    digest=hashlib.sha256(a.output.read_bytes()).hexdigest();a.output.with_name(a.output.name+'.sha256').write_text(digest+'  '+a.output.name+'\n')
    print(json.dumps(dict(file=str(a.output),bytes=a.output.stat().st_size,sha256=digest)),flush=True)

if __name__=='__main__':main()
