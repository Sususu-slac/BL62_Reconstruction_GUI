"""Export the exact tested upstream inference source for a reproducible build."""
import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

COMMIT='4bf4a8aaff2ab34789731810eed45e6a61a73cfd'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,default=Path(__file__).parent/'runtime_sam3');a=parser.parse_args()
    if (a.destination/'sam3/model_builder.py').is_file():print('SAM3 source already prepared:',a.destination);return
    if a.destination.exists():raise FileExistsError('Destination exists but is incomplete; choose a new directory')
    with tempfile.TemporaryDirectory() as temp:
        archive=Path(temp)/'source.zip'
        request=urllib.request.Request(f'https://codeload.github.com/facebookresearch/sam3/zip/{COMMIT}',headers={'User-Agent':'BL62-Reconstruction-Builder'})
        with urllib.request.urlopen(request,timeout=120) as src,archive.open('wb') as dst:shutil.copyfileobj(src,dst)
        prefix=f'sam3-{COMMIT}/'
        with zipfile.ZipFile(archive) as zip:
            for name in zip.namelist():
                if not name.startswith(prefix):continue
                relative=Path(name[len(prefix):])
                if not relative.parts or '..' in relative.parts:continue
                if relative.parts[0]!='sam3' and relative.as_posix() not in ('README.md','pyproject.toml','LICENSE'):continue
                target=a.destination/relative
                if name.endswith('/'):target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with zip.open(name) as source,target.open('wb') as output:shutil.copyfileobj(source,output)
    print('Prepared SAM3',COMMIT,'at',a.destination)

if __name__=='__main__':main()
