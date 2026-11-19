import hashlib,io,subprocess,sys,tarfile,tempfile,unittest
from pathlib import Path

class TofuPackageTests(unittest.TestCase):
    def test_relocated_pax_names_do_not_escape_the_python_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'base.tar.gz';packed=root/'python.tar';out=root/'runtime.tar.gz'
            with tarfile.open(base,'w:gz') as t:
                item=tarfile.TarInfo('etc/os-release');data=b'ID=ubuntu\n';item.size=len(data);t.addfile(item,io.BytesIO(data))
            name='lib/python3.12/site-packages/'+('long_directory_name/'*8)+'module.py'
            with tarfile.open(packed,'w',format=tarfile.PAX_FORMAT) as t:
                item=tarfile.TarInfo(name);item.pax_headers={'path':name};item.size=4;t.addfile(item,io.BytesIO(b'pass'))
            for n in ['ufo','plugin']:(root/n).mkdir()
            command=[sys.executable,str(Path(__file__).resolve().parents[1]/'tofu_support/package_runtime.py'),'--base',str(base),'--base-sha256',hashlib.sha256(base.read_bytes()).hexdigest(),'--python-tar',str(packed),'--ufo-prefix',str(root/'ufo'),'--plugin',str(root/'plugin'),'--output',str(out)]
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
            with tarfile.open(out,'r:gz') as t:
                self.assertEqual(t.extractfile('opt/bl62-tofu/python/'+name).read(),b'pass')
                self.assertFalse(any(m.name.startswith('lib/python') for m in t))
            self.assertTrue(out.with_name(out.name+'.sha256').is_file())

if __name__=='__main__':unittest.main()
