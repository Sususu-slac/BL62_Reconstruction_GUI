"""Process-local preparation bypass; installed TomoCuPy sources stay untouched."""
import os
from tomocupy.processing.proc_functions import ProcFunctions

state=os.environ.get('BL62_INPUT_STATE','raw')
if state in ('normalized','attenuation'):
    def identity(self,data,dark,flat):return data
    ProcFunctions.darkflat_correction=identity

if __name__=='__main__':
    # Reuse the existing inclusive trial-grid adapter; it leaves full runs intact.
    if os.environ.get('BL62_INCLUSIVE_CENTER_SEARCH')=='1':
        from tomocupy_manual_entry import main
    else:
        from tomocupy.__main__ import main
    raise SystemExit(main())
