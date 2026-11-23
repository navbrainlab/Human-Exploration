#!/usr/bin/env python3
"""Rebuild with original project inputs denied and compare pixel arrays."""
from pathlib import Path
import runpy,sys,json,hashlib
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
PROJECT=Path('/home/Scarbu/Projects/mini_DG')
DENIED=[]
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=Path(args[0]).resolve()
        if PROJECT in path.parents and ROOT not in path.parents and path!=ROOT:
            DENIED.append(str(path))
            raise PermissionError('Original project access denied: '+str(path))
sys.dont_write_bytecode=True
sys.addaudithook(guard)
runpy.run_path(str(ROOT/'run.py'),run_name='__main__')
ref=ROOT/'reference/fig5.png';out=ROOT/'output/fig5.png'
with Image.open(ref) as a,Image.open(out) as b:
    av=np.asarray(a.convert('RGB'));bv=np.asarray(b.convert('RGB'))
    same_shape=av.shape==bv.shape
    pixels_equal=same_shape and np.array_equal(av,bv)
    record={'status':'PASS' if pixels_equal and not DENIED else 'FAIL',
        'command':'python verify.py','original_project_access_denied':True,
        'denied_access_attempts':DENIED,'pixel_identical':pixels_equal,
        'differing_pixels':int(np.count_nonzero(np.any(av!=bv,axis=2))) if same_shape else None,
        'shape':list(bv.shape),'mode':b.mode,'dpi':list(b.info.get('dpi',[])),
        'reference_sha256':hashlib.sha256(ref.read_bytes()).hexdigest(),
        'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'scope':'Exact frozen-data rendering only; no fitting, simulation or new statistical tests.'}
(ROOT/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
assert record['status']=='PASS'
