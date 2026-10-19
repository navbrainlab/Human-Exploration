#!/usr/bin/env python3
"""Relocate and independently rebuild all seven figures; compare PNG pixels."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parent

def verify_one(n):
    import numpy as np
    from PIL import Image
    start=time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f'mini_dg_independent_fig{n}_') as temp:
        relocated=Path(temp)/f'fig{n}'
        shutil.copytree(ROOT/f'fig{n}',relocated,ignore=shutil.ignore_patterns('output','__pycache__','.pytest_cache','*.pyc'))
        env=os.environ.copy()
        env['MPLBACKEND']='Agg'
        env['MPLCONFIGDIR']=str(Path(temp)/'matplotlib_cache')
        env['OPENBLAS_NUM_THREADS']='1'
        env['OMP_NUM_THREADS']='1'
        env.pop('PYTHONPATH',None)
        result=subprocess.run([sys.executable,'-B',str(ROOT/'_audit_entry.py'),str(relocated)],cwd=temp,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
        (ROOT/'qa'/f'fig{n}_relocated.log').write_text(result.stdout)
        record={'figure':n,'exit_code':result.returncode,'seconds':round(time.monotonic()-start,3),'relocated_outside_project':True,'command':f'python fig{n}/run.py'}
        if result.returncode:
            record['status']='FAIL'
            record['log']=f'qa/fig{n}_relocated.log'
            return record
        actual=relocated/f'output/fig{n}.png'
        reference=ROOT/f'fig{n}/reference/fig{n}.png'
        with Image.open(actual) as a,Image.open(reference) as b:
            aa,bb=np.asarray(a.convert('RGB')),np.asarray(b.convert('RGB'))
            same_shape=aa.shape==bb.shape
            changed=int(np.count_nonzero(np.any(aa!=bb,axis=2))) if same_shape else None
            record.update(pixels=list(a.size),mode=a.mode,dpi=list(a.info.get('dpi',())),changed_pixels=changed,pixel_identical=same_shape and changed==0)
        record['file_identical']=actual.read_bytes()==reference.read_bytes()
        record['output_sha256']=hashlib.sha256(actual.read_bytes()).hexdigest()
        record['reference_sha256']=hashlib.sha256(reference.read_bytes()).hexdigest()
        audit=json.loads((relocated/'independent_read_audit.json').read_text())
        record['read_guard']=audit
        record['status']='PASS' if record['pixel_identical'] and record['mode']=='RGB' and all(abs(d-600)<.01 for d in record['dpi']) else 'FAIL'
        shutil.copy2(actual,ROOT/f'fig{n}/output/fig{n}.png')
        print(f"Fig. {n}: {record['status']}, {changed} different pixels",flush=True)
        return record

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',nargs='+',type=int,choices=range(1,8),default=list(range(1,8)))
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args()
    (ROOT/'qa').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        reports=list(pool.map(verify_one,args.figures))
    original=json.loads((ROOT/'original_png_hashes.json').read_text())
    protection={path:Path(path).is_file() and hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in original.items()}
    # Original source files may be absent after delivery; independent redraw remains valid.
    available=any(Path(p).exists() for p in original)
    report={'python':sys.version,'platform':platform.platform(),'figures':reports,'original_png_checks':protection,'original_sources_available':available,'all_original_pngs_unchanged':all(protection.values()) if available else None,'status':'PASS' if all(r['status']=='PASS' for r in reports) and (not available or all(protection.values())) else 'FAIL'}
    (ROOT/'qa/independent_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'figures':len(reports),'original_pngs_unchanged':report['all_original_pngs_unchanged']},indent=2))
    if report['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
