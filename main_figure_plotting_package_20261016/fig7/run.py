#!/usr/bin/env python3
"""Render current Fig.7 from local frozen plotting inputs (no inference)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=None)
    args=parser.parse_args()
    package=Path(__file__).resolve().parent
    output=(args.output_dir or package/'output').resolve()
    output.mkdir(parents=True,exist_ok=True)
    sys.dont_write_bytecode=True
    os.environ['MPLBACKEND']='Agg'
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    with tempfile.TemporaryDirectory(prefix='mini_dg_fig7_mpl_') as config:
        os.environ['MPLCONFIGDIR']=config
        import matplotlib.font_manager as fm
        for font in sorted((package/'assets/fonts').glob('*.ttf')):
            fm.fontManager.addfont(str(font))
        sys.path.insert(0,str(package/'code'))
        os.chdir(package)
        entry=package/'code/render_fig6_full_joint90_free_beta.py'
        sys.argv=[str(entry),'--output-dir',str(output)]
        runpy.run_path(str(entry),run_name='__main__')
    from PIL import Image
    import numpy as np
    actual=output/'fig7.png'
    reference=package/'reference/fig7.png'
    with Image.open(actual) as image,Image.open(reference) as original:
        a,b=np.asarray(image.convert('RGB')),np.asarray(original.convert('RGB'))
        same_shape=a.shape==b.shape
        differing=int(np.count_nonzero(np.any(a!=b,axis=2))) if same_shape else None
        report={'figure':7,'output':str(actual),'reference':str(reference),
                'pixels':list(image.size),'mode':image.mode,'dpi':image.info.get('dpi'),
                'same_shape':same_shape,'different_pixels':differing,
                'pixel_identical':same_shape and differing==0,
                'file_identical':hashlib.sha256(actual.read_bytes()).hexdigest()==hashlib.sha256(reference.read_bytes()).hexdigest(),
                'scope':'Frozen plotting tables only; no model fitting, simulation or statistical tests.'}
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copy2(output/'verification.json',package/'verification.json')
    print(json.dumps(report,indent=2))
    if not report['pixel_identical']:
        raise SystemExit('Rendering differs from frozen reference; see output/verification.json')


if __name__=='__main__':
    main()
