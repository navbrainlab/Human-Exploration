#!/usr/bin/env python3
"""Run selected standalone main-figure plotting packages."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--figures',nargs='+',type=int,choices=range(1,8),default=list(range(1,8)))
args=parser.parse_args()
root=Path(__file__).resolve().parent
for n in args.figures:
    folder=root/f'fig{n}'
    print(f'Rendering Fig. {n}',flush=True)
    env=os.environ.copy()
    env['MPLBACKEND']='Agg'
    env['OPENBLAS_NUM_THREADS']='1'
    env['OMP_NUM_THREADS']='1'
    subprocess.run([sys.executable,'-B',str(folder/'run.py')],cwd=folder,env=env,check=True)
