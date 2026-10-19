#!/usr/bin/env python3
"""Rebuild from bundled frozen inputs. No original project access is needed."""
from pathlib import Path
import os,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
(ROOT/'output/qa').mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLBACKEND','Agg')
sys.path.insert(0,str(ROOT/'code'))
import reassemble
if __name__=='__main__':
    reassemble.figure6()
