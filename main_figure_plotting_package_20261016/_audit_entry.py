#!/usr/bin/env python3
"""Independent child-process read guard for relocated plotting verification."""
from pathlib import Path
import json
import os
import runpy
import sys

folder=Path(sys.argv[1]).resolve()
blocked=[Path('/home/Scarbu/Projects/mini_DG'),Path('/home/Scarbu/.codex'),Path('/home/Scarbu/.agents')]
reads=set()
denied=[]

def audit(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):
        return
    path=Path(os.fsdecode(args[0])).resolve()
    mode=args[1] if len(args)>1 else None
    flags=args[2] if len(args)>2 else 0
    writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT)))
    if not writing:
        if any(path.is_relative_to(p) for p in blocked) and not path.is_relative_to(folder):
            denied.append(str(path))
            raise PermissionError(f'Unbundled project/skill dependency: {path}')
        if path.is_relative_to(folder):
            reads.add(str(path.relative_to(folder)))

sys.addaudithook(audit)
sys.dont_write_bytecode=True
os.chdir(folder)
sys.argv=[str(folder/'run.py')]
runpy.run_path(str(folder/'run.py'),run_name='__main__')
(folder/'independent_read_audit.json').write_text(json.dumps({'original_project_and_codex_reads_denied':True,'attempted_external_reads':denied,'local_reads':sorted(reads)},indent=2)+'\n')
