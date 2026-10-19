from pathlib import Path
import sys,os,json,hashlib
ROOT=Path(__file__).resolve().parent
PROJECT=Path('/home/Scarbu/Projects/mini_DG')
READS=set()
def audit(event,args):
 if event!='open':return
 try:p=Path(os.fsdecode(args[0])).resolve()
 except (TypeError,ValueError):return
 mode=args[1];flags=args[2]
 writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT))
 if not writing:
  if p.is_relative_to(PROJECT) and not p.is_relative_to(ROOT):raise PermissionError('Original project reads denied: '+str(p))
  if p.is_relative_to(ROOT):READS.add(str(p.relative_to(ROOT)))
sys.addaudithook(audit)
sys.dont_write_bytecode=True
from matplotlib import font_manager
for font in sorted((ROOT/'assets/fonts').glob('*.ttf')):font_manager.fontManager.addfont(str(font))

from pathlib import Path
import runpy,sys,shutil
ROOT=Path(__file__).resolve().parent
(ROOT/'output').mkdir(exist_ok=True)
sys.argv=[str(ROOT/'code/reorganize_main_figures_20261015.py'),'--figure','2']
runpy.run_path(sys.argv[0],run_name='__main__')
shutil.copy2(ROOT/'output/fig2_nhb_reordered_examples_20261014.png',ROOT/'output/fig2.png')

from PIL import Image
import numpy as np
reference=ROOT/'reference/fig2.png';output=ROOT/'output/fig2.png'
with Image.open(reference) as a,Image.open(output) as b:
 A=np.asarray(a);B=np.asarray(b);assert A.shape==B.shape
 changed=int(np.count_nonzero(np.any(A!=B,axis=2)))
 assert changed==0,changed
 report={'status':'PASS','pixel_identical':True,'changed_pixels':changed,'dimensions':list(a.size),'mode':b.mode,'dpi':list(b.info.get('dpi',[])),'original_project_reads_denied':True,'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'package_reads':sorted(READS)}
(ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS: identical pixels; original project reads denied')
