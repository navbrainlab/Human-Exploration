"""Render using only the portable package; compare decoded RGB pixels."""
from pathlib import Path
import hashlib, json, os, sys, runpy, tempfile
ROOT=Path(__file__).resolve().parent
PROJECT=Path('/home/Scarbu/Projects/mini_DG').resolve()
sys.dont_write_bytecode=True
_MPLCONFIG=tempfile.TemporaryDirectory(prefix='dgem-figure-mpl-')
os.environ['MPLCONFIGDIR']=_MPLCONFIG.name
os.environ['MPLBACKEND']='Agg'
# Fail on accidental dependencies in the original project. Python libraries,
# interpreter files and this package are allowed; figures never read reference/.
def protect_project(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=Path(os.fsdecode(args[0])).resolve()
    if p.is_relative_to(PROJECT) and not p.is_relative_to(ROOT):
        raise PermissionError(f'External project access forbidden: {p}')
sys.addaudithook(protect_project)
from matplotlib import font_manager
# Prefer the bundled fonts rather than another system DejaVu installation.
font_manager.fontManager.ttflist=[f for f in font_manager.fontManager.ttflist if f.name!='DejaVu Sans']
for p in sorted((ROOT/'assets/fonts').glob('*.ttf')):font_manager.fontManager.addfont(p)
font_manager.fontManager._findfont_cached.cache_clear()
output=ROOT/'output';output.mkdir(exist_ok=True)
FIGURE='fig4'
target=output/(FIGURE+'.png')
if target.exists():target.unlink()
sys.path.insert(0,str(ROOT/'code'))
sys.argv=[str(ROOT/'code'/('render_'+FIGURE+'.py')),'--output-dir',str(output)]
runpy.run_path(str(ROOT/'code'/('render_'+FIGURE+'.py')),run_name='__main__')
from PIL import Image
import numpy as np
with Image.open(target) as im:
    actual=np.asarray(im.convert('RGB'));info={'mode':im.mode,'pixels':list(im.size),'dpi':list(im.info.get('dpi',()))}
with Image.open(ROOT/'reference'/(FIGURE+'.png')) as im:expected=np.asarray(im.convert('RGB'))
identity=actual.shape==expected.shape and bool(np.array_equal(actual,expected))
report={'status':'PASS' if identity else 'FAIL','figure':FIGURE,'rendered':str(target.relative_to(ROOT)),
    'reference':str(Path('reference')/(FIGURE+'.png')),'decoded_rgb_pixels_identical':identity,
    'png_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
    'reference_sha256':hashlib.sha256((ROOT/'reference'/(FIGURE+'.png')).read_bytes()).hexdigest(),
    'png':info,'external_project_access':'Denied by Python audit hook; rendering succeeded without it',
    'statistical_analysis_rerun':False,'fitting_or_simulation_rerun':False}
if actual.shape==expected.shape:report['different_pixel_count']=int(np.any(actual!=expected,axis=2).sum())
(ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if not identity:raise SystemExit('Rebuilt image differs from active reference; see verification.json')
