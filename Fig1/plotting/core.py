"""Native 600 dpi rendering helpers exported from the narrative figure workflow."""

from pathlib import Path
from io import BytesIO
import argparse
import gc
import gzip
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "data"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output" / ".mplconfig"))

import cloudpickle
import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .compose import compose

DPI = 600
PPM = DPI / 25.4
WIDTH = 4251
PAGE_WIDTH, PAGE_HEIGHT = 4960, 7015
QA = {"native_parts": [], "figures": {}, "row_crops": []}


def register_fonts():
    paths = sorted((PUB / "fonts").glob("*.TTF"))
    for path in paths:
        font_manager.fontManager.addfont(str(path))
    packaged = {str(path) for path in paths}
    font_manager.fontManager.ttflist = [
        entry for entry in font_manager.fontManager.ttflist
        if entry.name != "Arial" or entry.fname in packaged
    ]


def bind_explicit_arial_paths(figure):
    packaged_fonts = {path.name.lower(): path for path in (PUB / "fonts").glob("*.TTF")}
    for label in figure.findobj(Text):
        properties = label.get_fontproperties()
        original_path = properties.get_file()
        if original_path:
            local_path = packaged_fonts.get(Path(original_path).name.lower())
            if local_path is not None:
                properties.set_file(local_path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


NAMES = ('fig1_feedback_only_shifted_titles_20261104', 'fig2_F2S2_mean_sem_20261101', 'fig3_d_lightblue_larger_stars_20261104', 'fig4_nhb_convergent_transfer_compact_rows_20261027', 'fig5_probe_DIS_heatmaps_20261103', 'fig6_f_from_functional_backgrounds_20261103')


def arrsig(v):
 a=np.ma.asarray(v)
 return [list(a.shape),str(a.dtype),hashlib.sha256(np.asarray(a.data).tobytes()).hexdigest(),hashlib.sha256(np.ma.getmaskarray(a).tobytes()).hexdigest()]

def science(f):
 r=[]
 for a in f.axes:
  row={'lines':[[arrsig(l.get_xdata()),arrsig(l.get_ydata())] for l in a.lines],
       'collections':[], 'images':[[arrsig(i.get_array()),list(i.get_extent())] for i in a.images]}
  for c in a.collections:
   z={'offsets':arrsig(c.get_offsets()),'paths':[arrsig(p.vertices) for p in c.get_paths()]}
   if hasattr(c,'_offsets3d'):z['xyz']=[arrsig(x) for x in c._offsets3d]
   row['collections'].append(z)
  r.append(row)
 return hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()

def load(package,part):
 meta=json.loads((package/'rendering.json').read_text());row=meta['parts'][part]
 path=package/row['path'];assert sha(path)==row['sha256']
 with gzip.open(path,'rb') as h:v=cloudpickle.load(h)
 matplotlib.rcParams.update(v['rc'])
 f=v['figure'];FigureCanvasAgg(f);f._original_dpi=DPI;f.set_dpi(DPI)
 bind_explicit_arial_paths(f)
 return f,path

def style(f,letters=None):
 letters=letters or {}
 changed=[]
 for t in f.findobj(Text):
  text=t.get_text();old=t.get_fontsize()
  is_letter=len(text)==1 and text.islower() and text.isalpha() and (old>=8 or t.get_gid()=='nhb-panel-letter') and t.get_fontweight() in ['bold',700]
  t.set_fontfamily('Arial')
  if is_letter:
   t.set_text(letters.get(text,text));t.set_fontsize(10);t.set_gid('nhb-panel-letter')
  elif text and set(text)<=set('*'):
   t.set_fontsize(7);t.set_fontweight('bold')
  else:
   t.set_fontsize(6 if old<=6 else 7);t.set_fontweight('normal')
  if old!=t.get_fontsize():changed.append([text,old,t.get_fontsize()])
 return changed

def render(f):
 f.canvas.draw();b=BytesIO();f.savefig(b,format='png',dpi=DPI,facecolor='white');b.seek(0)
 return np.asarray(Image.open(b).convert('RGB')).copy()

def revise_part(f,number,part):
 """User-authorized narrative-only edits; the frozen source stays untouched."""
 removed=[]
 if number==6 and part in ('g1','g2'):
  axis=f.axes[5 if part=='g1' else 6]
  target='Task 1 (4D)' if part=='g1' else 'Task 2'
  for t in axis.get_children():
   if not isinstance(t,Text):continue
   if t.get_text()==target and t.get_visible():
    t.set_visible(False);removed.append(target)
  assert removed==[target],(part,removed)
 return removed

def native(package,part,letters=None,number=None):
 f,path=load(package,part);before=science(f);changes=style(f,letters)
 removed=revise_part(f,number,part);assert science(f)==before
 r=render(f)
 if number==1 and part=='top':
  QA['fig1_task_column_centers_px']=[float((axis.bbox.x0+axis.bbox.x1)/2)
                                    for axis in f.axes[:2]]
 visible=[t for t in f.findobj(Text) if t.get_visible() and (t.axes is None or t.axes.get_visible()) and t.get_text().strip()]
 sizes=sorted(set(t.get_fontsize() for t in visible));assert set(sizes)<={6,7,10}
 QA['native_parts'].append({'source':str(path.relative_to(PUB)),'sha256':sha(path),'science_hash':before,
 'scientific_arrays_unchanged':True,'font_sizes':sizes,'style_changes':changes,
 'removed_titles':removed})
 plt.close(f);del f;gc.collect()
 print('Rendered native',package.name,part,flush=True)
 return r

def current(number,letters=None,parts=None):
 p=PUB/'main_figure'/NAMES[number-1];meta=json.loads((p/'rendering.json').read_text())
 keys=parts or list(meta['parts']);arrays={k:native(p,k,letters,number) for k in keys}
 comp=[c for c in meta['composition'] if c['part'] in keys]
 return compose(arrays,comp,meta['shape_pixels'])

def bounds(arr):
 mask=np.any(arr<250,axis=2);ys,xs=np.nonzero(mask)
 return [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]

def row(arr,y0=0,y1=None,x0=354,x1=4605,name=''):
 y1=len(arr) if y1 is None else y1
 assert np.all(arr[y0]>249) and np.all(arr[y1-1]>249),('Cut crosses visible content',name)
 tile=arr[y0:y1,x0:x1]
 b=bounds(tile);pad=round(PPM*1.5)
 top=max(0,b[1]-pad);bottom=min(len(tile),b[3]+pad)
 # No horizontal crop of visible marks allowed.
 assert not np.any(arr[y0:y1,:x0]<250) and not np.any(arr[y0:y1,x1:]<250),(name,b)
 QA['row_crops'].append({'name':name,'source_crop':[x0,y0+top,x1,y0+bottom],'ink_bbox':b})
 return tile[top:bottom].copy()

def stack(rows,gap=4):
 margin=round(PPM*2);g=round(PPM*gap)
 h=sum(len(r) for r in rows)+g*(len(rows)-1)+2*margin
 out=np.full((h,WIDTH,3),255,np.uint8);y=margin;layout=[]
 for index,r in enumerate(rows):
  assert r.shape[1]==WIDTH
  out[y:y+len(r)]=r
  x0,y0,x1,y1=bounds(r)
  layout.append({'row':index+1,'tile_top_px':y,'tile_height_px':len(r),
                 'ink_bbox_px':[x0,y+y0,x1,y+y1]})
  y+=len(r)+g
 for index,item in enumerate(layout[:-1]):
  band=[item['ink_bbox_px'][3],layout[index+1]['ink_bbox_px'][1]]
  assert np.all(out[band[0]:band[1]]>249)
  item['white_band_to_next_mm']=(band[1]-band[0])/PPM
  assert item['white_band_to_next_mm']>=3
 QA.setdefault('stack_layouts',[]).append(layout)
 return out

def trim_native_row(arr,name):
 """Remove only top/bottom white space, preserving every original native pixel."""
 assert arr.shape[1]==WIDTH
 _,y0,_,y1=bounds(arr);pad=round(PPM*1.5)
 top=max(0,y0-pad);bottom=min(len(arr),y1+pad)
 assert np.all(arr[:top]>249) and np.all(arr[bottom:]>249)
 QA['row_crops'].append({'name':name,'source_crop':[0,top,WIDTH,bottom],
                         'ink_bbox':bounds(arr)})
 return arr[top:bottom].copy()

def on_page(arr):
 """Place native 180 mm artwork on the reference A4 canvas, without resizing."""
 assert arr.shape[1]==WIDTH and len(arr)<=PAGE_HEIGHT
 x=(PAGE_WIDTH-WIDTH)//2;y=(PAGE_HEIGHT-len(arr))//2
 page=np.full((PAGE_HEIGHT,PAGE_WIDTH,3),255,np.uint8)
 page[y:y+len(arr),x:x+WIDTH]=arr
 np.testing.assert_array_equal(page[y:y+len(arr),x:x+WIDTH],arr)
 return page,[x,y]

def task_learning_row(task_centers):
 package=PUB/'main_figure'/NAMES[0];f,path=load(package,'bottom')
 before=science(f)
 selected_axes=(0,2)
 for index,axis in enumerate(f.axes):
  axis.set_visible(index in selected_axes)
 for label in f.texts:
  label.set_visible(label.get_text() in ('c','e'))
 changes=style(f,{'e':'d'})
 assert [label.get_text() for label in f.texts if label.get_visible()]==['c','d']
 assert science(f)==before
 axes=[f.axes[index] for index in selected_axes]
 from .audit_panel_alignment import require_matplotlib_panel_alignment
 alignment=require_matplotlib_panel_alignment(f,axes=axes,panel_ids=['c','d'],row_groups=[['c','d']])
 source=render(f)
 assert source.shape[1]==WIDTH
 split=int((axes[0].bbox.x1+axes[1].bbox.x0)/2)
 assert np.all(source[:,split-1:split+1]==255)
 arranged=np.full_like(source,255)
 panels={}
 target_right_previous=0
 for label,source_letter,axis,center,(left,right) in zip('cd','ce',axes,task_centers,((0,split),(split,WIDTH))):
  occupied=np.flatnonzero(np.any(source[:,left:right]<255,axis=(0,2)))
  source_left,source_right=left+int(occupied[0]),left+int(occupied[-1])+1
  axis_center=(axis.bbox.x0+axis.bbox.x1)/2
  shift=round(center-axis_center)
  target_left,target_right=source_left+shift,source_right+shift
  assert target_right_previous<=target_left<target_right<=WIDTH
  arranged[:,target_left:target_right]=source[:,source_left:source_right]
  target_right_previous=target_right
  panels[label]={'source_panel':source_letter,
                 'cohorts':[legend_label.get_text() for legend_label in axis.get_legend().get_texts()],
                 'target_task_center_px':center,'plot_center_px':float(axis_center+shift),
                 'horizontal_translation_px':shift,'target_ink_x_px':[target_left,target_right]}
 QA['native_parts'].append({'source':str(path.relative_to(PUB)),'sha256':sha(path),
  'science_hash':before,'scientific_arrays_unchanged':True,
  'font_sizes':[6,7,10],'style_changes':changes,'selected_axes':list(selected_axes),
  'panels':panels,'source_plot_alignment':alignment,
  'native_row_no_scaling':True})
 plt.close(f);del f;gc.collect()
 return trim_native_row(arranged,'original_fig1_ce')

def main(number, make_rows, gap):
    parser = argparse.ArgumentParser(description=f"Reproduce narrative Fig. {number} at 600 dpi.")
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument("--verify", action="store_true", help="Compare the rendered pixels with the original figure.")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["figure"] == number
    expected = manifest["reference_png"]
    output = args.output or ROOT / "output" / expected["file"]

    for relative_path, expected_sha in manifest["input_sha256"].items():
        path = ROOT / relative_path
        if sha(path) != expected_sha:
            raise SystemExit(f"Frozen plotting input changed: {path}")

    register_fonts()
    content = stack(make_rows(), gap=gap)
    canvas, offset = on_page(content)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output, dpi=(DPI, DPI))
    file_hash = sha(output)
    with Image.open(output) as rendered:
        pixel_hash = hashlib.sha256(rendered.tobytes()).hexdigest()
        mode = rendered.mode
        size = list(rendered.size)
        dpi = list(rendered.info["dpi"])
    pixels_match = (
        pixel_hash == expected["rgb_sha256"]
        and mode == expected["mode"]
        and size == expected["size_px"]
        and all(abs(value - expected["dpi"]) < .01 for value in dpi)
    )
    file_matches = file_hash == expected["file_sha256"]
    audit = {
        "figure": number,
        "file": output.name,
        "file_sha256": file_hash,
        "rgb_sha256": pixel_hash,
        "mode": mode,
        "size_px": size,
        "dpi": dpi,
        "pixels_match_reference": pixels_match,
        "file_matches_reference": file_matches,
        "content_offset_xy_px": offset,
        "native_components": QA["native_parts"],
    }
    output.with_suffix(".render.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output, flush=True)
    if args.verify and not pixels_match:
        raise SystemExit(f"Rendered pixels differ from the reference figure (SHA-256 {pixel_hash}).")
    if args.verify:
        print(f"Fig. {number}: pixel match. PNG file hash match: {file_matches}.", flush=True)
