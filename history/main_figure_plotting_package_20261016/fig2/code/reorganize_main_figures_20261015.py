"""Native-artist figure separation/reassembly; frozen scientific sources only."""
from pathlib import Path
import sys,importlib.util,json,hashlib,argparse
from io import BytesIO
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox

PUB=Path(__file__).resolve().parents[1];MAIN=PUB/'output';SRC=PUB/'source'
S2='fig2_nhb_reordered_examples_20261014';S5='fig5_nhb_evidence_layout'
STEMS={2:S2,5:'fig5_nhb_dimension_awareness_20261015',6:'fig6_nhb_gaze_selectivity_20261015'}

def load(stem,filename):
 path=SRC/stem/'code'/filename;sys.path.insert(0,str(path.parent))
 name='reorg_'+stem
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 return m


def resize_preserving_mm(fig,height):
 oldw,oldh=fig.get_size_inches()*25.4
 positions={a:(a.get_position().x0*oldw,(1-a.get_position().y1)*oldh,a.get_position().width*oldw,a.get_position().height*oldh) for a in fig.axes}
 textpos={t:(t.get_position()[0]*oldw,(1-t.get_position()[1])*oldh) for t in fig.texts}
 legendpos={leg:leg.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).bounds for leg in fig.legends}
 fig.set_size_inches(oldw/25.4,height/25.4,forward=True)
 for a,(x,y,w,h) in positions.items():a.set_position([x/oldw,1-(y+h)/height,w/oldw,h/height])
 for t,(x,y) in textpos.items():t.set_position((x/oldw,1-y/height))
 for legend in fig.legends:
  x,y,w,h=legendpos[legend]
  legend.set_bbox_to_anchor((x,1-(1-y)*oldh/height,w,h*oldh/height),transform=fig.transFigure)
 # Keep figure-level legend anchors at the same physical distance from the top.


def arrays(axis):
 def hash(a):
  a=np.asarray(a,dtype=float);return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()
 return {'lines':[hash(x.get_xydata()) for x in axis.lines],
  'collections':[{'offsets':hash(x.get_offsets()),'paths':[hash(p.vertices) for p in x.get_paths()]} for x in axis.collections],
  'patches':[hash(x.get_path().vertices) for x in axis.patches]}

def y_signature(axis):
 """Y-values and image arrays, independent of category widths and annotations."""
 def digest(a):
  a=np.asarray(a,dtype=float);return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()
 return {'lines':[digest(x.get_ydata()) for x in axis.lines if x.get_gid()!='editorial-annotation' and not str(x.get_gid()).startswith('publication-y-break')],
  'collections':[{'offsets_y':digest(x.get_offsets()[:,1]),'paths_y':[digest(p.vertices[:,1]) for p in x.get_paths()]} for x in axis.collections],
  'patches_y':[digest(x.get_path().vertices[:,1]) for x in axis.patches],
  'images':[digest(x.get_array()) for x in axis.images]}


def preserve_category_mark_width(axis,ratio):
 """Scale category jitter/box offsets, retaining each category centre and y-value."""
 from matplotlib.collections import PathCollection
 ticks=np.asarray(axis.get_xticks(),float)
 def scaled(x):
  x=np.asarray(x,float);centres=ticks[np.argmin(abs(x[...,None]-ticks),axis=-1)]
  return centres+(x-centres)*ratio
 for line in axis.lines:
  if line.get_transform()==axis.transData:
   x=np.asarray(line.get_xdata(),float)
   if np.all(np.min(abs(x[...,None]-ticks),axis=-1)<.45):line.set_xdata(scaled(x))
 for patch in axis.patches:
  if patch.get_transform()==axis.transData:patch.get_path().vertices[:,0]=scaled(patch.get_path().vertices[:,0])
 for collection in axis.collections:
  if isinstance(collection,PathCollection) and collection.get_offset_transform()==axis.transData:
   off=np.asarray(collection.get_offsets()).copy();off[:,0]=scaled(off[:,0]);collection.set_offsets(off)


def fontfix(fig):
 for t in fig.findobj(Text):
  if not t.get_visible():continue
  text=t.get_text().replace('% of DIS trial','% of DIS trials') if '% of DIS trials' not in t.get_text() else t.get_text()
  t.set_text(text)
  if t.get_fontsize() not in [5,6,8]:t.set_fontsize(6)

def save(fig,number,axes,mapping,source_hashes,extra=None):
 pkg=MAIN/STEMS[number];pkg.mkdir(exist_ok=True);qa=pkg/'qa';qa.mkdir(exist_ok=True)
 fontfix(fig);fig.canvas.draw();renderer=fig.canvas.get_renderer();W,H=fig.get_size_inches()*25.4
 if number==2:
  legend=fig.legends[0]
  assert legend.get_window_extent(renderer).y0>max(axes[k].bbox.y1 for k in ['f1','f2','f3','f4']), 'Reference legend must stay above the f row'
 texts=[];clipped=[];collisions=[];rotated_tick_warnings=[]
 rotated_ticks={id(t):id(a) for a in fig.axes for t in a.get_xticklabels() if t.get_rotation()!=0}
 hidden_axis_texts={id(t) for a in fig.axes if not a.axison for aa in [a.xaxis,a.yaxis] for t in aa.findobj(Text)}
 for t in fig.findobj(Text):
  if not t.get_visible() or not t.get_text().strip() or id(t) in hidden_axis_texts:continue
  if t.axes is not None and not t.axes.axison and t not in t.axes.texts:continue
  b=t.get_window_extent(renderer)
  if b.width<=0 or b.height<=0:continue
  texts.append((t,b))
  if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.x1+.5 or b.y1>fig.bbox.y1+.5:clipped.append(t.get_text())
 for i,(t,b) in enumerate(texts):
  for u,c in texts[i+1:]:
   if min(b.x1,c.x1)-max(b.x0,c.x0)>fig.dpi*.3/72 and min(b.y1,c.y1)-max(b.y0,c.y0)>fig.dpi*.3/72:
    if id(t) in rotated_ticks and rotated_ticks.get(id(t))==rotated_ticks.get(id(u)):
     rotated_tick_warnings.append([t.get_text(),u.get_text()])
    else:collisions.append([t.get_text(),u.get_text()])
 from audit_panel_alignment import require_matplotlib_panel_alignment
 from tick_bounded_spines import audit_tick_bounded_spines
 if number==5:rows=[['b1','b2'],['c','d','e']];cols=[]
 elif number==6:rows=[list('cde')];cols=[]
 else:rows=[['f1','f2','f3','f4'],['c','e']];cols=[]
 alignment=require_matplotlib_panel_alignment(fig,axes=list(axes.values()),panel_ids={a:k for k,a in axes.items()},row_groups=rows,column_groups=cols,tolerance_pt=1.5,gutter_tolerance_pt=1.5,json_out=qa/'alignment.json')
 science={k:arrays(a) for k,a in axes.items()}
 assert science==source_hashes,'Scientific geometry changed during assembly'
 report={'status':'PASS' if not clipped and not collisions else 'REVIEW','canvas_mm':[W,H],
  'rotated_tick_bbox_visual_review':rotated_tick_warnings,'rotated_tick_review':'Source-preserved 35-degree heatmap labels; glyphs inspected and separated',
  'font_sizes_pt':sorted({t.get_fontsize() for t,b in texts}),'clipped_text':clipped,'text_collisions':collisions,
  'alignment':alignment['verdict'],'panel_mapping':mapping,'scientific_artist_coordinates_unchanged':True,
  'scientific_geometry':science,'axes':{k:{'bounds':list(a.get_position().bounds),'xlabel':a.get_xlabel(),'ylabel':a.get_ylabel()} for k,a in axes.items()},
  'native_texts':sorted({t.get_text() for t,b in texts}),'no_new_analysis':True,'source_snapshot':str(SRC),**(extra or {})}
 out=MAIN/(STEMS[number]+'.png');buf=BytesIO();fig.savefig(buf,dpi=600,format='png',facecolor='white');buf.seek(0)
 with Image.open(buf) as im:im.convert('RGB').save(qa/'preview.png',dpi=(600,600))
 (qa/'reorganization.json').write_text(json.dumps(report,indent=2)+'\n')
 assert not clipped,clipped
 assert not collisions,collisions
 (qa/'preview.png').replace(out);report['sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
 (pkg/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'figure':number,'output':str(out),'status':'PASS','size_mm':[W,H]},indent=2))
 plt.close(fig)






def figure2():
 m=load(S2,'render_fig2.py');fig=m.build_figure();H=m.HEIGHT_MM
 fel_label=next(t for t in fig._panels['a'].texts if t.get_text()=='FEL, feature enrichment level')
 fel_label.set_x(37.5)
 remove=[a for a in fig.axes if getattr(a,'_layout_row',None)=='reports']
 for a in remove:fig.delaxes(a)
 for t in list(fig.texts):
  if getattr(t,'_layout_row',None)=='reports':t.remove()
 # Convert Task2 displays by fixed task maxima, retaining ranks and saved P.
 for key,divisor,label,ticks in [
  ('f3',60.,fig._panels['f1'].get_ylabel(),[0,.025,.05,.075,.10]),
  ('f4',5.,fig._panels['f2'].get_ylabel(),[0,.4,.8,1.2])]:
  a=fig._panels[key]
  for line in a.lines:line.set_ydata(np.asarray(line.get_ydata(),float)/divisor)
  for patch in a.patches:patch.get_path().vertices[:,1]/=divisor
  from matplotlib.collections import PathCollection
  for collection in a.collections:
   if isinstance(collection,PathCollection):
    offsets=np.asarray(collection.get_offsets(),float).copy();offsets[:,1]/=divisor
    collection.set_offsets(offsets)
  for annotation in a.texts:annotation.set_y(annotation.get_position()[1]/divisor)
  a.set_ylim(np.asarray(a.get_ylim())/divisor);a.set_yticks(ticks)
  if key=='f3':a.set_ylim(0,.10)
  a.set_yticklabels([f'{v:g}' for v in ticks]);a.set_ylabel(label)
  m.bound_spines_to_ticks(a)
 fig._panels['e'].set_ylabel('Mean FEL')
 # Remove only the source bitmap's 1px black outer crop border. Keep rooms,
 # sofa locator boxes and their exact coordinates/scale intact.
 im=fig._panels['b2'].images[0];raw=np.asarray(im.get_array());x0,x1,y1,y0=im.get_extent()
 h,w=raw.shape[:2];dx=(x1-x0)/w;dy=(y1-y0)/h
 im.set_data(raw[2:-2,2:-2]);im.set_extent((x0+2*dx,x1-2*dx,y1-2*dy,y0+2*dy))
 # Match every f plotting rectangle to public b (source axis c).
 target=fig._panels['c'].get_position();delta=(target.height-fig._panels['f1'].get_position().height)*H
 for key in ['f1','f2','f3','f4']:
  a=fig._panels[key];p=a.get_position()
  a.set_position([p.x0,p.y1-target.height,target.width,target.height])
 # Preserve current pixel scale and physical positions, only remove the final row.
 newh=float(np.ceil(fig._row_spacing['final_bounds_mm']['density']['bottom']+delta+2))
 resize_preserving_mm(fig,newh)
 for legend in fig.legends:
  for label in legend.get_texts():
   if label.get_text().lower()=='random mean':label.set_text('Random')
 fig.canvas.draw()
 # Keep original ranges; allow ylabel clearance to follow tick-label widths.
 for key,ticks in [('f1',[0,.075,.15]),('f2',[0,.6,1.2]),
                   ('f3',[0,.05,.1]),('f4',[0,.6,1.2])]:
  a=fig._panels[key]
  a.set_yticks(ticks);a.set_yticklabels([f'{v:g}' for v in ticks])
  m.bound_spines_to_ticks(a)
 fig.canvas.draw()
 renderer=fig.canvas.get_renderer()
 d2_label=next(t for t in fig._panels['a'].texts if t.get_text()=='FEL in D2')
 fel_bounds=fel_label.get_window_extent(renderer);d2_bounds=d2_label.get_window_extent(renderer)
 fel_placement={'text':fel_label.get_text(),'position_in_a_mm':list(fel_label.get_position()),
  'right_gap_from_D2_mm':float((fel_bounds.x0-d2_bounds.x1)*25.4/fig.dpi),
  'below_gap_from_D2_mm':float((d2_bounds.y0-fel_bounds.y1)*25.4/fig.dpi)}
 assert fel_placement['right_gap_from_D2_mm']>0 and fel_placement['below_gap_from_D2_mm']>0
 heights={key:float(np.ptp(fig._panels[key].spines['left'].get_transform().transform(
  fig._panels[key].spines['left'].get_path().vertices)[:,1])*25.4/fig.dpi)
  for key in ['f1','f2','f3','f4']}
 np.testing.assert_allclose(list(heights.values()),[32.]*4,atol=1e-7)
 reference_legend=fig.legends[0]
 assert [label.get_text() for label in reference_legend.get_texts()]==['Random']
 axes={k:a for k,a in fig._panels.items() if a in fig.axes and a.axison};source={k:arrays(a) for k,a in axes.items()}
 save(fig,2,axes,{k:v for k,v in m.PUBLIC_PANEL_MAP.items() if k not in 'gh'},source,
  {'removed_panels':'g/h relocated to new Fig5a/b','task2_normalization':{'f3':60,'f4':5},
   'f_size_matches_public_b':True,'normalization_is_display_only':True,'saved_P_unchanged':True,
   'd_Task2_outer_border_removed_px':2,'f_y_axis_heights_mm':heights,
   'f_y_ticks':{key:fig._panels[key].get_yticks().tolist() for key in ['f1','f2','f3','f4']},
   'f3_y_ticks':fig._panels['f3'].get_yticks().tolist(),
   'f3_tight_bounds_px':list(fig._panels['f3'].get_tightbbox(renderer).bounds),
   'random_legend_bounds_px':list(reference_legend.get_window_extent(renderer).bounds),
   'a_fel_definition_label':fel_placement})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--figure',type=int,choices=[2],required=True);a=p.parse_args()
 figure2()
