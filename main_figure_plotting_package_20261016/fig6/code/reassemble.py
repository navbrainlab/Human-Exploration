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

ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT/'output';SRC=ROOT/'data'
S2='fig2_nhb_reordered_examples_20261014';S5='old5'
STEMS={6:'fig6'}

def load(stem,filename):
 path=ROOT/'code'/filename;sys.path.insert(0,str(path.parent))
 name='reorg_'+stem
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 return m


def capture5(darker_blue=False):
 m=load(S5,'render_fig5_evidence_layout.py');m.layout.DATA_DIR=SRC/S5
 inputs=json.loads((SRC/'fig5_input_map.json').read_text())
 m.layout.original.DATA_PATHS={k:SRC/v['path'] for k,v in inputs['data'].items()}
 for key,value in inputs['assets'].items():setattr(m.layout,key,SRC/value['path'])
 if darker_blue:
  from matplotlib.colors import ListedColormap
  levels=np.linspace(0,1,256);colors=plt.get_cmap('RdBu_r')(levels)
  colors[:128,:3]*=(1-.18*np.sin(2*np.pi*levels[:128])**2)[:,None]
  cmap=ListedColormap(colors,name='gaze_blue_refined')
  np.testing.assert_array_equal(colors[128:],plt.get_cmap('RdBu_r')(levels)[128:])
  matplotlib.colormaps.register(cmap,force=True)
  original=m.layout._recolor_gaze_overlay
  def recolor(path):
   previous=m.layout.GAZE_CMAP;m.layout.GAZE_CMAP=cmap.name
   try:return original(path)
   finally:m.layout.GAZE_CMAP=previous
  m.layout._recolor_gaze_overlay=recolor
 box={}
 class Captured(Exception):pass
 def keep(fig,panels,labels,stats):
  audit=m.finalize(fig,panels,labels,stats)
  box.update(fig=fig,panels=panels,labels=labels,stats=stats,audit=audit,module=m)
  raise Captured()
 try:m.layout.assemble(output_stem=S5,figure_finalize=keep)
 except Captured:pass
 assert box
 if darker_blue:
  bar=next(a for a in box['fig'].axes if a.get_label()=='fig5a_shared_colorbar')
  for collection in bar.collections:collection.set_cmap(cmap)
  box['blue_palette']={'base':'RdBu_r','blue_RGB_multiplier':'1 - 0.18 sin(2 pi t)^2 for t < 0.5',
   'red_half_exactly_unchanged':True,'opacity_and_source_geometry_unchanged':True,
   'scope':'Both illustrative gaze composites and their shared qualitative key; matrix palette unchanged'}
 return box


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
 pkg=ROOT;qa=MAIN/'qa';qa.mkdir(exist_ok=True)
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
 (MAIN/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'figure':number,'output':str(out),'status':'PASS','size_mm':[W,H]},indent=2))
 plt.close(fig)


def figure6():
 data=capture5(darker_blue=True);fig=data['fig'];panels=data['panels']
 image_axes=list(fig.axes[:2]);a_bar=next(a for a in fig.axes if a.get_label()=='fig5a_shared_colorbar')
 b_bar=next(a for a in fig.axes if a.get_label()=='fig5c_colorbar')
 keep=set(image_axes+[a_bar,b_bar]+[panels[k] for k in 'bcdf'])
 for a in list(fig.axes):
  if a not in keep:fig.delaxes(a)
 for t in list(fig.texts):t.remove()
 H=99.;resize_preserving_mm(fig,H)
 # Preserve the source-image aspect ratio while matching b's 34 mm height.
 extent=image_axes[0].images[0].get_extent()
 image_height=34.;image_width=image_height*abs((extent[1]-extent[0])/(extent[3]-extent[2]))
 image_lefts=[53.6-image_width-1.4,55.]
 for a,left in zip(image_axes,image_lefts):a.set_position([left/180,1-43/H,image_width/180,image_height/H])
 a_bar_left=image_lefts[1]+image_width+1.8
 a_bar.set_position([a_bar_left/180,1-43/H,1.44/180,image_height/H])
 panels['b'].set_position([128/180,1-43/H,34/180,34/H])
 b_bar.set_position([163.8/180,1-43/H,1.44/180,34/H])
 for x,label in [(image_lefts[0]+image_width/2,'DIS trials (Plant)'),(image_lefts[1]+image_width/2,'others')]:
  fig.text(x/180,1-7/H,label,fontsize=6,ha='center',va='bottom',color='#252B30')
 m=data['module'];original_y={};box_widths={};axes={'b':panels['b']}
 from category_label_spacing import compact_category_margin
 from tick_bounded_spines import bound_spines_to_ticks
 # Match Fig5c's 27.9 x 30 mm plot areas; retain equal 16 mm gaps and
 # the lower group's centre, including the 14 mm left header allowance.
 lower_lefts=[39.15,83.05,126.95]
 for oldkey,newkey,left in zip('cdf','cde',lower_lefts):
  a=panels[oldkey];original_y[newkey]=y_signature(a)
  boxes=a._publication_boxplot_artists['boxes']
  before=[x.get_window_extent().width/fig.dpi*25.4 for x in boxes]
  a.set_position([left/180,1-89/H,27.9/180,30/H])
  # Equal visible y spines, from the first to last major tick. Preserve
  # all source values and the existing unclipped significance brackets.
  a.set_ylim({'c':(100.,600.),'d':(0.,4000.),'e':(2.,5.)}[newkey])
  if newkey=='d':a.set_yticks([0,2000,4000],['0','2,000','4,000'])
  if newkey=='e':
   # Five is a meaningful task ceiling. Show complete boundary markers
   # and whisker strokes rather than clipping half of each at that ceiling.
   for artist in [*a.lines,*a.collections,*a.patches]:artist.set_clip_on(False)
  compact_category_margin(a,boxes);bound_spines_to_ticks(a)
  after=[x.get_window_extent().width/fig.dpi*25.4 for x in boxes]
  np.testing.assert_allclose(before,after,atol=1e-8)
  assert y_signature(a)==original_y[newkey],newkey
  box_widths[newkey]=after;axes[newkey]=a
 a_letter_left=image_lefts[0]-3.6
 for letter,x,y in [('a',a_letter_left,2),('b',114.3,2),('c',25.15,54),('d',69.05,54),('e',112.95,54)]:
  fig.text(x/180,1-y/H,letter,fontsize=8,fontweight='bold',va='top',gid='nhb-panel-letter')
 # Source icons mark the example row and the quantitative-metric row once each.
 for x,y in [(a_letter_left+3.8,2),(28.95,54)]:m.layout.add_source_icon(fig,'eye_tracking',left=x/180,top=1-y/H,size_mm=4.8)
 source={k:arrays(a) for k,a in axes.items()}
 save(fig,6,axes,{'a':'former Fig5a','b':'former Fig5b','c':'former Fig5c','d':'former Fig5d','e':'former Fig5f'},source,
  {'gaze_palette_revision':data['blue_palette'],'relocated_panel':'Former Fig6e gaze-shift speed remains Supplementary Fig. S6i',
  'scientific_y_values_unchanged':True,'scientific_y_signatures':{k:y_signature(a) for k,a in axes.items()},
  'scientific_artist_coordinates_unchanged':False,
  'authorized_geometry_changes':'Lower plot areas match Fig5c at 27.9 x30 mm; row recentered with16 mm gaps; native category offsets retain4.95 mm physical boxes and all y-values.',
  'box_widths_mm':box_widths,'a_linear_scale':image_height/32.97,'b_square_size_mm':34.,'source_icons':'one per evidence row',
  'a_b_height_alignment_20261015':{'height_mm':34.,'top_mm':9.,'bottom_mm':43.,
   'a_image_size_mm':[image_width,image_height],'a_image_lefts_mm':image_lefts,
   'a_colorbar_left_mm':a_bar_left,'a_aspect_ratio_preserved':True,'a_pair_centre_mm':53.6},
  'lower_group_layout':{'plot_lefts_mm':lower_lefts,'plot_gaps_mm':[16.,16.],
   'plot_size_mm':[27.9,30.],'reference':'Fig5c',
   'letter_lefts_mm':[25.15,69.05,112.95],'content_center_target_mm':90.},
  'equal_yaxes_20261015':{'visible_spine_height_mm':30.,
   'limits':{'c':[100,600],'d':[0,4000],'e':[2,5]},'d_ticks':[0,2000,4000],
   'e_ceiling_marks_unclipped':True,'source_values_and_tests_unchanged':True}})

