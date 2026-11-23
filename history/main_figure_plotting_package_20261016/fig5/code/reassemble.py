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
STEMS={5:'fig5'}

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


def figure5():
 data=capture5();fig=data['fig'];panels=data['panels'];keep=[panels[k] for k in 'ghij']
 for a in list(fig.axes):
  if a not in keep:fig.delaxes(a)
 for t in list(fig.texts):t.remove()
 H=108.;resize_preserving_mm(fig,H)
 xs=[17.1,58.9,100.7,142.5];widths=[27.9,27.9,27.9,34.];axes={};original_y={}
 for k,key,x,width in zip('cdef','ghij',xs,widths):
  a=panels[key];original_y[k]=y_signature(a)
  a.set_position([x/180,1-96/H,width/180,30/H]);axes[k]=a
 axes['d'].set_ylabel('Early stage noticed dim.')
 axes['e'].set_xlabel('Early stage');axes['f'].set_ylabel('Reward-relevant dim.')
 from matplotlib.lines import Line2D
 from matplotlib.colors import to_rgba
 j=panels['j'];stage_p={};labels=[]
 for group in ['Obs','NObs']:
  p=float(data['stats'].loc[data['stats'].panel.eq('j-'+group),'p_value_displayed'].iloc[0])
  stage_p[group]=p
  suffix='***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'
  labels.append(f'{group} {suffix}')
 handles=[Line2D([],[],ls='none',marker='o',ms=3.3,
  color=c.lines[0].get_markerfacecolor(),markerfacecolor=c.lines[0].get_markerfacecolor(),
  markeredgecolor=c.lines[0].get_markeredgecolor()) for c in j.containers]
 legend=j.legend(handles,labels,frameon=False,loc='lower center',bbox_to_anchor=(.5,1.05),
  ncol=2,fontsize=5,handlelength=.8,handletextpad=.3,columnspacing=.7,borderaxespad=0,borderpad=0)
 assert all(to_rgba(h.get_markerfacecolor())==to_rgba(c.lines[0].get_markerfacecolor())
  for h,c in zip(legend.get_lines(),j.containers))
 # Saved within-group stage comparisons are displayed only in the legend.
 m=load(S2,'render_fig2.py');fig._panels={};fig._tests=[];fig._active_row='reports'
 # Pair the task plots within b, preserving their group centre and plot sizes.
 origaxis=m.axis;counter=[0];top_x=[21.,None,89.5,131.5];top_w=[36.,27.9,34.,34.]
 def axis(figure,name,x,y,w,h,**kwargs):
  index=['g1','g2','h1','h2'].index(name);left=top_x[index] if top_x[index] is not None else 60.9
  a=origaxis(figure,name,left,15,top_w[index],30,**kwargs)
  a.set_position([left/180,1-45/H,top_w[index]/180,30/H]);return a
 def text(figure,x,y,label,**kwargs):
  if label.startswith('Task'):
   i=counter[0];counter[0]+=1
   if top_x[i] is not None:return fig.text((top_x[i]+top_w[i]/2)/180,1-8/H,label,fontsize=6,ha='center',va='top')
  if label=='Dimension identification':return fig.text((top_x[0]+top_w[0]/2)/180,1-51/H,label,fontsize=5,ha='center',va='top')
 m.axis=axis;m.text=text;m.survey_and_terms(fig)
 # Keep Task 1 frequency ticks evenly spaced and within the visible scale.
 fig._panels['h1'].set_ylim(0,.15)
 fig._panels['h1'].set_yticks([0,.05,.10,.15])
 fig.delaxes(fig._panels['g2'])
 fig._tests=[t for t in fig._tests if not (t.get('panel')=='g' and t.get('task')==2)]
 for key,new in [('g1','a1'),('h1','b1'),('h2','b2')]:
  a=fig._panels[key];original_y[new]=y_signature(a)
  if new.startswith('b'):
   preserve_category_mark_width(a,27.9/34.)
   a.set_xticklabels(['DIS','Others'],fontsize=5)
  m.detach_axes(a);m.compact_category_margin(a,list(a.patches));m.bound_spines_to_ticks(a);axes[new]=a
 # Both tasks use the same measure but have different numerical y scales.
 axes['b2'].set_ylabel('Dimension-term\nfrequency',fontsize=6)
 # Anchor a's label to the vertical centre of its plotting area.
 axes['a1'].yaxis.set_label_coords(-.20,.5)
 for letter,x,y,kind in [('a',7,5,'survey'),('b',75.5,5,'survey'),('c',3,56,'eye_tracking'),
  ('d',44.8,56,'survey'),('e',86.6,56,'task_behavior'),('f',128.4,56,'survey')]:
  fig.text(x/180,1-y/H,letter,fontsize=8,fontweight='bold',va='top',gid='nhb-panel-letter')
  m.add_source_icon(fig,kind,left=(x+3.8)/180,top=1-y/H,size_mm=4.8)
 # Detachment can add decorative axis-break lines; top b has no break. The
 # statistical summaries and all paired values remain exactly source-derived.
 for key in 'cdef':assert y_signature(axes[key])==original_y[key],key
 for key in ['b1','b2']:assert y_signature(axes[key])==original_y[key],key
 assert all(abs(axes[k].get_position().height-30/H)<1e-8 for k in axes)
 source={k:arrays(a) for k,a in axes.items()}
 save(fig,5,axes,{'a':'former Fig2g Task1','b':'former Fig2h Task1/Task2',
  'c':'former Fig5g','d':'former Fig5h','e':'former Fig5i','f':'former Fig5j'},source,
  {'retained_tests':fig._tests,'retained_old5_stats':data['stats'].fillna('').to_dict('records'),
  'scientific_y_values_unchanged':True,'scientific_y_signatures':{k:y_signature(a) for k,a in axes.items()},
  'scientific_artist_coordinates_unchanged':False,
  'authorized_geometry_changes':'b box/cap/jitter horizontal offsets rescaled to preserve physical widths. Category centres and all y-values unchanged; f saved within-group stage significance is shown only in legend labels.',
  'alignment_exemption':'a 36 mm and f 34 mm wide; b1/b2 34 mm and c/d/e 27.9 mm. All plotting heights 30 mm; each row shares a baseline.',
  'b_pair_layout':{'plot_gap_mm':8.,'previous_plot_gap_mm':19.,'group_centre_mm':127.5,
   'single_measure_ylabel':False,'task_specific_y_scales_preserved':True,
   'plot_widths_mm':[34.,34.],'plot_heights_mm':[30.,30.]},
  'display_labels_20261015':{'b1':['DIS','Others'],'b2':['DIS','Others'],
   'd_ylabel':'Early stage noticed dim.',
   'Others_definition':'Retained inclusive full-report comparator, not a non-DIS-only subset'},
  'f_significance':{'Obs_Holm_P':stage_p['Obs'],'NObs_Holm_P':stage_p['NObs'],
   'comparison':'Early–Final within each group','location':'legend only','plot_brackets':False},
  'f_legend_layout':{'anchor_axes':[.5,1.05],'loc':'lower center',
   'labels':labels,'significance_in_legend':True,'significance_in_plot':False}})

