"""Native 600-dpi Fig2 with documented PPT statistical-ink imports."""
from __future__ import annotations
from axis_break_marks import mark_nonzero_yaxis, audit_yaxis_breaks
from detached_axes import detach_axes, audit_detached_axes
from tick_bounded_spines import bound_spines_to_ticks, audit_tick_bounded_spines
from category_label_spacing import compact_category_margin, category_gaps
from matplotlib.ticker import FuncFormatter
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle, FancyBboxPatch, PathPatch
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from PIL import Image

from draw_data_source_icons import add_source_icon
from audit_panel_alignment import require_matplotlib_panel_alignment

PACKAGE=Path(__file__).resolve().parents[1]
STEM=PACKAGE.name
WIDTH_MM=180
HEIGHT_MM=170
SOURCE_TO_PUBLIC={'a':'a','b1':'d1','b2':'d2','c':'b','d':'e','e':'c',**{f'{letter}{task}':f'{letter}{task}' for letter in 'fgh' for task in range(1,5) if letter=='f' or task<=2}}
PUBLIC_PANEL_MAP={'a':'unchanged FEL space','b':'source c: highest-FEL human-minus-random frequency','c':'source e: mean FEL learning progress','d':'source b: illustrative DIS choices in both tasks','e':'source d: persistence/coverage schematic','f':'unchanged source f endpoints','g':'unchanged source g identification comparisons','h':'unchanged source h paired language comparisons'}
QUANTITATIVE_AXES=('c','e','f1','f2','f3','f4','g1','g2','h1','h2')
CATEGORICAL_BOX_AXES=('c','f1','f2','f3','f4','g1','g2','h1','h2')
DPI=600
# Fig5c: 4.95-mm boxes in a 27.9-mm plot area; scale by each axis span.
BOX_AXES_WIDTH_RATIO=4.95/27.9
FEL_ORIGIN_MM=np.array([37.5,28.])
FEL_VECTORS_MM=np.array([[0.,-19.],[-16.5,16.],[22.,2.]])
INK='#20242B'
GRAY='#8D9399'
TASK1='#0965C0'
TASK1_BOX_ALPHA=.58
# Approved Fig1c phase palette; only the two Task1 views inside panel f use it.
TASK1_PHASE_COLORS={'3D-NE':'#079E6C','4D-E':'#0965C0','4D-NE':'#14BAEC'}
TASK2='#D57224'
PURPLE='#60458F'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial','Helvetica'],
 'pdf.fonttype':42,'svg.fonttype':'none','font.size':6,
 'axes.labelsize':6,'axes.titlesize':6,'xtick.labelsize':5,'ytick.labelsize':5,
 'legend.fontsize':5,'axes.linewidth':.70,'xtick.major.width':.70,'ytick.major.width':.70,
 'xtick.major.size':2,'ytick.major.size':2,'text.color':INK,'axes.labelcolor':INK,
 'xtick.color':INK,'ytick.color':INK,'axes.grid':False,'savefig.dpi':DPI})


def read(name):return pd.read_csv(PACKAGE/'csv'/name)


def text(fig,x,y,label,*,size=6,weight='normal',color=INK,ha='left'):
    t=fig.text(x/WIDTH_MM,1-y/HEIGHT_MM,label,fontsize=size,fontweight=weight,
                    ha=ha,va='top',color=color)
    t._layout_row=fig._active_row
    return t


def axis(fig,name,x,y,w,h,*,schematic=False):
    a=fig.add_axes((x/WIDTH_MM,1-(y+h)/HEIGHT_MM,w/WIDTH_MM,h/HEIGHT_MM),label=name)
    fig._panels[name]=a
    a._layout_row=fig._active_row
    if schematic:
        a.set(xlim=(0,w),ylim=(h,0));a.set_axis_off()
    else:
        a.spines[['top','right']].set_visible(False);a.tick_params(pad=2)
        a.xaxis.labelpad=3;a.yaxis.labelpad=3
    return a


def heading(fig,letter,x,y,icon=None):
    t=text(fig,x,y,letter,size=8,weight='bold');t.set_gid('nhb-panel-letter')
    if icon:
        ia=add_source_icon(fig,icon,left=(x+5)/180,top=1-y/HEIGHT_MM,size_mm=4.8)
        ia._layout_row=fig._active_row
        ia._source_panel=letter
    return t


def arrow(a,start,end,*,color=GRAY,lw=.8):
    a.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=6,
                              color=color,lw=lw,shrinkA=0,shrinkB=0))


def pictogram(a,name,bounds,crop=None):
    path=PACKAGE/name if str(name).startswith('assets/') else PACKAGE/'assets/ppt_media'/name
    im=Image.open(path)
    if crop:im=im.crop(crop)
    x,y,w,h=bounds
    if not hasattr(a,'_pictogram_rects'):a._pictogram_rects=[]
    a._pictogram_rects.append(dict(asset=name,box_mm=list(bounds),crop=list(crop) if crop else None))
    a.imshow(im,extent=(x,x+w,y+h,y),interpolation='lanczos',aspect='auto',zorder=2)


def source_shapes():
    return json.loads((PACKAGE/'assets/ppt_schematic_shapes.json').read_text())['records']


def fel_projection(data):
    """Recover raw FEL exactly for balanced Task 1's 3-by-3 count tables."""
    energy=data[['E1','E2','E3']].to_numpy(dtype=float)
    lookup={0:3,4:5,6:6,10:7,18:9}
    numerators=np.rint(3*energy)
    if not (np.isfinite(energy).all() and
            np.allclose(3*energy,numerators,rtol=0,atol=1e-9) and
            np.isin(numerators,list(lookup)).all()):
        raise ValueError('FEL recovery requires balanced Task 1 energy classes')
    fel=np.array([[lookup[int(value)] for value in row] for row in numerators])
    return fel,FEL_ORIGIN_MM+(fel-3)/6@FEL_VECTORS_MM


def endpoint_example(a,dimension,label,bounds,crop,old_bounds,marker_radius):
    """Link an illustrative arrangement to its unchanged FEL endpoint."""
    endpoint=FEL_ORIGIN_MM+FEL_VECTORS_MM[dimension-1]
    x,y,w,h=bounds
    frame_pad=.25
    color='#929CA5'
    for i,target in enumerate(((x-frame_pad,y-frame_pad),(x-frame_pad,y+h+frame_pad))):
        vector=np.array(target)-endpoint
        # Start outside the data marker and DIS ring, not across their ink.
        start=endpoint+vector/np.linalg.norm(vector)*(marker_radius+.3)
        a.plot([start[0],target[0]],[start[1],target[1]],color=color,lw=.65,
               zorder=1,gid=f'endpoint-zoom-D{dimension}-{i}')
    pictogram(a,'shape_46.png',bounds,crop=crop)
    if label=='Dessert':
        # Reconstruct the erroneous illustrative cell from approved feature
        # assets: one beverage, one dessert and one staple. Original source
        # raster is retained verbatim; only this native display cell changes.
        sx,sy=w/(crop[2]-crop[0]),h/(crop[3]-crop[1])
        cx,cy=x+27*sx,y+151*sy
        cw,ch=122*sx,118*sy
        a.add_patch(FancyBboxPatch((cx,cy),cw,ch,
            boxstyle=f'round,pad=0,rounding_size={17*sx}',
            fc='white',ec='#D9E0F0',lw=.5,zorder=3))
        icons=[('food3_beverage_3.png',16,12,35,43,True),
               ('food3_dessert_3.png',72,12,40,44,False),
               ('food3_staple_1.png',36,64,52,43,True)]
        for filename,ix,iy,iw,ih,muted in icons:
            with Image.open(PACKAGE/'assets/food3'/filename) as source:
                icon=source.convert('RGBA');icon=icon.crop(icon.getchannel('A').getbbox())
                # Native schematic uses grey non-attended feature icons.
                if muted:
                    rgba=np.asarray(icon).copy()
                    grey=np.dot(rgba[:,:,:3],[.299,.587,.114])
                    rgba[:,:,:3]=(.25*grey[:,:,None]+.75*255).astype('uint8')
                    icon=Image.fromarray(rgba)
                iw0,ih0=icon.size;scale=min(iw/iw0,ih/ih0)
                ww,hh=iw0*scale*sx,ih0*scale*sy
                left,top=cx+ix*sx+(iw*sx-ww)/2,cy+iy*sy+(ih*sy-hh)/2
                a.imshow(icon,extent=(left,left+ww,top+hh,top),interpolation='lanczos',aspect='auto',zorder=4)
        a._dessert_correction=dict(row=2,column=1,dessert_count=1,
            features=['milk','cupcake','pizza'],native_assets=[item[0] for item in icons],
            source_raster_unchanged=True)
    a.add_patch(Rectangle((x-frame_pad,y-frame_pad),w+2*frame_pad,h+2*frame_pad,
                         fc='none',ec=color,lw=.65,zorder=2))
    a.text(x+w/2,y-2.4,label,fontsize=5,ha='center',va='top')
    distance=lambda b:float(np.linalg.norm(endpoint-np.clip(endpoint,b[:2],np.array(b[:2])+b[2:])))
    a._endpoint_callouts.append(dict(dimension=dimension,label=label,
        endpoint_mm=endpoint.tolist(),box_mm=list(bounds),line_count=2,
        old_gap_mm=distance(old_bounds),new_gap_mm=distance(bounds)))


def non_dis_example(a,clusters,fel,xy,sizes):
    spec=json.loads((PACKAGE/'assets/non_dis_example.json').read_text())
    rows=np.asarray(spec['rows'],dtype=int)
    observed=pd.read_csv(PACKAGE/spec['source_behavior'])
    trial=observed.loc[observed.phase.eq(spec['phase']) &
                       observed.subject.eq(spec['subject']) & observed['round'].eq(spec['round'])]
    assert len(trial)==1
    actual=np.asarray([[tuple(map(int,str(value).zfill(3)))
                        for value in ast.literal_eval(trial.iloc[0][block])]
                       for block in ('block1','block2','block3')])
    np.testing.assert_array_equal(rows,actual)
    recovered=np.asarray([sum(np.bincount(row[:,d],minlength=4)[1:].max()
                             for row in rows) for d in range(3)])
    np.testing.assert_array_equal(recovered,spec['fel'])
    assert recovered.max()<9 and int(trial.iloc[0].if_FDS)==0
    indices=np.flatnonzero(clusters.energy_cell_id.eq(spec['cluster_id']))
    assert len(indices)==1
    index=indices[0]
    np.testing.assert_array_equal(fel[index],recovered)
    point=xy[index]
    x,y,w,h=1.4,29.2,15.,14.4
    pad=.25
    color='#929CA5'
    radius=np.sqrt(sizes[index])/2*25.4/72+.15*25.4/72
    for i,target in enumerate(((x+w+pad,y-pad),(x+w+pad,y+h+pad))):
        vector=np.asarray(target)-point
        start=point+vector/np.linalg.norm(vector)*(radius+.3)
        a.plot([start[0],target[0]],[start[1],target[1]],color=color,lw=.65,
               zorder=1,gid=f'non-dis-zoom-{i}')
    a.add_patch(Rectangle((x-pad,y-pad),w+2*pad,h+2*pad,
                         fc='white',ec=color,lw=.65,zorder=2))
    a._pictogram_rects.append(dict(asset='native non-DIS arrangement',box_mm=[x,y,w,h]))
    icon_hashes={}
    for r,row in enumerate(rows):
        for c,stimulus in enumerate(row):
            left,top=x+c*5,y+r*4.8
            a.add_patch(FancyBboxPatch((left+.14,top+.14),4.72,4.52,
                boxstyle='round,pad=0,rounding_size=.4',fc='white',ec='#D9E0E6',
                lw=.35,zorder=3))
            for d,(u,v) in enumerate(((2.55,.45),(1.52,2.6),(.45,.45))):
                filename=spec['feature_assets'][f'D{d+1}'][stimulus[d]-1]
                path=PACKAGE/'assets/food3'/filename
                with Image.open(path) as source:
                    icon=source.convert('RGBA')
                    icon=icon.crop(icon.getchannel('A').getbbox())
                side=1.85
                iw,ih=icon.size
                scale=side/max(iw,ih)
                ww,hh=iw*scale,ih*scale
                cx,cy=left+u+(side-ww)/2,top+v+(side-hh)/2
                a.imshow(icon,extent=(cx,cx+ww,cy+hh,cy),interpolation='lanczos',
                         aspect='auto',zorder=4)
                icon_hashes[filename]=hashlib.sha256(path.read_bytes()).hexdigest()
    a._non_dis_example=dict(source=spec['source_behavior'],phase=spec['phase'],
        subject=spec['subject'],round=spec['round'],rows=rows.tolist(),fel=recovered.tolist(),
        cluster_id=spec['cluster_id'],cluster_choices=int(clusters.iloc[index].total_visits),
        coordinates_mm=point.tolist(),box_mm=[x,y,w,h],line_count=2,
        feature_assets_sha256=icon_hashes,interpretation=spec['scope'])


def enrichment_space(fig):
    a=axis(fig,'a',3,3,88,49,schematic=True)
    data=read('fig2a_clusters.csv')
    fel,xy=fel_projection(data)
    for v in FEL_VECTORS_MM:arrow(a,FEL_ORIGIN_MM,FEL_ORIGIN_MM+1.12*v,color=INK,lw=.7)
    sizes=data.total_visits.to_numpy()*.33
    points=a.scatter(xy[:,0],xy[:,1],s=sizes,c=data.relative_visit_order,
                     cmap='viridis',norm=Normalize(0,1),edgecolor='#7D858D',lw=.3,zorder=3)
    vertex=data.is_FDS_vertex.astype(str).str.lower().eq('true')
    a.scatter(xy[vertex,0],xy[vertex,1],s=sizes[vertex]*1.35,facecolors='none',edgecolors='#D6A124',lw=.7,zorder=4)
    a.text(35.5,7.3,'FEL in D1',ha='right',fontsize=6)
    a.text(26,44.5,'FEL in D2',ha='left',va='top',fontsize=6)
    a.text(59.5,41,'FEL in D3',ha='left',fontsize=6)
    # Exact stimulus crops contain no scientific text or data marks. No bitmap
    # cluster geometry, original colorbar or inherited fonts are imported.
    # Actual task coding: D1 dessert, D2 main dish, D3 drink. Only the
    # D2 example is omitted; its axis, visits and DIS endpoint are retained.
    a._endpoint_callouts=[]
    for dim,label,bounds,crop,old_bounds in [
        (1,'Dessert',(42.5,.3,14,13.4),(1095,120,1542,546),(50,3.2,18,17.2)),
        (3,'Drink',(65,22,16,15.4),(14,974,432,1373),(69.5,17.2,16,15.4)),
    ]:
        endpoint=FEL_ORIGIN_MM+FEL_VECTORS_MM[dim-1]
        index=np.flatnonzero(np.all(np.isclose(xy,endpoint),axis=1))[0]
        radius=np.sqrt(sizes[index]*1.35)/2*25.4/72+.35*25.4/72
        endpoint_example(a,dim,label,bounds,crop,old_bounds,radius)
    non_dis_example(a,data,fel,xy,sizes)
    title=a.text(12,4.2,'Visit order',fontsize=5,va='top',ha='center')
    cax=a.inset_axes([2/88,1-10.1/49,20/88,1.6/49])
    cb=fig.colorbar(points,cax=cax,orientation='horizontal',ticks=[0,1])
    cb.ax.set_xticklabels(['Early','Late'])
    cb.ax.tick_params(labelsize=5,length=0,pad=1.8)
    cb.outline.set_linewidth(.4)
    cb.ax.get_xticklabels()[0].set_ha('left')
    cb.ax.get_xticklabels()[1].set_ha('right')
    examples=[]
    for x,size in [(4.5,3),(8,5)]:
        dot,=a.plot([x],[17],marker='o',markersize=size,ls='none',
                    markerfacecolor='#929CA5',markeredgecolor='#7D858D',
                    markeredgewidth=.3,gid='legend-visits',zorder=5)
        examples.append(dot)
    ring,=a.plot([6.25],[22],marker='o',markersize=5,ls='none',
                 markerfacecolor='none',markeredgecolor='#D6A124',
                 markeredgewidth=.7,gid='legend-dis',zorder=5)
    labels=[a.text(11.5,17,'Choice frequency',fontsize=5,va='center'),
            a.text(11.5,22,'DIS',fontsize=5,va='center')]
    a._visit_legend_artists=[title,*examples,ring,*labels]
    a._visit_colorbar=cb
    a._visit_legend=dict(location='upper-left blank area',orientation='horizontal',
        title='Visit order',endpoints=['Early','Late'],
        colorbar_size_mm=[20,1.6],examples=['Choice frequency','DIS'],
        marker_area_encoding='proportional to the total number of choices pooled across participants',
        footer='FEL, feature enrichment level')
    # Preserve the hero row's physical lower edge, so statistical panels stay put.
    a.text(1,49.132,'FEL, feature enrichment level',
           va='bottom',fontsize=5)
    clearances=[]
    for i,point in enumerate(xy):
        radius=np.sqrt(sizes[i]*(1.35 if vertex.iloc[i] else 1))/2*25.4/72+.35*25.4/72
        for pic in a._pictogram_rects:
            x,y,w,h=pic['box_mm']
            distance=np.linalg.norm(point-np.clip(point,[x,y],[x+w,y+h]))
            clearances.append(float(distance-radius))
    a._minimum_marker_image_clearance_mm=min(clearances)
    assert a._minimum_marker_image_clearance_mm>=.5,a._minimum_marker_image_clearance_mm
    a._fel_projection=dict(definition='FEL_d = sum_r max_k n[d,k,r]',
        source='csv/fig2a_clusters.csv',energy_to_fel={'0':3,'4/3':5,'2':6,'10/3':7,'6':9},
        normalization='(FEL_d - 3) / 6',origin_mm=FEL_ORIGIN_MM.tolist(),
        vectors_mm=FEL_VECTORS_MM.tolist(),fel_counts=fel.tolist(),coordinates_mm=xy.tolist(),
        unique_fel_triplets=int(len(np.unique(fel,axis=0))),
        unique_projected_points=int(len(np.unique(xy,axis=0))),
        total_visits=int(data.total_visits.sum()))
    a._source_count=len(data)


def canonical_choices(fig):
    label=text(fig,44.,54.,'Dimension-isolating sampling (DIS)',ha='center')
    label.set_gid('dis-panel-heading')
    for task,x,path,color in [(1,3,'shape_2049.png',TASK1),(2,46.5,'shape_151.png',TASK2)]:
        b=axis(fig,f'b{task}',x,62.2,38,30.5,schematic=True)
        image=Image.open(PACKAGE/'assets/ppt_media'/path)
        image_w=36.8
        image_h=image_w*image.height/image.width
        image_x=(38-image_w)/2
        pictogram(b,path,(image_x,0,image_w,image_h))
        if task==2:
            # Native locator outlines identify the shared sofa; source pixels
            # and all four rooms are retained without alteration.
            for left,top,right,bottom in [(310,305,680,499),(1045,292,1450,498),
                                           (291,908,649,1108),(1060,909,1455,1120)]:
                b.add_patch(Rectangle((image_x+left/image.width*image_w,top/image.height*image_h),
                    (right-left)/image.width*image_w,(bottom-top)/image.height*image_h,fc='none',ec=TASK2,lw=.85,zorder=4))
        text(fig,x+19,58.2,f'Task {task}',ha='center')
        text(fig,x+19,94.0,'Fruit-enriched' if task==1 else 'Sofa-enriched',ha='center')


def enrichment_diff(fig):
    stats=read('c_statistics.csv')
    a=axis(fig,'c',100.5,12,28.5,32)
    a.axhline(0,color=GRAY,ls=(0,(3,2)),lw=.65)
    counts={}
    for task,color in [(1,TASK1),(2,TASK2)]:
        frame=read(f'c_task{task}_subject_differences.csv')
        # Lower is the exact negative of Highest for every participant. Show
        # this single contrast once, keeping both frozen columns and BH(2).
        vals=frame.loc[frame.category.eq('Highest'),'diff_pct'].to_numpy()
        position=task-1
        a.boxplot([vals],positions=[position],widths=BOX_AXES_WIDTH_RATIO*2.1,patch_artist=True,showfliers=False,
            boxprops=dict(facecolor=matplotlib.colors.to_rgba(color,TASK1_BOX_ALPHA if task==1 else .38),edgecolor=INK,lw=.65),
            medianprops=dict(color=INK,lw=.85),whiskerprops=dict(color=INK,lw=.65),
            capprops=dict(color=INK,lw=.65))
        # All observations enter box summaries and tests; individual marks are
        # hidden at the user's request. Summary mean diamonds remain.
        a.scatter([position],[np.mean(vals)],s=9,marker='D',fc=INK,ec='white',lw=.4,zorder=4)
        p=stats.loc[stats.task.eq(task)&stats.category.eq('Highest')].iloc[0]
        a.text(position,104,p.sig,fontsize=6,fontweight='bold',ha='center',va='center')
        fig._tests.append(dict(panel='c',task=task,comparison='highest FEL human-minus-random against zero',
                               label=p.sig,p_adjusted=float(p.p_fdr),correction='BH(2) within task'))
        counts[str(task)]=len(vals)
    a.set(ylim=(-10,110),xlim=(-.55,1.55),yticks=[0,50,100],xticks=[0,1],
          xticklabels=['Task 1','Task 2'],ylabel='Highest-FEL choice rate\nHuman − random')
    a.spines['bottom'].set_bounds(0,1)
    a._source_count=sum(counts.values());a._task_counts=counts


def room_object_icon(a,name,x,y,w=4.3,h=4.1):
    """Compact native object glyphs; each represents a dimension, not a feature."""
    edge='#59616B'
    def rect(u,v,ww,hh,fill='#EFE7DA',rounded=False):
        cls=FancyBboxPatch if rounded else Rectangle
        kwargs=dict(boxstyle='round,pad=0,rounding_size=.22') if rounded else {}
        a.add_patch(cls((x+u*w,y+v*h),ww*w,hh*h,fc=fill,ec=edge,lw=.5,zorder=4,**kwargs))
    if name=='Sofa':
        rect(.12,.26,.76,.44,rounded=True)
        rect(.18,.49,.64,.23,fill='#D5C6AE',rounded=True)
        rect(.04,.40,.16,.37,rounded=True);rect(.80,.40,.16,.37,rounded=True)
        a.plot([x+.23*w,x+.23*w],[y+.76*h,y+.86*h],color=edge,lw=.6,zorder=4)
        a.plot([x+.77*w,x+.77*w],[y+.76*h,y+.86*h],color=edge,lw=.6,zorder=4)
    elif name=='Table':
        rect(.10,.29,.80,.17,fill='#D5C6AE',rounded=True)
        a.plot([x+.24*w,x+.18*w],[y+.46*h,y+.84*h],color=edge,lw=.65,zorder=4)
        a.plot([x+.76*w,x+.82*w],[y+.46*h,y+.84*h],color=edge,lw=.65,zorder=4)
    elif name=='Carpet':
        rect(.08,.29,.84,.43,fill='#E6E9E3',rounded=True)
        for t in [.34,.49,.64]:
            a.plot([x+.20*w,x+.80*w],[y+t*h,y+t*h],color='#99A38F',lw=.45,zorder=4)
        for t in [.20,.40,.60,.80]:
            a.plot([x+t*w,x+t*w],[y+.72*h,y+.79*h],color=edge,lw=.4,zorder=4)
    else:raise ValueError(name)


def persistence_schematic(fig):
    a=axis(fig,'d',92,62.2,84,25.5,schematic=True)
    a.set_ylim(28,0)
    a.set_xlim(0,92)
    arrow(a,(13.5,.4),(88,.4),color=GRAY)
    a.text(51,-.4,'Trial progress',fontsize=6,ha='center',va='bottom')
    rows=[r for r in source_shapes() if 2252 in r['path'] and r['fill'] and r['rect_mm'][2]<3]
    # Exact source sequences: Task1 nine ordered 3x3 blocks (81 squares),
    # Task2 seven groups of four selected-feature squares (28 squares).
    for r in rows:
        x,y,w,h=r['rect_mm'];task=1 if y<125 else 2
        x=13.5+(x-19.045916666666663)*.78
        if task==1:
            yy=3.1+(y-113.50944444444444)*.61;hh=h*.61
            bases={'accent3':'#B35AAF','accent1':'#299CCC','accent5':'#36A36A'}
            mod=next(float(v['lumMod'])/100000 for v in r['fill']['modifiers'] if 'lumMod' in v)
            rgba=matplotlib.colors.to_rgba(bases[r['fill']['scheme']],{.6:1.,.4:.58,.2:.27}[mod])
        else:
            yy=20+(y-134.91466666666668)*.61;hh=h*.61
            rgba=(*r['fill']['rgb'],r['fill']['alpha'])
        cell=FancyBboxPatch((x,yy),w*.78,hh,boxstyle='round,pad=0,rounding_size=.17',
                           fc=rgba,ec='none',zorder=2)
        cell._source_task=task;a.add_patch(cell)
    # Source group extents, with Task1 colours aligned to the dimensions in a.
    a._dimension_icons=[]
    for task,y,labely,segments in [
        (1,9.5,11.2,[(13.3,35.9,'#B35AAF'),(38.3,52.9,'#299CCC'),(56.2,86.3,'#36A36A')]),
        (2,22.7,24.3,[(13.3,42.7,'#BA601D'),(45.2,64.0,'#7252B4'),(66.1,84.9,'#3E7F24')])]:
        a.text(.3,5.4 if task==1 else 21,f'Task {task}',fontsize=6,color=INK,va='center')
        for dim,(x1,x2,color) in enumerate(segments,1):
            top=10.15 if task==1 else 23.10
            icon_bounds=[x1+.1,top,4.3,4.1]
            label=['Dessert','Main dish','Drink'][dim-1] if task==1 else ['Sofa','Table','Carpet'][dim-1]
            if task==1:
                name=['dessert','main_dish','drink'][dim-1]
                pictogram(a,f'assets/d_dimension_icons/{name}.png',icon_bounds)
            else:room_object_icon(a,label,*icon_bounds)
            a.text(x1+5.2,top+2.05,f'D{dim}',fontsize=6,ha='left',va='center')
            a.plot([x1+9.5,x2],[top+2.05,top+2.05],color=color,lw=1.1)
            a._dimension_icons.append(dict(task=task,dimension=dim,label=label,
                color=color,box_mm=icon_bounds,representative_only=True))
        a.text(88,5.8 if task==1 else 21.4,'…',fontsize=6,va='center')
    coverage_color='#626C76'
    a.add_patch(FancyBboxPatch((11.6,18.1),75.8,9.5,boxstyle='round,pad=0,rounding_size=1.1',
                              fc='none',ec=coverage_color,lw=.75,ls=(0,(3,2)),zorder=1))
    a.add_patch(FancyBboxPatch((12.3,18.9),31.5,8.3,boxstyle='round,pad=0,rounding_size=.6',
                              fc='#EAF3F7',ec=TASK1,lw=.65,zorder=0))
    # Run: one dimension-specific segment. Coverage: the entire multi-dimension
    # envelope. Separate origins and a routed leader make this scope explicit.
    arrow(a,(28.05,27.2),(28.05,31.8),color=TASK1,lw=.85)
    a.patches[-1].set_clip_on(False)
    a.plot([49.5,49.5,70.2],[27.6,29.7,29.7],color=coverage_color,
           lw=.8,ls=(0,(3,2)),clip_on=False)
    arrow(a,(70.2,29.7),(70.2,31.8),color=coverage_color,lw=.8)
    a.patches[-1].set_clip_on(False)
    a.patches[-1].set_linestyle((0,(3,2)))
    a.text(28.05,33.5,'Same-dimension run',ha='center',va='top',fontsize=6,color=TASK1)
    a.text(70.2,33.5,'Dimension coverage',ha='center',va='top',fontsize=6,color=coverage_color)
    a._callouts=dict(run=dict(origin=[28.05,27.2],color=TASK1,scope='single dimension segment',linestyle='solid'),
                     coverage=dict(origin=[49.5,27.6],color=coverage_color,scope='whole multi-dimension envelope',linestyle='dashed'))
    a._source_count=109
    a._task_counts={str(task):sum(getattr(p,'_source_task',None)==task for p in a.patches)
                    for task in [1,2]}
    assert a._task_counts=={'1':81,'2':28}


def mean_enrichment(fig):
    ax=axis(fig,'e',144,12,28.5,32)
    frame=read('e_normalized_progress.csv')
    for task,color,marker in [(1,TASK1,'o'),(2,TASK2,'s')]:
        df=frame[frame.task.eq(task)]
        xx=df.progress_center.to_numpy();mean=df.mean_percent_maximum.to_numpy();se=df.sem_percent_maximum.to_numpy()
        ax.fill_between(xx,mean-se,mean+se,color=color,alpha=.20,lw=0)
        ax.plot(xx,mean,color=color,marker=marker,ms=2.1,mfc='white',mew=.65,lw=1.0,label=f'Task {task}')
    ax.set(xlim=(0,1),xticks=[0,.5,1],ylim=(65,95),yticks=[65,75,85,95],ylabel='Mean FEL\n(% of task maximum)')
    legend=ax.legend(loc='lower right',bbox_to_anchor=(1,1.11),ncol=2,frameon=False,
                     fontsize=5,columnspacing=.9,handlelength=1.5,borderaxespad=0)
    ax._trajectory_legend=legend
    ax._uncertainty_min=float((frame.mean_percent_maximum-frame.sem_percent_maximum).min())
    ax._uncertainty_max=float((frame.mean_percent_maximum+frame.sem_percent_maximum).max())
    ax._source_count=len(frame)
    text(fig,158.25,51.3,'Normalized trial progress',ha='center',size=5)


def ppt_statistics(fig):
    audit=json.loads((PACKAGE/'analysis/source_consistency_statistics_audit.json').read_text())
    for name,digest in audit['outputs'].items():
        assert hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()==digest,name
    tests=pd.read_csv(PACKAGE/'analysis/f_source_consistent_statistics.csv')
    source=pd.read_csv(PACKAGE/'analysis/f_source_data.csv')
    assert len(tests)==10 and tests.multiplicity.eq('BH-FDR within panel').all()
    labels=['Within-dim. run length\n(normalized)','Dimension coverage\n(normalized)','Single-dim. bout length\n(trials)','Dimensions explored']
    limits=[(0,.15),(0,1.20),(0,5),(0,6)]
    ticks=[[0,.05,.1,.15],[0,.4,.8,1.2],[0,2,4],[0,2,4,6]]
    for i,x in enumerate([13.5,57,100.5,144]):
        a=axis(fig,f'f{i+1}',x,112,28.5,21)
        rows=tests[tests.panel.eq(f'f{i+1}')].set_index('group')
        reference=rows.iloc[0]
        a.axhline(reference.random_mean,color='#737B83',ls=(0,(3,2)),lw=.7,zorder=.5)
        a._random_reference=dict(n=int(reference.n_random),mean=float(reference.random_mean),
            p025=float(reference.random_p025),p975=float(reference.random_p975),
            definition='2.5th-97.5th percentiles across supplied simulated-participant summaries; not a mean confidence interval',
            zero_width_range=bool(reference.random_p025==reference.random_p975),range_shown=False)
        groups=['3D-NE','4D-E','4D-NE'] if i<2 else ['Obs','NObs']
        values=[source.loc[source.panel.eq(f'f{i+1}')&source.group.eq(group),'value'].dropna().to_numpy()
                for group in groups]
        colors=[TASK1_PHASE_COLORS[group] for group in groups] if i<2 else [TASK2,'#B68F00']
        boxplot(a,values,colors)
        a.set_ylim(limits[i]);a.set_yticks(ticks[i]);a.set_ylabel(labels[i],fontsize=6)
        a.set_xticks(np.arange(len(groups)));a.set_xticklabels(groups,fontsize=5)
        a.set_yticklabels([f'{v:g}' for v in ticks[i]])
        a._human_range=[float(min(min(v) for v in values)),float(max(max(v) for v in values))]
        a._human_group_counts=[len(v) for v in values]
        assert a.get_ylim()[0]<=a._summary_range[0]<a._summary_range[1]<a.get_ylim()[1]
        for j,group in enumerate(groups):
            row=rows.loc[group]
            y=[.132,1.10,4.55,5.55][i]
            annotation=a.text(j,y,row.stars,ha='center',va='center',fontsize=6,
                              fontweight='bold',color=INK,zorder=6)
            if not hasattr(a,'_f_annotations'):a._f_annotations=[]
            a._f_annotations.append(annotation)
            fig._tests.append(dict(panel=f'f{i+1}',group=group,comparison=f'{group} vs supplied random reference',
                test='two-sided Mann–Whitney U; asymptotic; ties and continuity corrected',
                correction='BH-FDR within panel',n_human=int(row.n_human),n_random=int(row.n_random),
                U=float(row.U),p_raw=float(row.p_raw),p_adjusted=float(row.p_display),label=row.stars))
        a.spines['bottom'].set_bounds(0,len(groups)-1)
        a._source_count=int(rows.n_human.sum())
    text(fig,49.5,106,'Task 1',ha='center')
    text(fig,136.5,106,'Task 2',ha='center')
    handles=[Line2D([],[],color='#737B83',ls=(0,(3,2)),lw=.7,label='Random mean')]
    leg=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,1-106/HEIGHT_MM),ncol=1,frameon=False,fontsize=5,columnspacing=1.2,handlelength=1.8,borderaxespad=0)
    leg._layout_row=fig._active_row


def boxplot(a,values,colors,*,paired_means=False):
    span=max(len(values)-1,1)+1.1
    width=BOX_AXES_WIDTH_RATIO*span
    bp=a.boxplot(values,positions=np.arange(len(values)),widths=width,patch_artist=True,showfliers=False,
                 boxprops=dict(lw=.65),medianprops=dict(lw=.85,color=INK),whiskerprops=dict(lw=.65),capprops=dict(lw=.65))
    a._box_patches=bp['boxes']
    for i,(patch,color) in enumerate(zip(bp['boxes'],colors)):
        patch.set_facecolor(matplotlib.colors.to_rgba(color,TASK1_BOX_ALPHA if color==TASK1 else .58))
        patch.set_edgecolor(INK)
        bp['medians'][i].set_color(INK)
        for k in range(2):
            bp['whiskers'][2*i+k].set_color(INK);bp['caps'][2*i+k].set_color(INK)
    means=[float(np.mean(v)) for v in values]
    shown_values=[float(y) for whisker in bp['whiskers'] for y in whisker.get_ydata()]+means
    a._summary_range=[min(shown_values),max(shown_values)]
    if paired_means:a.plot(np.arange(len(values)),means,color=INK,lw=.65,zorder=3)
    a.scatter(np.arange(len(values)),means,s=8,marker='D',fc=INK,ec='white',lw=.4,zorder=4)
    a.set_xlim(-.55,len(values)-.45)
    a.spines['bottom'].set_bounds(0,len(values)-1)
    a._source_count=sum(len(v) for v in values)


def bracket(a,x1,x2,y,sig,*,height=.022):
    a.plot([x1,x1,x2,x2],[y,y+height,y+height,y],color=INK,lw=.65,clip_on=False)
    a.text((x1+x2)/2,y+height+.012,sig,ha='center',va='bottom',fontsize=6,
           fontweight='bold' if str(sig).startswith('*') else 'normal')


def survey_and_terms(fig):
    stats=pd.read_csv(PACKAGE/'analysis/g_source_consistent_statistics.csv')
    for task,x,color,levels in [(1,13.5,TASK1,[0,1,2]),(2,57,TASK2,[1,2])]:
        a=axis(fig,f'g{task}',x,141,28.5,21)
        frame=read(f'g_task{task}_subjects.csv')
        values=[frame.loc[frame.level.eq(k),'DIS_ratio'].to_numpy() for k in levels]
        boxplot(a,values,[color]*len(values))
        a.spines['bottom'].set_position(('outward',3.5))
        a.set_xticks(np.arange(len(values)))
        a.set_xticklabels([["None","Partial","Perfect"][k] for k in levels],fontsize=5)
        a.set(ylim=(0,1.32),yticks=[0,.5,1],ylabel='% of DIS trial')
        a.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{100*value:g}'))
        tests=stats[stats.task.eq(task)&stats.shown]
        for _,row in tests.iterrows():
            # Preserve the complete box samples and source displayed pairs;
            # only the test uses source percentile trimming.
            bracket(a,levels.index(row.group_a),levels.index(row.group_b),
                    1.04+(.12 if task==1 and row.group_a==1 else 0),row.stars)
            fig._tests.append(dict(panel='g',task=task,group_a=int(row.group_a),group_b=int(row.group_b),
                                   label=row.stars,p_raw=float(row.p_display),
                                   n_plot_a=int(row.n_plot_a),n_plot_b=int(row.n_plot_b),
                                   n_test_a=int(row.n_test_a),n_test_b=int(row.n_test_b),
                                   correction='Unadjusted, as supplied',test=row.test))
        text(fig,x+14.25,138,f'Task {task}',ha='center')
    interview_stats=pd.read_csv(PACKAGE/'analysis/h_source_consistent_statistics.csv')
    for task,x,color,upper in [(1,100.5,TASK1,.14),(2,144,TASK2,.6)]:
        a=axis(fig,f'h{task}',x,141,28.5,21)
        frame=read(f'h_task{task}_source_metrics.csv')
        values=[frame.DIS.to_numpy(),frame.All.to_numpy()]
        for left,right in zip(*values):a.plot([0,1],[left,right],color=GRAY,alpha=.13,lw=.35,zorder=0)
        boxplot(a,values,[PURPLE,GRAY],paired_means=True)
        a.spines['bottom'].set_position(fig._panels[f'g{task}'].spines['bottom'].get_position())
        offsets=.16*((np.arange(len(frame))*np.sqrt(2))%1-.5)
        for i,(vals,c) in enumerate(zip(values,[PURPLE,GRAY])):
            a.scatter(i+offsets,vals,s=2.5,color=c,alpha=.50,ec='none',zorder=3)
        a.set_xticks([0,1]);a.set_xticklabels(['DIS','Full'])
        ticks=[0,.05,.10] if task==1 else [0,.2,.4,.6]
        a.set(ylim=(0,upper),yticks=ticks,ylabel='Dimension-term\nfrequency')
        sig=interview_stats.loc[interview_stats.task.eq(task),'stars'].iloc[0]
        bracket(a,0,1,upper*.89,sig,height=upper*.025)
        # The absolute bracket text offset is scaled for these frequency axes.
        a.texts[-1].set_y(upper*.94)
        fig._tests.append(dict(panel='h',task=task,comparison='DIS vs Full paired',label=sig,
                               p_raw=float(interview_stats.loc[interview_stats.task.eq(task),'p_display'].iloc[0]),
                               correction='Unadjusted, as supplied'))
        text(fig,x+14.25,138,f'Task {task}',ha='center')
    text(fig,49.5,167.5,'Dimension identification',size=5,ha='center')


def content_bounds(fig,row):
    """Rendered envelope, including ticks, axis labels, legends and icons."""
    renderer=fig.canvas.get_renderer()
    boxes=[]
    for a in fig.axes:
        if getattr(a,'_layout_row',None)==row:
            b=a.get_tightbbox(renderer)
            if b is not None:boxes.append(b)
    for t in [*fig.texts,*fig.legends]:
        if getattr(t,'_layout_row',None)==row and t.get_visible():
            boxes.append(t.get_window_extent(renderer))
    assert boxes,row
    scale=25.4/fig.dpi
    return dict(left=min(b.x0 for b in boxes)*scale,
                top=HEIGHT_MM-max(b.y1 for b in boxes)*scale,
                right=max(b.x1 for b in boxes)*scale,
                bottom=HEIGHT_MM-min(b.y0 for b in boxes)*scale)


def shift_row(fig,row,down_mm):
    for a in fig.axes:
        if getattr(a,'_layout_row',None)==row:
            x,y,w,h=a.get_position().bounds
            a.set_position((x,y-down_mm/HEIGHT_MM,w,h))
    for t in fig.texts:
        if getattr(t,'_layout_row',None)==row:
            x,y=t.get_position();t.set_position((x,y-down_mm/HEIGHT_MM))
    for leg in fig.legends:
        if getattr(leg,'_layout_row',None)==row:
            b=leg.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
            leg.set_bbox_to_anchor((b.x0,b.y0-down_mm/HEIGHT_MM,b.width,b.height),
                                  transform=fig.transFigure)


def reflow_rows(fig):
    fig.canvas.draw()
    initial={row:content_bounds(fig,row) for row in ['hero','metrics','density','reports']}
    moves={}
    for previous,row,gap in [('hero','metrics',2.),('metrics','density',3.),('density','reports',3.)]:
        down=content_bounds(fig,previous)['bottom']+gap-content_bounds(fig,row)['top']
        shift_row(fig,row,down);moves[row]=float(down)
        fig.canvas.draw()
    final={row:content_bounds(fig,row) for row in initial}
    gaps={f'{a}_to_{b}':float(final[b]['top']-final[a]['bottom'])
          for a,b in [('hero','metrics'),('metrics','density'),('density','reports')]}
    assert np.allclose(list(gaps.values()),[2,3,3],atol=.02),gaps
    assert final['reports']['bottom']<HEIGHT_MM-1,final
    # A separate local check records the user's requested b-to-c/e gap.
    renderer=fig.canvas.get_renderer()
    bboxes=[fig._panels[n].get_tightbbox(renderer) for n in ['c','e']]
    bboxes.extend(t.get_window_extent(renderer) for t in fig.texts
                  if getattr(t,'_layout_row',None)=='hero' and t.get_position()[0]>.50)
    b_bottom=HEIGHT_MM-min(b.y0 for b in bboxes)*25.4/fig.dpi
    fig._row_spacing=dict(initial_bounds_mm=initial,final_bounds_mm=final,
                         translations_down_mm=moves,content_gaps_mm=gaps,
                         upper_metrics_to_examples_clearance_mm=float(final['metrics']['top']-b_bottom),
                         definition='Union of final native tight axes bounds, icons, figure captions and external legends')


def apply_tick_endpoints_and_category_spacing(fig):
    """Display-only categorical margins; true numeric observations stay fixed."""
    # Public b uses exactly the three requested ticks; its negative margin
    # retains the original lower whisker without a zero-omission marker.
    # Public c retains its uniform65/75/85/95 ticks and full SEM envelope.
    fig._panels['c'].set_yticks([0,50,100])
    fig._panels['e'].set_yticks([65,75,85,95])
    fig._panels['h1'].set_yticks([0,.05,.10,.14])
    spacing={}
    for name in CATEGORICAL_BOX_AXES:
        a=fig._panels[name]
        boxes=[patch for patch in a.patches if isinstance(patch,PathPatch)]
        record=compact_category_margin(a,boxes)
        gap=record['after']['ylabel_to_box_mm']
        old_pad=a.yaxis.labelpad
        # Only two labels slightly exceed10mm after category compaction.
        # Retain positive label padding and verify actual tick-text separation.
        if gap>10:
            proposed=old_pad-(gap-9.9)*72/25.4
            if proposed>=1:
                a.yaxis.labelpad=proposed
        record['ylabel_padding_pt']={'before':float(old_pad),'after':float(a.yaxis.labelpad)}
        record['after']=category_gaps(a,boxes)
        spacing[SOURCE_TO_PUBLIC[name]]=dict(record,source_axis=name)
    fig._category_label_spacing=spacing
    for name in QUANTITATIVE_AXES:
        bound_spines_to_ticks(fig._panels[name])


def qa_and_save(fig,output):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    fonts=sorted({round(t.get_fontsize(),3) for t in texts})
    assert fonts==[5.,6.,8.], fonts
    clipped=[];bounds=[]
    for t in texts:
        b=t.get_window_extent(renderer)
        rec=dict(text=t.get_text(),box_pixels=[float(v) for v in b.extents])
        bounds.append((t,b,rec))
        if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.x1+.5 or b.y1>fig.bbox.y1+.5:clipped.append(rec)
    collisions=[]
    for i,(t,b,r) in enumerate(bounds):
        for u,c,s in bounds[i+1:]:
            overlap=(min(b.x1,c.x1)-max(b.x0,c.x0),min(b.y1,c.y1)-max(b.y0,c.y0))
            if min(overlap)>.8:
                collisions.append(dict(first=t.get_text(),second=u.get_text(),overlap_pixels=overlap))
    icon_collisions=[]
    icon_axes=[a for a in fig.axes if a.get_label().startswith('source_')]
    for icon in icon_axes:
        for t,b,_ in bounds:
            c=icon.bbox
            overlap=(min(b.x1,c.x1)-max(b.x0,c.x0),min(b.y1,c.y1)-max(b.y0,c.y0))
            if min(overlap)>.8:
                icon_collisions.append(dict(icon=icon.get_label(),text=t.get_text(),overlap_pixels=overlap))
    for name,a in fig._panels.items():
        for icon in getattr(a,'_dimension_icons',[]):
            x,y,w,h=icon['box_mm']
            corners=np.array([a.transData.transform((x,y)),a.transData.transform((x+w,y+h))])
            c=Bbox(np.array([corners.min(axis=0),corners.max(axis=0)]))
            for t,b,_ in bounds:
                overlap=(min(b.x1,c.x1)-max(b.x0,c.x0),min(b.y1,c.y1)-max(b.y0,c.y0))
                if min(overlap)>.8:
                    icon_collisions.append(dict(icon=f'{name} Task{icon["task"]} D{icon["dimension"]}',
                                                text=t.get_text(),overlap_pixels=overlap))
    font_files=sorted({Path(font_manager.findfont(t.get_fontproperties())).name for t in texts})
    assert all(name.startswith('DejaVuSans') for name in font_files), font_files
    font_families=sorted({font_manager.FontProperties(fname=font_manager.findfont(t.get_fontproperties())).get_name()
                          for t in texts})
    assert font_families==['DejaVu Sans'] and len(fonts)<=3,(font_families,fonts)
    report=dict(font_sizes_pt=fonts,resolved_font_files=font_files,clipped_text=clipped,text_collisions=collisions,
                icon_text_collisions=icon_collisions,
                a_minimum_marker_image_clearance_mm=fig._panels['a']._minimum_marker_image_clearance_mm,
                a_fel_projection=fig._panels['a']._fel_projection,
                source_icon_panels=[a._source_panel for a in icon_axes],
                text_count=len(texts),scope='Native Matplotlib visible-text geometry; all quantitative f marks drawn from participant source data. No PDF audit under PNG-only contract.')
    report['font_families']=font_families
    report['internal_source_axes_to_public']=SOURCE_TO_PUBLIC
    a=fig._panels['a']
    sequence_box=fig._panels['d'].get_tightbbox(renderer)
    examples_right=max(fig._panels[name].get_tightbbox(renderer).x1 for name in ['b1','b2'])
    middle_gap_mm=(sequence_box.x0-examples_right)*25.4/fig.dpi
    assert middle_gap_mm>=4,middle_gap_mm
    report['middle_example_sequence_spacing']=dict(sequence_width_mm=84,visible_gap_mm=float(middle_gap_mm))
    colorbar=a._visit_colorbar.ax
    legend_boxes=[artist.get_window_extent(renderer) for artist in a._visit_legend_artists]
    legend_boxes.extend([colorbar.bbox,*[t.get_window_extent(renderer)
                                         for t in colorbar.get_xticklabels()]])
    legend_bounds=Bbox.union(legend_boxes)
    title_box=a._visit_legend_artists[0].get_window_extent(renderer)
    title_center=(title_box.x0+title_box.x1)/2
    colorbar_center=(colorbar.bbox.x0+colorbar.bbox.x1)/2
    title_center_error_mm=abs(title_center-colorbar_center)*25.4/fig.dpi
    assert title_center_error_mm<.001,title_center_error_mm
    marker_clearances=[]
    for collection in a.collections[:2]:
        centers=collection.get_offset_transform().transform(collection.get_offsets())
        radii=np.sqrt(collection.get_sizes())/2*fig.dpi/72
        radii+=collection.get_linewidths()[0]/2*fig.dpi/72
        for center,radius in zip(centers,radii):
            nearest=np.clip(center,legend_bounds.p0,legend_bounds.p1)
            marker_clearances.append((np.linalg.norm(center-nearest)-radius)*25.4/fig.dpi)
    assert min(marker_clearances)>=.8,min(marker_clearances)
    report['a_visit_legend']=dict(a._visit_legend,
        bounds_mm=[float(v*25.4/fig.dpi) for v in legend_bounds.extents],
        title_alignment='centered over colorbar',title_center_error_mm=float(title_center_error_mm),
        minimum_data_marker_clearance_mm=float(min(marker_clearances)))
    widths=[]
    for name in ['c','f1','f2','f3','f4','g1','g2','h1','h2']:
        a=fig._panels[name]
        axis_width_mm=a.bbox.width*25.4/fig.dpi
        patches=[p for p in a.patches if isinstance(p,PathPatch)]
        assert len(patches)==(3 if name in ['f1','f2','g1'] else 2),(name,len(patches))
        for patch in patches:
            w=patch.get_window_extent(renderer).width*25.4/fig.dpi
            ratio=w/axis_width_mm
            widths.append(dict(panel=name,width_mm=float(w),axes_width_mm=float(axis_width_mm),
                               box_axes_width_ratio=float(ratio)))
            assert abs(ratio-BOX_AXES_WIDTH_RATIO)<1e-12,(name,ratio)
    report['native_box_widths']=widths
    report['box_ratio_reference']=dict(figure='fig5_nhb_evidence_layout',panel='c',
        box_width_mm=4.95,axes_width_mm=27.9,ratio=BOX_AXES_WIDTH_RATIO)
    task1_boxes={name:[p for p in fig._panels[name].patches if isinstance(p,PathPatch)]
                 for name in ['c','g1']}
    target=matplotlib.colors.to_rgba(TASK1,TASK1_BOX_ALPHA)
    for name,patches in task1_boxes.items():
        for patch in patches[:1] if name=='c' else patches:
            assert np.allclose(patch.get_facecolor(),target),(name,patch.get_facecolor())
    phase_boxes={}
    for name in ['f1','f2']:
        a=fig._panels[name]
        records=[]
        for group,patch in zip(TASK1_PHASE_COLORS,a._box_patches):
            expected=matplotlib.colors.to_rgba(TASK1_PHASE_COLORS[group],TASK1_BOX_ALPHA)
            assert np.allclose(patch.get_facecolor(),expected),(name,group,patch.get_facecolor())
            records.append(dict(group=group,base_hex=TASK1_PHASE_COLORS[group],
                                box_rgba=list(patch.get_facecolor())))
        phase_boxes[name]=records
    counts={}
    for name in ['c','f1','f2','f3','f4','g1','g2']:
        a=fig._panels[name]
        assert len(a.collections)==(2 if name=='c' else 1),name
        # The only collections are summary mean diamonds, never individuals.
        points=sum(len(collection.get_offsets()) for collection in a.collections)
        assert points==(3 if name in ['f1','f2','g1'] else 2),(name,points)
        counts[name]=0
    for name in ['f1','f2','f3','f4']:
        assert not any(isinstance(p,Rectangle) for p in fig._panels[name].patches),name
    assert np.array_equal(fig._panels['f4'].get_yticks(),[0,2,4,6])
    for name in ['g1','g2']:
        assert fig._panels[name].spines['bottom'].get_position()==('outward',3.0)
    baselines={}
    for task in (1,2):
        g,h=[fig._panels[f'{name}{task}'] for name in ('g','h')]
        spine_y=lambda a:a.spines['bottom'].get_transform().transform(
            a.spines['bottom'].get_path().vertices)[0,1]
        spine_delta_pt=(spine_y(h)-spine_y(g))*72/fig.dpi
        text_delta_pt=(h.get_xticklabels()[0].get_window_extent(renderer).y0-
                       g.get_xticklabels()[0].get_window_extent(renderer).y0)*72/fig.dpi
        assert abs(spine_delta_pt)<.03 and abs(text_delta_pt)<.03,(task,spine_delta_pt,text_delta_pt)
        baselines[str(task)]=dict(h_spine_position=list(h.spines['bottom'].get_position()),
            spine_delta_pt=float(spine_delta_pt),tick_text_delta_pt=float(text_delta_pt))
    report['g_h_x_axis_alignment']=baselines
    report['detached_axes']=audit_detached_axes({name:fig._panels[name] for name in ['c','e','f1','f2','f3','f4','g1','g2','h1','h2']})
    report['tick_bounded_spines']=audit_tick_bounded_spines(
        {SOURCE_TO_PUBLIC[name]:fig._panels[name] for name in QUANTITATIVE_AXES})
    report['tick_endpoint_exemptions']={'a':'FEL-space schematic and embedded colour key',
        'd1/d2':'Illustrative source-image areas','e':'Persistence/coverage sequence schematic'}
    report['category_label_spacing']=fig._category_label_spacing
    report['task1_color_consistency']=dict(status='PASS',panels=['c','g1'],
        base_hex=TASK1,box_rgba=target,individual_point_groups=counts)
    report['task1_phase_palette']=dict(status='PASS',reference='fig1_f1ab_nhb_20261014.png, panel c',
        mapping=TASK1_PHASE_COLORS,box_alpha=TASK1_BOX_ALPHA,panels=phase_boxes)
    report['box_only_revision']=dict(individual_points=False,mean_diamonds=True,f_random_range=False,
        f_random_mean=True,f4_yticks=[0,2,4,6],g_x_spine_outward_pt=3.0,
        example_image_width_mm=36.8,b_Task2_sofa_outline_lw_pt=.85)
    report['axis_callout_revision']=dict(f_ylims={n:list(fig._panels[n].get_ylim()) for n in ['f1','f2','f3','f4']},
        f_yticks={n:fig._panels[n].get_yticks().tolist() for n in ['f1','f2','f3','f4']},
        g_sample_count_labels=False,example_axes_mm=[38,30.5],sequence_axes_mm=[84,25.5],d_callouts=fig._panels['d']._callouts,
        font_families=font_families,font_sizes_pt=fonts)
    qa=output/STEM/'qa';qa.mkdir(parents=True,exist_ok=True)
    (qa/'native_text_qa.json').write_text(json.dumps(report,indent=2)+'\n')
    (qa/'row_spacing.json').write_text(json.dumps(fig._row_spacing,indent=2)+'\n')
    (qa/'tick_endpoints.json').write_text(json.dumps(dict(axes=report['tick_bounded_spines'],
        exemptions=report['tick_endpoint_exemptions']),indent=2)+'\n')
    (qa/'category_label_spacing.json').write_text(json.dumps(fig._category_label_spacing,indent=2)+'\n')
    assert not clipped, clipped
    assert not collisions, collisions
    assert not icon_collisions, icon_collisions
    e=fig._panels['e']
    assert e.get_ylim()[0]<e._uncertainty_min<e._uncertainty_max<e.get_ylim()[1]
    # The entire legend sits above the plotting rectangle, so it cannot cover
    # either series or its uncertainty band in this compact two-series view.
    assert e._trajectory_legend.get_window_extent(renderer).y0>e.bbox.y1
    aligned=require_matplotlib_panel_alignment(fig,axes=list(fig._panels.values()),
        row_groups=[['b1','b2'],['c','e'],['f1','f2','f3','f4'],['g1','g2','h1','h2']],
        column_groups=[['f1','g1'],['f2','g2'],['c','f3','h1'],['e','f4','h2']],
        exemptions=[dict(panels=['d'],checks=['panel-width','horizontal-gutter'],
                         reason='The84mm sequence schematic spans two columns in the second row; it has a role-specific25.5mm plot height. The two DIS example image areas are matched38x30.5mm. Quantitative top-row axes are matched28.5x32mm.'),
                    dict(panels=['c','e','f3','f4','h1','h2'],checks=['vertical-gutter'],
                         reason='Visible whole-row clearances are equal3mm; different label/legend loads require different raw axes gaps. Column widths/edges remain checked at1.5pt.')],
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,json_out=qa/'alignment.json')
    temp=qa/'render.png';final=output/(STEM+'.png')
    fig.savefig(temp,dpi=DPI,facecolor='white')
    with Image.open(temp) as im:im.convert('RGB').save(final,dpi=(DPI,DPI))
    temp.unlink()
    with Image.open(final) as im:pixels=im.size;dpi=im.info['dpi']
    plot={name:dict(box_mm=[float(v*25.4/fig.dpi) for v in a.bbox.extents],
                    ylabel=a.get_ylabel(),source_records=getattr(a,'_source_count',None),
                    task_counts=getattr(a,'_task_counts',None),ylim=list(a.get_ylim()),
                    pictograms=getattr(a,'_pictogram_rects',None),
                    dimension_icons=getattr(a,'_dimension_icons',None),
                    callouts=getattr(a,'_callouts',None),
                    random_reference=getattr(a,'_random_reference',None),
                    human_range=getattr(a,'_human_range',None),
                    shown_summary_range=getattr(a,'_summary_range',None),
                    yticks=a.get_yticks().tolist(),xticklabels=[t.get_text() for t in a.get_xticklabels()],
                    human_group_counts=getattr(a,'_human_group_counts',None))
          for name,a in fig._panels.items()}
    plot['a']['fel_projection']=fig._panels['a']._fel_projection
    plot['a']['endpoint_callouts']=fig._panels['a']._endpoint_callouts
    plot['a']['visit_legend']=report['a_visit_legend']
    plot['a']['non_dis_example']=fig._panels['a']._non_dis_example
    plot['a']['dessert_correction']=fig._panels['a']._dessert_correction
    dis_heading=next(t for t in fig.texts if t.get_gid()=='dis-panel-heading')
    heading_record=dict(text=dis_heading.get_text(),font_size_pt=dis_heading.get_fontsize(),
        bbox_pixels=[float(v) for v in dis_heading.get_window_extent(renderer).extents])
    verification=dict(status='PASS',png=str(final),sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
         pixels=pixels,dpi=dpi,size_mm=[WIDTH_MM,HEIGHT_MM],font_sizes_pt=fonts,
         panel_letters='abcdefgh',native_text_qa='PASS',alignment=aligned['verdict'],panels={SOURCE_TO_PUBLIC[name]:dict(record,source_axis=name) for name,record in plot.items()},
         dis_example_heading=dict(heading_record,public_panel="d"),
         middle_example_sequence_spacing=report['middle_example_sequence_spacing'],
         significance_annotations=[dict(row,source_panel=row["panel"],panel={"c":"b","b":"d","d":"e","e":"c"}.get(row["panel"],row["panel"])) for row in fig._tests],
         task1_phase_palette=report['task1_phase_palette'],
         c_uncertainty_extrema=[e._uncertainty_min,e._uncertainty_max],
         c_legend_above_axes=True,
         row_spacing=fig._row_spacing,
         f_source='All valid endpoints used in native boxes and mean diamonds; individual marks hidden; random mean only; source panelwise BH-FDR',
         y_axis_breaks=fig._y_axis_breaks,
         box_ratio_reference=report['box_ratio_reference'],native_box_widths=[dict(row,source_axis=row['panel'],panel=SOURCE_TO_PUBLIC[row['panel']]) for row in widths],
         g_h_x_axis_alignment=baselines,
         detached_axes=report['detached_axes'],
         detached_axes_helper_sha256=hashlib.sha256((PACKAGE/'code/detached_axes.py').read_bytes()).hexdigest(),
         tick_bounded_spines=report['tick_bounded_spines'],
         tick_endpoint_exemptions=report['tick_endpoint_exemptions'],
         category_label_spacing=fig._category_label_spacing,
         display_style_revision_20261014=dict(backup='revisions/20261014_before_box_fliers_temperature',
             verification='code/verify_box_fliers_temperature.py',
             frozen_scientific_values_and_statistics_unchanged=True,
             numeric_view_ranges_unchanged=True,
             explicit_user_major_ticks={'b':[0,50,100]},
             negative_b_view_margin_preserved=[-10,0],
             category_translation_at_unchanged_span=True,
             tick_helper_sha256=hashlib.sha256((PACKAGE/'code/tick_bounded_spines.py').read_bytes()).hexdigest(),
             category_spacing_helper_sha256=hashlib.sha256((PACKAGE/'code/category_label_spacing.py').read_bytes()).hexdigest()),
         ylabel_revision_20261015=dict(panel='b',source_axis='c',
             ylabel='Highest-FEL choice rate\nHuman − random',
             unit_in_caption='percentage points',numeric_scaling_changed=False,
             backup='revisions/20261015_before_b_ylabel',
             current_verifier='code/verify_b_ylabel.py'),
         panel_map=PUBLIC_PANEL_MAP,source_axes_to_public=SOURCE_TO_PUBLIC)
    (output/STEM/'manifest.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps({"status":"PASS","png":str(final),"size_mm":[WIDTH_MM,HEIGHT_MM],"pixels":pixels,"fonts":fonts,"public_to_previous":{"b":"c","c":"e","d":"b","e":"d"}}))


def build_figure():
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4),dpi=DPI,facecolor='white');fig._panels={};fig._tests=[]
    fig._active_row='hero'
    enrichment_space(fig);enrichment_diff(fig);mean_enrichment(fig)
    for letter,x in [('a',3),('b',91),('c',134)]:heading(fig,letter,x,1,'task_behavior')
    fig._active_row='metrics'
    canonical_choices(fig);persistence_schematic(fig)
    heading(fig,'d',3,54,'task_behavior');heading(fig,'e',91,54)
    fig._active_row='density'
    ppt_statistics(fig);heading(fig,'f',3,106,'task_behavior')
    fig._active_row='reports'
    survey_and_terms(fig);heading(fig,'g',3,135,'survey');heading(fig,'h',91,135,'survey')
    mark_nonzero_yaxis(fig._panels['e'],panel='c')
    for name in ['c','e','f1','f2','f3','f4','g1','g2','h1','h2']:detach_axes(fig._panels[name])
    apply_tick_endpoints_and_category_spacing(fig)
    reflow_rows(fig)
    fig._y_axis_breaks=audit_yaxis_breaks({'c':fig._panels['e']})
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=PACKAGE.parent)
    parser.add_argument('--replace-draft',action='store_true')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/(STEM+'.png')).exists() and not args.replace_draft:raise FileExistsError('Existing candidate; use --replace-draft explicitly.')
    # Protect all supplied sources and ensure frozen plotting inputs are exact.
    for name,h in json.loads((PACKAGE/'plot_input_hashes.json').read_text()).items():
        assert hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()==h,name
    visual=json.loads((PACKAGE/'assets/visual_inputs.json').read_text())
    assert hashlib.sha256((PACKAGE/visual['progress_table']).read_bytes()).hexdigest()==visual['progress_sha256']
    for record in visual['source_density_adaptation']:
        assert hashlib.sha256((PACKAGE/record['derived']).read_bytes()).hexdigest()==record['derived_sha256']
    fig=build_figure()
    qa_and_save(fig,args.output_dir)
    plt.close(fig)


if __name__=='__main__':main()
