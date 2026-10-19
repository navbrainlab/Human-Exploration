"""Native Fig. 3 reconstruction; the renderer only reads frozen plot tables."""
from __future__ import annotations
import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image
from audit_panel_alignment import require_matplotlib_panel_alignment
from axis_break_marks import mark_nonzero_yaxis, audit_yaxis_breaks
from draw_data_source_icons import add_source_icon
from detached_axes import detach_axes,audit_detached_axes
from tick_bounded_spines import bound_spines_to_ticks,audit_tick_bounded_spines
from ei_definition_schematic import draw_ei_definition

PACKAGE=Path(__file__).resolve().parents[1]
STEM='fig3'
W,H,DPI=180.,196.1,600
INK='#252B30';BLUE='#0965C0';ORANGE='#D57224';DIS='#60458F';OTHER='#8D9399'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6,'axes.labelsize':6,
    'axes.titlesize':6,'xtick.labelsize':5,'ytick.labelsize':5,'legend.fontsize':5,
    'axes.linewidth':.7,'xtick.major.width':.7,'ytick.major.width':.7,
    'xtick.major.size':2,'ytick.major.size':2,'text.color':INK,'axes.labelcolor':INK,
    'xtick.color':INK,'ytick.color':INK,'axes.grid':False,'figure.facecolor':'white',
    'axes.facecolor':'white','savefig.dpi':DPI})


def read(name,group='derived_behavior'):
    return pd.read_csv(PACKAGE/'data'/group/name)


def ft(fig,x,y,s,*,size=6,weight='normal',ha='left',va='top',color=INK,rotation=0):
    t=fig.text(x/W,1-y/H,s,fontsize=size,fontweight=weight,ha=ha,va=va,
        color=color,rotation=rotation)
    return t


def heading(fig,letter,x,y,kind='task_behavior'):
    t=ft(fig,x,y,letter,size=8,weight='bold');t.set_gid('nhb-panel-letter')
    a=add_source_icon(fig,kind,left=(x+5)/W,top=1-y/H,size_mm=4.8)
    a._source_panel=letter


def axis(fig,name,x,y,w=25,h=21,*,schematic=False):
    a=fig.add_axes((x/W,1-(y+h)/H,w/W,h/H),label=name);fig._panels[name]=a
    a.spines[['top','right']].set_visible(False)
    a.tick_params(pad=2);a.xaxis.labelpad=3;a.yaxis.labelpad=3
    a.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
    if schematic:a.set(xlim=(0,w),ylim=(h,0));a.set_axis_off()
    return a


def task(fig,number,x,y):ft(fig,x,y,f'Task {number}',ha='center')


def curve(a,x,mean,error,color,*,ls='-',marker='o',alpha=.17):
    x=np.asarray(x,float);mean=np.asarray(mean,float);error=np.asarray(error,float)
    a.fill_between(x,mean-error,mean+error,color=color,alpha=alpha,lw=0)
    a.plot(x,mean,color=color,lw=1.1,ls=ls,marker=marker,ms=2.2,mew=.5)


def right_probability(fig,left,data,error_column,*,key):
    right=left.twinx();right.set_label(key)
    curve(right,data.x,data.P,data[error_column],DIS,ls=(0,(3.2,2.2)),marker='o',alpha=.12)
    right.set(ylim=(0,.9),yticks=[0,.3,.6,.9])
    right.spines[['top','left','bottom']].set_visible(False)
    right.spines['right'].set_color(DIS);right.spines['right'].set_linewidth(.7)
    right.tick_params(axis='y',colors=DIS,labelsize=5,pad=2)
    right.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{100*v:g}'))
    # A single label spans each task pair; ticks remain on every right axis.
    right.set_ylabel('% of DIS trials',fontsize=6,color=DIS)
    right.yaxis.set_label_coords(1+7.8/25,.5)
    fig._twins[key]=right
    return right


def top_progress(fig):
    data=read('a_progress_summary.csv')
    for n,y in [(1,14),(2,47)]:
        a=axis(fig,f'a{n}',12,y)
        d=data.loc[data.task.eq(n)].sort_values('x')
        curve(a,d.x,d.U,d.U_ci,BLUE if n==1 else ORANGE)
        a.set(xlim=(0,1),xticks=[0,.5,1],ylim=(0,90 if n==1 else 150),
            yticks=[0,30,60,90] if n==1 else [0,50,100,150],ylabel='Total uncertainty')
        a.set_xlabel('Normalized progress' if n==2 else '',fontsize=5)
        right_probability(fig,a,d,'P_ci',key=f'a{n}_probability')
        task(fig,n,24.5,y-5)


def rank(fig):
    data=read('b_rank_summary.csv');stats=read('b_rank_statistics.csv')
    for n,y in [(1,14),(2,47)]:
        # Task order and row boundaries match the surrounding quantitative views.
        a=axis(fig,f'b{n}',64,y,w=40,h=21)
        d=data.loc[data.task.eq(n)].sort_values('rank')
        a.errorbar(d['rank'],d['mean'],yerr=d['sem'],fmt='o-',color=BLUE if n==1 else ORANGE,
            lw=1.1,ms=2.8,mec='white',mew=.5,elinewidth=.75,capsize=1.5,capthick=.75)
        maxrank=int(d['rank'].max())
        a.set(xlim=(.75,maxrank+.25),ylim=(0,.3),xticks=d['rank'],yticks=[0,.1,.2,.3],
            ylabel='DIS probability')
        a.set_xlabel('Uncertainty rank (low → high)' if n==2 else '',fontsize=5)
        for _,s in stats.loc[stats.task.eq(n)].iterrows():
            if pd.notna(s.stars) and s.stars:
                row=d.loc[d['rank'].eq(s.comparison_rank)].iloc[0]
                a.text(s.comparison_rank,row['mean']+row['sem']+.013,s.stars,
                    ha='center',va='bottom',fontsize=6,weight='bold')
                fig._annotations.append(dict(panel='b',task=n,rank=int(s.comparison_rank),
                    label=s.stars,p_adjusted=float(s.p_fdr_bh),family=s.correction_family))
        task(fig,n,84,y-5)


def onset(fig):
    data=read('c_event_curves.csv','derived_figure9')
    summary=read('c_event_reduction_summary.csv','derived_figure9')
    stats=read('c_event_statistics.csv','derived_figure9')
    for n,y in [(1,14),(2,47)]:
        a=axis(fig,f'c{n}',127,y,w=28)
        for kind,color in [('DIS',DIS),('Progress-matched other',OTHER)]:
            d=data.loc[data.task.eq(n)&data.event_type.eq(kind)].sort_values('relative')
            curve(a,d.relative,d['mean'],d['sem'],color,alpha=.17 if color==DIS else .13)
        a.axvline(0,color='#737B83',ls=(0,(3,2)),lw=.65,zorder=0)
        a.set(xlim=(-3,3),xticks=[-3,0,3],ylim=(3,9) if n==1 else (8,14),
            yticks=[3,5,7,9] if n==1 else [8,10,12,14],ylabel='Max. dim.\nuncertainty')
        a.set_xlabel('Trial offset' if n==2 else '',fontsize=5)
        mark_nonzero_yaxis(a,panel=f'c{n}')
        reduction=axis(fig,f'c{n}_reduction',164,y,w=14)
        lo,hi=(2.2,4.0) if n==1 else (2.0,3.6)
        span=hi-lo
        upper=[]
        for x,(kind,color) in enumerate([('DIS',DIS),('Progress-matched other',OTHER)]):
            row=summary.loc[summary.task.eq(n)&summary.event_type.eq(kind)].iloc[0]
            reduction.errorbar(x,row['mean'],yerr=row['sem'],fmt='o',color=color,ms=2.8,
                mec=INK,mew=.45,elinewidth=.75,capsize=1.5,capthick=.75)
            upper.append(row['mean']+row['sem'])
        reduction.set(xlim=(-.5,1.5),ylim=(lo,hi),yticks=[2.2,2.8,3.4,4.0] if n==1 else [2.0,2.4,2.8,3.2,3.6],
            xticks=[0,1],xticklabels=['DIS','Others'])
        reduction.tick_params(axis='x',labelsize=5,pad=2)
        s=stats.loc[stats.task.eq(n)].iloc[0]
        bracket=max(upper)+.7/21*span
        height=.4/21*span
        reduction.plot([0,0,1,1],[bracket,bracket+height,bracket+height,bracket],lw=.65,color=INK)
        reduction.text(.5,bracket+height+.2/21*span,s.stars,ha='center',va='bottom',fontsize=6,weight='bold')
        mark_nonzero_yaxis(reduction,panel=f'c{n}_reduction')
        fig._annotations.append(dict(panel='c',task=n,label=s.stars,p_adjusted=float(s.p_fdr_bh),
            family=s.correction_family))
        task(fig,n,141,y-5)
        ft(fig,171,y-5,'Δ Max. dim.\nuncertainty',size=5,ha='center')


def ei_progress(fig):
    for n,y in [(1,92),(2,126)]:
        d=read(f'ei_progress_task{n}.csv','derived_ei_language')
        a=axis(fig,f'e{n}',76,y)
        curve(a,d.x,d.EI,d.ES,BLUE if n==1 else ORANGE,marker=None)
        # Source view limits clipped supplied means/SEM; expand only the view.
        a.set(xlim=(0,1),xticks=[0,.5,1],ylim=(10,28) if n==1 else (3.9,4.8),
            yticks=[10,16,22,28] if n==1 else [3.9,4.2,4.5,4.8],ylabel='Exploration index')
        a.set_xlabel('Trial progress' if n==2 else '',fontsize=5)
        right_probability(fig,a,d,'PS',key=f'e{n}_probability')
        mark_nonzero_yaxis(a,panel=f'e{n}')
        task(fig,n,88.5,y-5)


def density(fig):
    data=read('e_squared_distance_kde.csv')
    stats=read('e_squared_EI_statistics.csv','derived_figure9')
    for n,y in [(1,92),(2,126)]:
        a=axis(fig,f'f{n}',132,y,w=40)
        for status,color,label in [(True,DIS,'DIS'),(False,OTHER,'Others')]:
            d=data.loc[data.task.eq(n)&data.is_DIS.eq(status)].sort_values('x')
            scale=1000 if n==1 else 1
            density=d.density.to_numpy(float)*scale
            a.fill_between(d.x,0,density,color=color,alpha=.16,lw=0)
            a.plot(d.x,density,color=color,lw=1.1,label=label)
        a.set(xlim=(0,1500) if n==1 else (0,32),xticks=[0,500,1000,1500] if n==1 else [0,16,32],
            ylim=(0,2) if n==1 else (0,.06),yticks=[0,1,2] if n==1 else [0,.02,.04,.06],
            ylabel='Density (×10⁻³)' if n==1 else 'Density')
        a.set_xlabel('Squared exploration index' if n==2 else '',fontsize=5)
        task(fig,n,152,y-5)
        statistic=stats.loc[stats.task.eq(n)].iloc[0]
        if pd.notna(statistic.stars) and statistic.stars:
            a.text(.96,.95,statistic.stars,transform=a.transAxes,ha='right',va='top',
                fontsize=6,weight='bold',gid=f'figure3f-significance-task{n}')
            fig._annotations.append(dict(panel='f',task=n,label=statistic.stars,
                p_adjusted=float(statistic.p_fdr_bh),p_raw=float(statistic.p_raw),
                n_pairs=int(statistic.n_pairs),test=statistic.test,family=statistic.correction_family,
                endpoint=statistic.endpoint))
        # Full observed range is retained; Gaussian density outside the viewport
        # (especially negative support) is not renormalized away.
        a._density_scale=1000 if n==1 else 1


def academic_wordcloud(fig,rows,cx,cy,task_number):
    """Compact native word cloud with exact continuous frequency-size mapping."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    bounds=np.array([cx-15,cy-12,cx+15,cy+12],float)
    color='#163E67' if task_number==1 else '#703D07'
    maximum=float(rows.DIS_token_count.max());records=[]
    rgb=np.array(matplotlib.colors.to_rgb(color))
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    contrast=1.05/(float(linear@np.array([.2126,.7152,.0722]))+.05)
    # The most frequent word anchors the center; smaller terms surround it.
    # Spatial location is layout only, while font size alone encodes frequency.
    offsets=[(0,0),(-3,-4.7),(9,-1.9),(-8,3.8),(7,5),(-5,8.2),(6.5,-8)]
    for row,(dx,dy) in zip(rows.itertuples(index=False),offsets):
        x,y=cx+dx,cy+dy
        relative_count=float(row.DIS_token_count)/maximum
        size=5.+3.*np.sqrt(relative_count)
        word=ft(fig,x,y,row.word,size=size,ha='center',va='center',color=color)
        word.set_gid(f'wordcloud-task{task_number}-{row.word}')
        box=word.get_window_extent(renderer)
        b=np.array([box.x0*25.4/fig.dpi,H-box.y1*25.4/fig.dpi,
                    box.x1*25.4/fig.dpi,H-box.y0*25.4/fig.dpi])
        assert b[0]>=bounds[0] and b[1]>=bounds[1] and b[2]<=bounds[2] and b[3]<=bounds[3]
        assert contrast>=4.5
        records.append(dict(task=task_number,word=row.word,count=int(row.DIS_token_count),
            rank=int(row.rank),relative_count=relative_count,fontsize_pt=size,color=color,
            contrast_on_white=float(contrast),anchor_mm=[float(x),float(y)],bbox_mm=b.tolist()))
    assert all(a['count']>=b['count'] for a,b in zip(records,records[1:]))
    def box_gap(a,b):
        a,b=np.array(a['bbox_mm']),np.array(b['bbox_mm'])
        dx=max(a[0]-b[2],b[0]-a[2],0)
        dy=max(a[1]-b[3],b[1]-a[3],0)
        return np.hypot(dx,dy)
    gap=min(box_gap(a,b) for i,a in enumerate(records) for b in records[i+1:])
    assert gap>=.5
    fig._wordcloud_layout.append(dict(task=task_number,container_mm=bounds.tolist(),
        native_words=records,nominal_font_classes=sorted({record['fontsize_pt'] for record in records}),all_horizontal=True,
        within_task_intensity_rule='Task hue is constant; shade does not encode frequency',
        within_task_size_rule='fontsize_pt = 5 + 3*sqrt(count / task maximum)',
        layout_mode='Compact native word cloud: largest term centered, smaller terms distributed around it',
        frequency_order='Native records sorted by count; spatial location does not encode frequency',
        user_authorized_type_exception='User2026-10-14 authorizes continuous5–8pt frequency coding for words; ordinary typography remains5/6/8pt.',
        minimum_word_box_gap_mm=float(gap)))


def language(fig):
    words=read('language_top7.csv','derived_ei_language')
    effects=read('language_saved_effects.csv','derived_ei_language')
    fig._wordcloud_layout=[]
    fig._frequency_colorbars=[]
    fig._frequency_size_key=[]
    # Lead with the supplied aggregate effect; lexical examples follow below.
    a=axis(fig,'g_effect',40,169,w=42,h=14)
    for _,r in effects.iterrows():
        y=2-int(r.Task);color=BLUE if r.Task==1 else ORANGE
        a.errorbar(r.mean_difference_per_1000_tokens,y,
            xerr=[[r.mean_difference_per_1000_tokens-r.CI_low_per_1000_tokens],
                  [r.CI_high_per_1000_tokens-r.mean_difference_per_1000_tokens]],
            fmt='o',color=color,ms=3.5,mec='white',mew=.5,elinewidth=1.1,capsize=2,capthick=.75)
        a.text(r.CI_high_per_1000_tokens+2,y,r.significance,ha='left',va='center',fontsize=6,weight='bold')
        fig._annotations.append(dict(panel='g',task=int(r.Task),label=r.significance,
            p_adjusted=float(r.q_value_BH),family='BH over the two saved language contrasts'))
    a.set(xlim=(-2,75),ylim=(-.55,1.55),xticks=[0,20,40,60,75],yticks=[1,0],
        yticklabels=['Task 1','Task 2'],xlabel='Term-frequency difference')
    a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0)
    a.axvline(0,color=OTHER,lw=.65,ls=(0,(3,2)),zorder=0)
    for n,cx in [(1,103),(2,139)]:
        cy=178
        label=ft(fig,cx,160,f'Task {n} (DIS context)',ha='center')
        label.set_gid(f'wordcloud-heading-task{n}')
        rows=words.loc[words.Task.eq(n)].sort_values(['DIS_token_count','rank'],ascending=[False,True],kind='stable')
        academic_wordcloud(fig,rows,cx,cy,n)
    # The user requested two actual word clouds without a separate size key.
    # The caption defines the unchanged count-to-size mapping.


def key(fig,x,y,handles,labels):
    leg=fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(x/W,1-y/H),
        frameon=False,ncol=1,fontsize=5,handlelength=1.5,handletextpad=.6,
        borderaxespad=0,labelspacing=.35)
    return leg


def protect_sources():
    # Portable equivalent of the archival integrity gate: verify every bundled
    # frozen plot table. Historical handover code/PNGs are not rendering inputs.
    inputs=json.loads((PACKAGE/'plot_input_hashes.json').read_text())
    for name,h in inputs.items():
        assert hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()==h,name
    return {'n_frozen_plot_tables':len(inputs),'local_copies_unchanged':True,
        'scope':'Portable frozen plot tables; original archival hashes recorded in provenance.json'}


def text_qa(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    fonts=sorted({t.get_fontsize() for t in texts})
    prefixes=('wordcloud-task','wordcloud-frequency-size-key')
    ordinary=[t for t in texts if not str(t.get_gid()).startswith(prefixes)]
    frequency=[t for t in texts if str(t.get_gid()).startswith(prefixes)]
    ordinary_fonts=sorted({t.get_fontsize() for t in ordinary})
    frequency_fonts=sorted({t.get_fontsize() for t in frequency})
    assert all(5<=t.get_fontsize()<=8 for t in frequency)
    clipped=[];collisions=[];icons=[]
    bounds=[(t,t.get_window_extent(renderer)) for t in texts]
    for i,(t,b) in enumerate(bounds):
        if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.x1+.5 or b.y1>fig.bbox.y1+.5:clipped.append(t.get_text())
        for u,c in bounds[i+1:]:
            overlap=[min(b.x1,c.x1)-max(b.x0,c.x0),min(b.y1,c.y1)-max(b.y0,c.y0)]
            if min(overlap)>.8:collisions.append({'a':t.get_text(),'b':u.get_text(),'overlap_pixels':overlap})
        for a in fig.axes:
            if not a.get_label().startswith('source_'):continue
            c=a.bbox
            if min(min(b.x1,c.x1)-max(b.x0,c.x0),min(b.y1,c.y1)-max(b.y0,c.y0))>.8:icons.append([t.get_text(),a.get_label()])
    family=sorted({font_manager.FontProperties(fname=font_manager.findfont(t.get_fontproperties())).get_name() for t in texts})
    return dict(font_sizes_pt=fonts,ordinary_font_sizes_pt=ordinary_fonts,frequency_encoded_font_sizes_pt=frequency_fonts,font_families=family,clipped=clipped,collisions=collisions,icon_collisions=icons)


def audit_right_axes(fig):
    result={}
    for name,a in fig._twins.items():
        root_name=name.split('_probability')[0];bottom=fig._panels[root_name].spines['bottom']
        right=a.spines['right'];scale=25.4/fig.dpi
        verts=right.get_transform().transform(right.get_path().vertices)
        yend=verts[np.argmin(verts[:,1])]
        verts=bottom.get_transform().transform(bottom.get_path().vertices)
        xend=verts[np.argmax(verts[:,0])]
        gap=np.linalg.norm(yend-xend)*scale-(right.get_linewidth()+bottom.get_linewidth())/2*25.4/72
        assert gap>=.5,(name,gap)
        result[name]={'right_position':list(right.get_position()),
            'bottom_position':list(bottom.get_position()),'visible_corner_gap_mm':float(gap)}
    return result


def save(fig,out):
    out.mkdir(parents=True,exist_ok=True);qa=out/STEM/'qa';qa.mkdir(parents=True,exist_ok=True)
    report=text_qa(fig)
    (qa/'native_text_qa.json').write_text(json.dumps(report,indent=2)+'\n')
    # Internal diagnostic preview is allowed before the delivery gate.
    buf=BytesIO();fig.savefig(buf,format='png',dpi=DPI,facecolor='white');buf.seek(0)
    preview=qa/'layout_preview.png'
    with Image.open(buf) as im:im.convert('RGB').save(preview,dpi=(DPI,DPI))
    assert report['ordinary_font_sizes_pt']==[5,6,8],report['ordinary_font_sizes_pt']
    assert report['font_families']==['DejaVu Sans'],report['font_families']
    assert not report['clipped'],report['clipped']
    assert not report['collisions'],report['collisions']
    assert not report['icon_collisions'],report['icon_collisions']
    alignment=require_matplotlib_panel_alignment(fig,axes=list(fig._panels.values()),
        row_groups=[['a1','b1','c1','c1_reduction'],['a2','b2','c2','c2_reduction'],
                    ['e1','f1'],['e2','f2']],
        column_groups=[['a1','a2'],['b1','b2'],['c1','c2'],['c1_reduction','c2_reduction'],['e1','e2'],['f1','f2']],
        exemptions=[dict(panels=['a1','a2','c1','c2','c1_reduction','c2_reduction','e1','e2','g_effect'],
            checks=['panel-width','horizontal-gutter'],reason='Approved schematic-led three-tier layout: dual-axis curves25mm, ranks/KDE40mm, event28mm plus14mm summary, language effect42x14mm. Schematic and two native word clouds are non-Cartesian figure artists; all repeated task-specific quantitative roles retain their original dimensions.'),
            dict(panels=['a1','a2','b1','b2','e1','e2','f1','f2'],checks=['vertical-gutter'],
                reason='Task-pair pitch33mm in the upper tier and34mm in the middle tier is retained; bottom language effect is a separate role.')],
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,json_out=qa/'alignment.json')
    fig.canvas.draw()
    tick_endpoints=audit_tick_bounded_spines({**fig._panels,**fig._twins})
    (qa/'tick_endpoints.json').write_text(json.dumps(tick_endpoints,indent=2)+'\n')
    axis_spacing=audit_detached_axes(fig._panels)
    axis_spacing['dual_probability_right_spines']=audit_right_axes(fig)
    (qa/'detached_axes.json').write_text(json.dumps(axis_spacing,indent=2)+'\n')
    fig._breaks=audit_yaxis_breaks({n:a for n,a in fig._panels.items()})
    sources=protect_sources();source_hashes={}
    for folder in ['derived_behavior','derived_ei_language','derived_figure9']:
        for p in (PACKAGE/'data'/folder).glob('*.csv'):source_hashes[str(p.relative_to(PACKAGE))]=hashlib.sha256(p.read_bytes()).hexdigest()
    final=out/(STEM+'.png');preview.replace(final)
    with Image.open(final) as im:png=dict(mode=im.mode,pixels=im.size,dpi=im.info['dpi'])
    panels={n:dict(rect_mm=(np.array(a.bbox.extents)*25.4/fig.dpi).tolist(),
        xlabel=a.get_xlabel(),ylabel=a.get_ylabel(),xlim=list(a.get_xlim()),ylim=list(a.get_ylim()),
        visible_data_source='frozen derived tables') for n,a in fig._panels.items()}
    manifest=dict(status='PASS',png=png,size_mm=[W,H],font_sizes_pt=report['font_sizes_pt'],
        ordinary_font_sizes_pt=report['ordinary_font_sizes_pt'],frequency_encoded_font_sizes_pt=report['frequency_encoded_font_sizes_pt'],
        frequency_type_exception='User2026-10-14 continuous5–8pt words and size key',
        panel_letters='abcdefg',panels=panels,significance=fig._annotations,alignment=alignment['verdict'],
        y_axis_breaks=fig._breaks,source_protection=sources,plot_input_hashes=source_hashes,
        sha256=hashlib.sha256(final.read_bytes()).hexdigest(),backend='native Matplotlib; PNG-only',
        wordcloud_layout=fig._wordcloud_layout,
        ei_definition=fig._ei_definition,
        panel_mapping_from_previous=dict(a='a',b='b',c='c',d='e',e='f',f='g'),
        wordcloud_size_key=fig._frequency_size_key,
        tick_endpoints=tick_endpoints,
        wordcloud_colorbars=[dict(task=b['task'],rect_mm=(np.array(b['colorbar'].ax.bbox.extents)*25.4/fig.dpi).tolist(),relative_count_range=b['relative_count_range'],ticks=b['ticks'],label=b['label'],definition=b['definition']) for b in fig._frequency_colorbars],
        axis_corner_spacing=axis_spacing,
        notes=['No source PNGs scaled or inserted; all marks and words are native.',
            'Two native compact word clouds preserve all14source terms/counts; fontsize=5+3sqrt(count/taskmax) continuously encodes relative frequency. User-approved local type exception; ordinary text5/6/8pt. No Word legend or size key; constant task hues.',
            'New d is an explanatory EI schematic, not a fitted predictor. Previous d/e/f become e/f/g; frozen cache names retain their source aliases.',
            'All quantitative plot sizes, source means, intervals and P values remain unchanged. Height grows by900native pixels to keep upper a-c at their original physical positions.',
            'Panel c uses the supplied Figure9 all-DIS-trial event curves and participant reductions; source two-task Mann-Whitney/BH comparisons are reproduced and frozen. Both positive-origin reduction axes have omission marks.',
            'Panel c displays trial offsets -3 to +3; complete frozen curves retain -5 to +5. Delta max. dimension uncertainty means Umax(0)-Umax(+3), unchanged from the source reduction.',
            'Panel f retains every existing KDE point and its paired participant-mean squared-EI Wilcoxon/BH stars; source correction-family alias Panel e is retained in frozen statistics.',
            'The former language panel is compressed vertically into a bottom strip without changing its42x14mm effect plot or any of the14words/counts/font sizes.',
            'Panel g is one centered compact group: effect x40mm, cloud centers103/139mm, equal6mm component gaps; g letter and source icon follow its left edge.',
            'Source fig3_rebuilt.png and its whole package are protected; this candidate does not replace the source.'])
    (out/STEM/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'PASS','png':str(final),'size_mm':[W,H],'pixels':png['pixels'],'fonts':report['font_sizes_pt']}))


def build_figure():
    fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=DPI);fig._panels={};fig._twins={};fig._annotations=[]
    top_progress(fig);rank(fig);onset(fig)
    draw_ei_definition(fig,W,H,ink=INK,purple=DIS,gray=OTHER)
    ei_progress(fig);density(fig);language(fig)
    for letter,x,y in [('a',3,1),('b',53,1),('c',113,1),('d',3,80),('e',64,80),('f',121,80),('g',25,160)]:
        if letter=='d':
            ft(fig,x,y,letter,size=8,weight='bold').set_gid('nhb-panel-letter')
        else:
            heading(fig,letter,x,y,'survey' if letter=='g' else 'task_behavior')
    # The line style already identifies the dual-axis curves; short legends
    # focus on comparisons whose meaning cannot be inferred from the axes.
    key(fig,146,1,[Line2D([],[],color=DIS,lw=1.1),Line2D([],[],color=OTHER,lw=1.1)],['DIS trial','Matched others'])
    key(fig,140,80,[Line2D([],[],color=DIS,lw=1.1),Line2D([],[],color=OTHER,lw=1.1)],['DIS','Others'])
    for a in fig._panels.values():detach_axes(a)
    for a in fig._twins.values():detach_axes(a,right=True)
    for a in [*fig._panels.values(),*fig._twins.values()]:bound_spines_to_ticks(a)
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=PACKAGE.parent)
    p.add_argument('--replace-draft',action='store_true');args=p.parse_args()
    if (args.output_dir/(STEM+'.png')).exists() and not args.replace_draft:raise FileExistsError('Existing figure; use --replace-draft only for this rebuilt version.')
    protect_sources()
    fig=build_figure()
    save(fig,args.output_dir);plt.close(fig)

if __name__=='__main__':main()
