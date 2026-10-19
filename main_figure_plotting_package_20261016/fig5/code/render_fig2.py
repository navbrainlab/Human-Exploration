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

PACKAGE=Path(__file__).resolve().parents[1]/'data/old2'
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













