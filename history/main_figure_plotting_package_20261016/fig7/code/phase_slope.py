"""Draw phase slopes from the joint seven-parameter free-beta fits."""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import NullLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACKAGE=Path(__file__).resolve().parents[1]
layout = None  # Supplied by the edition renderer.


def draw_slope(main,statistics):
    participants=pd.read_csv(PACKAGE/'data/participant_temperature_slope.csv',dtype={'subject':str})
    tests=pd.read_csv(PACKAGE/'data/slope_phase_tests.csv')
    assert len(participants)==156 and participants.subject.nunique()==105
    axis=main.add_axes((.495,.075,.215,.215))
    purple=layout.DGEM_PURPLE
    phases=['P1','P2','P2-only']
    labels=['3D','4D-E','4D-NE']
    paired=participants.loc[participants.phase.isin(['P1','P2'])].pivot(index='subject',columns='phase',values='temperature_slope')
    assert len(paired)==51 and paired.notna().all().all()
    counts=[]
    for i,phase in enumerate(phases):
        group=participants.loc[participants.phase.eq(phase)].sort_values('subject')
        values=group.temperature_slope.to_numpy(float)
        counts.append(len(values))
        assert (values>0).all()
        axis.boxplot([values],positions=[i],widths=.45,showfliers=False,
            patch_artist=True,zorder=3,
            boxprops={'facecolor':matplotlib.colors.to_rgba(purple,.17),'edgecolor':purple,'linewidth':.8},
            medianprops={'color':purple,'linewidth':1.1},
            whiskerprops={'color':purple,'linewidth':.7},
            capprops={'color':purple,'linewidth':.7})
    axis.set_yscale('log');axis.set_ylim(.007,50);axis.set_xlim(-.45,2.45)
    axis.set_yticks([.01,.1,1,10],['0.01','0.1','1','10'])
    axis.yaxis.set_minor_locator(NullLocator())
    axis.set_xticks([0,1,2],labels)
    axis.set_xlabel('Task phase');axis.set_ylabel('Temperature slope')
    axis.spines['bottom'].set_bounds(0,2)
    from matplotlib.transforms import blended_transform_factory
    bracket_transform=blended_transform_factory(axis.transData,axis.transAxes)
    brackets=[];stars=[]
    for left,right,height in [(0,1,.73),(1,2,.73),(0,2,.93)]:
        row=tests.loc[tests.phase_a.eq(phases[left])&tests.phase_b.eq(phases[right])]
        assert len(row)==1
        star=str(row.iloc[0].significance)
        line,=axis.plot([left,left,right,right],[height-.018,height,height,height-.018],
                  transform=bracket_transform,color=layout.INK,linewidth=.65,clip_on=False)
        text=axis.text((left+right)/2,height+.018,star,ha='center',va='bottom',
                  transform=bracket_transform,fontsize=6,
                  fontweight='normal' if star=='ns' else 'bold',color=layout.INK)
        brackets.append(line);stars.append(text)
    layout._style_data_axis(axis)
    axis._slope_display_spec={'source_counts':counts,'brackets':brackets,'stars':stars,
        'individual_points_displayed':False,'fliers_displayed':False,
        'variant':'free_bonus','beta_init_fitted':True}
    return axis

