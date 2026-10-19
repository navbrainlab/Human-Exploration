#!/usr/bin/env python3
"""Native, 600-dpi Fig1 with frozen source-backed plotting inputs; legacy Fig2 routes to archive.

Missing source panels remain visibly marked; no synthetic observations are used.
Rendering reads frozen CSVs and never runs fitting, replay or simulations.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent
CSV = ROOT / "csv"
QA = ROOT / "qa"
QA.mkdir(exist_ok=True)
SKILL_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment

COLORS = ["#079E6C", "#0965C0", "#14BAEC"]
LABELS = ["3D-NE", "4D-E", "4D-NE"]
TASK_CONFIG = {
    1:dict(groups=["P1","P2","P2-only"],column="phase",colors=COLORS,labels=LABELS,
           positions=[0,1,2]),
    2:dict(groups=["FDS-Obs","FDS-NoObs"],column="plot_group",
           colors=["#E67E28","#F7BA00"],labels=["Obs","NObs"],positions=[.2,.8]),
}
PURPLE = "#7B3FB2"
INK = "#20242B"
GRAY = "#68727C"
FRAME = "#CED4DB"
WIDTH_MM = 180
plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["DejaVu Sans", "Arial", "Helvetica"], "font.size":6,
    "axes.labelsize":6, "axes.titlesize":7, "xtick.labelsize":6,
    "ytick.labelsize":6, "legend.fontsize":6, "axes.linewidth":.6,
    "xtick.major.width":.6, "ytick.major.width":.6,
    "xtick.major.size":2, "ytick.major.size":2,
    "text.color":INK, "axes.labelcolor":INK, "xtick.color":INK,
    "ytick.color":INK, "axes.grid":False, "savefig.dpi":600})


def read(name):
    return pd.read_csv(CSV / name)


def make_canvas(height):
    fig = plt.figure(figsize=(WIDTH_MM/25.4, height/25.4), dpi=600, facecolor="white")
    fig._height_mm = height
    fig._panels = {}
    fig._rows = []
    fig._columns = []
    return fig


def axis(fig, name, x, y, width, height, *, schematic=False):
    ax = fig.add_axes([x/180, 1-(y+height)/fig._height_mm, width/180, height/fig._height_mm])
    ax.set_label(name)
    fig._panels[name] = ax
    if schematic:
        ax.set_xlim(0,width)
        ax.set_ylim(height,0)
        ax.axis("off")
    else:
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(pad=2)
        ax.yaxis.labelpad = 3
        ax.xaxis.labelpad = 3
    return ax


def figure_text(fig, x, y, text, *, size=6, weight="normal", color=INK, ha="left"):
    return fig.text(x/180, 1-y/fig._height_mm, text, fontsize=size,
                    fontweight=weight, color=color, ha=ha, va="top")


def panel_heading(fig, letter, x, y, title):
    figure_text(fig,x,y,letter,size=8,weight="bold")
    figure_text(fig,x+4.5,y+.3,title,size=7)


def box(ax, x, y, width, height, label=None, *, color="#F5F7FA", edge=FRAME, size=6):
    patch = Rectangle((x,y),width,height, facecolor=color, edgecolor=edge, linewidth=.6)
    ax.add_patch(patch)
    if label:
        ax.text(x+width/2,y+height/2,label,fontsize=size,ha="center",va="center",linespacing=1.25)
    return patch


def arrow(ax, start, end, *, color=GRAY, lw=.8):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=6,
                                linewidth=lw,color=color,shrinkA=0,shrinkB=0))


def source_gap(ax, message):
    if ax.axison:
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
        box(ax,.015,.015,.97,.97,color="#F8F9FB",edge="#D9DDE3")
        ax.text(.5,.62,"Source data missing",ha="center",va="center",fontsize=7,color=GRAY)
        ax.text(.5,.39,message,ha="center",va="center",fontsize=6,color=GRAY,linespacing=1.3)
    else:
        x0,x1=ax.get_xlim();y1,y0=ax.get_ylim()
        box(ax,x0+.4,y0+.4,x1-x0-.8,y1-y0-.8,color="#F8F9FB")
        ax.text((x0+x1)/2,y0+(y1-y0)*.38,"Source data missing",ha="center",va="center",fontsize=7,color=GRAY)
        ax.text((x0+x1)/2,y0+(y1-y0)*.62,message,ha="center",va="center",fontsize=6,color=GRAY,linespacing=1.3)


def food_icon(ax,cx,cy,w,h,kind,value):
    col=["#D89343", "#4D86B7", "#9B77B6"][int(value)%3]
    if kind==0:
        ax.add_patch(Ellipse((cx,cy),w*.8,h*.6,facecolor=col,edgecolor=col,lw=.25))
        ax.plot([cx-w*.43,cx,cx+w*.43],[cy,cy+h*.4,cy],color=col,lw=.45)
    elif kind==1:
        ax.add_patch(Rectangle((cx-w*.3,cy-h*.45),w*.6,h*.9,facecolor=col,edgecolor=col,lw=.25))
    elif kind==2:
        ax.add_patch(Polygon([(cx-w*.43,cy+h*.4),(cx+w*.43,cy+h*.4),(cx,cy-h*.45)],facecolor=col,edgecolor=col,lw=.25))
    else:
        ax.add_patch(Ellipse((cx,cy),w*.7,h*.8,facecolor=col,edgecolor=col,lw=.25))
        ax.plot([cx,cx+w*.2],[cy-h*.4,cy-h*.65],color="#47805A",lw=.45)


def meal(ax,x,y,w=5,h=4,values=(0,1,2,0)):
    """Four native pictograms: staple, beverage, dessert, fruit."""
    box(ax,x,y,w,h,color="white",edge="#D4DBE3")
    for k,v in enumerate(values):
        food_icon(ax,x+w*(.15+.235*k),y+h*.55,w*.2,h*.35,k,v)


def room(ax,x,y,w=6,h=5,feature=0,other_feature=0):
    box(ax,x,y,w,h,color="#F8F8F5",edge="#BEC8D3")
    shades=["#477BAC","#A77556","#70966C","#9B79AC"]
    col=shades[feature%4]
    # Sofa, carpet, table, plant, painting, all schematic.
    ax.add_patch(Rectangle((x+w*.15,y+h*.33),w*.48,h*.32,facecolor=col,edgecolor=INK,lw=.18))
    ax.add_patch(Rectangle((x+w*.09,y+h*.44),w*.09,h*.27,facecolor=col,edgecolor=INK,lw=.18))
    ax.add_patch(Rectangle((x+w*.6,y+h*.44),w*.09,h*.27,facecolor=col,edgecolor=INK,lw=.18))
    r,c=divmod(int(other_feature),4)
    carpet=["#CFBF9B","#B6CBC4","#C8BED5","#BECDD9"][(r+2*c)%4]
    painting=["#C5D2DE","#DAC4AF","#B9CFBF","#D3C3D9"][c%4]
    ax.add_patch(Ellipse((x+w*.43,y+h*.83),w*(.55+.045*c),h*.18,facecolor=carpet,edgecolor="none"))
    ax.plot([x+w*.32,x+w*.53],[y+h*.73,y+h*.73],color=shades[(2*r+c)%4],lw=.65)
    ax.plot([x+w*.35,x+w*.35],[y+h*.73,y+h*.88],color="#5B5B5B",lw=.45)
    ax.add_patch(Rectangle((x+w*.34,y+h*.07),w*.28,h*.18,facecolor=painting,edgecolor="#888888",lw=.2))
    ax.add_patch(Rectangle((x+w*.83,y+h*.7),w*.09,h*.19,facecolor="#A88968",edgecolor="none"))
    ax.add_patch(Ellipse((x+w*.87,y+h*.51),w*.16,h*(.23+.025*r),facecolor=shades[r%4],edgecolor="none"))


def task1_design(ax):
    ax.text(1,3,"Dimensions",fontsize=6,va="center")
    for k,name in enumerate(["Staple", "Beverage", "Dessert", "Fruit"]):
        for v in range(3):
            food_icon(ax,23+15*k+3*v,2.5,2.1,2.2,k,v)
        ax.text(26.5+15*k,7,name,ha="center",fontsize=6,va="top")
    ax.text(12,13,"9 meal bundles",ha="center",fontsize=6)
    ax.text(42,13,"Rank into 3 rows",ha="center",fontsize=6)
    bundles=[(r,c,(r+c)%3,(2*r+c)%3) for r in range(3) for c in range(3)]
    stimulus_order=[4,0,8,6,2,3,1,7,5]
    for r in range(3):
        for c in range(3):
            meal(ax,3+6*c,16+6*r,5.3,4.8,bundles[stimulus_order[r*3+c]])
            meal(ax,35+6*c,16+6*r,5.3,4.8,bundles[r*3+c])
    arrow(ax,(23,25),(32,25))
    arrow(ax,(55,25),(63,25))
    box(ax,65,19,16,12,"Feedback\n0–100",color="#EAF2F9",edge="#9FB3C7")
    for r in range(3):
        ax.text(53,18.4+6*r,str(r+1),fontsize=6,va="center",ha="left",color=GRAY)
    ax.text(42,37,"Two relevant dimensions; rules learned from feedback",ha="center",fontsize=6)
    ax.text(42,42,"3D / 4D · three feature values per dimension",ha="center",fontsize=6,color=GRAY)


def task2_design(ax):
    ax.text(42,3,"Sofa · Table · Carpet · Plant · Painting",ha="center",va="center",fontsize=6)
    ax.text(42,8,"5 dimensions · four feature values per dimension",ha="center",fontsize=6,color=GRAY)
    ax.text(16,13,"16 scene images",ha="center",fontsize=6)
    ax.text(49,13,"Choose 4 images",ha="center",fontsize=6)
    for r in range(4):
        for c in range(4):
            room(ax,1+7.5*c,16+5.5*r,6.8,4.8,(r+c)%4,r*4+c)
    for r in range(2):
        for c in range(2):
            room(ax,42+8*c,19+8*r,7,6,(r+c)%4,r*4+c)
    arrow(ax,(32,26),(39,26))
    arrow(ax,(59,26),(64,26))
    box(ax,66,20,16,12,"Feedback\n0–100",color="#FFF3E6",edge="#C8AE8A")
    ax.text(42,42,"Learn which dimensions and features affect the score",ha="center",fontsize=6)


def task1_groups(ax):
    ax.text(1,5,"Practice group\nn = 51",va="center",fontsize=6,linespacing=1.3)
    box(ax,24,1,18,9,"3D-NE\n≤30 trials",color="#E8F4EE",edge=COLORS[0])
    arrow(ax,(43,5.5),(47,5.5))
    box(ax,48,1,18,9,"4D-E\n≤50 trials",color="#E7F0FA",edge=COLORS[1])
    arrow(ax,(67,5.5),(71,5.5))
    box(ax,72,1,11,20,"Survey\n+\ninterview",color="#F6F7F8")
    ax.text(1,16,"No-practice group\nn = 54",va="center",fontsize=6,linespacing=1.3)
    box(ax,48,12,18,9,"4D-NE\n≤50 trials",color="#E8F7FC",edge=COLORS[2])
    arrow(ax,(67,16.5),(71,16.5))


def task2_groups(ax):
    box(ax,1,1,22,9,"Observation\nObs",color="#FFF0E4",edge="#E67E28")
    box(ax,1,12,22,9,"No observation\nNObs",color="#FFF8DF",edge="#DDA400")
    for y in [5.5,16.5]:
        arrow(ax,(24,y),(30,y))
    box(ax,31,1,25,20,"Task 2\n≤60 trials",color="#F4F6F8")
    arrow(ax,(57,11),(63,11))
    box(ax,64,1,18,20,"Survey\n+ interview",color="#F6F7F8")


def p_label(p):
    return "P < 0.001" if p < .001 else f"P = {p:.3f}"


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def nature_task1_axis(ax, *, categorical=False, categories=3, positions=None):
    """Use the final Fig. 5/6 stroke and detached-category-axis conventions."""
    ink="#252B30"
    for key in ["left","bottom"]:
        ax.spines[key].set_linewidth(.70)
        ax.spines[key].set_color(ink)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(axis="both",width=.65,length=2.5,pad=2,labelsize=6,colors=ink)
    ax.xaxis.label.set_color(ink)
    ax.yaxis.label.set_color(ink)
    if categorical:
        bounds=(0,categories-1) if positions is None else (positions[0],positions[-1])
        ax.spines["bottom"].set_bounds(*bounds)
        ax.spines["bottom"].set_position(("outward",3))
        ax.tick_params(axis="x",length=2.3,pad=2.5)


def nature_comparison(ax,left,right,p):
    """Shared physical annotation band, matching Fig. 5's final renderer."""
    ax.plot([left,left,right,right],[1.035,1.060,1.060,1.035],
            transform=ax.get_xaxis_transform(),color="#252B30",lw=.70,
            solid_capstyle="butt",clip_on=False)
    ax.text((left+right)/2,1.075,stars(p),transform=ax.get_xaxis_transform(),
            ha="center",va="bottom",fontsize=6,fontweight="bold",color="#252B30")


def bracket(ax,left,right,y,text,height):
    ax.plot([left,left,right,right],[y-height,y,y,y-height],lw=.65,color=INK,clip_on=False)
    ax.text((left+right)/2,y+height*.3,text,ha="center",va="bottom",fontsize=6)


def box_points(ax,values,labels,colors,*,seed=42,positions=None):
    # Randomness affects x-coordinate display jitter only; all measured y-values are retained.
    positions=np.arange(len(values)) if positions is None else np.asarray(positions)
    bp=ax.boxplot(values,positions=positions,widths=.28,patch_artist=True,showfliers=False,
        medianprops={"color":INK,"linewidth":.9},boxprops={"linewidth":.6},
        whiskerprops={"linewidth":.6},capprops={"linewidth":.6})
    rng=np.random.default_rng(seed)
    for x,v,color,patch in zip(positions,values,colors,bp["boxes"]):
        patch.set_facecolor(matplotlib.colors.to_rgba(color,.35))
        v=np.asarray(v)
        ax.scatter(x-.22+rng.uniform(-.055,.055,len(v)),v,s=5.5,color=color,alpha=.68,
                   edgecolor="white",linewidth=.18,zorder=3)
    ax.set_xticks(positions,labels)
    ax.set_xlim(positions.min()-.5,positions.max()+.5)


def outcome_statistics():
    return json.loads((CSV/"fig1_all_cohort_statistics.json").read_text())


def learning(ax,task=1):
    config=TASK_CONFIG[task]
    curves=read("fig1_learning_display.csv")
    curves=curves[curves.task.eq(task)]
    for group,color,label in zip(config["groups"],config["colors"],config["labels"]):
        part=curves[curves.group.eq(group)]
        x,m,se=part.trial.to_numpy(),part.mean_smooth.to_numpy(),part["sem"].to_numpy()
        ax.fill_between(x,m-se,m+se,color=color,alpha=.18,lw=0,zorder=1)
        ax.plot(x,m,color=color,lw=1.10,label=label,zorder=2)
    horizon=int(curves.trial.max())
    lower=float(np.floor(((curves.mean_smooth-curves["sem"]).min()-1)/5)*5)
    upper=float(np.ceil(((curves.mean_smooth+curves["sem"]).max()+1)/5)*5)
    yticks=np.unique(np.round(np.linspace(lower,upper,4)/5)*5)
    ax.set(xlim=(1,horizon),ylim=(lower,upper),xticks=[1,15,30,50] if task==1 else [1,20,40,60],
           yticks=yticks,xlabel="Trial",ylabel="Mean score")
    nature_task1_axis(ax)
    ax.legend(loc="lower right",frameon=False,handlelength=1.5,handletextpad=.4,
              labelspacing=.2,borderpad=.15,fontsize=6)


def perfect(ax,task=1):
    """Horizontal proportion points, with Wilson confidence intervals."""
    config=TASK_CONFIG[task]
    frame=read(f"fig1_task{task}_perfect_rate.csv")
    positions=np.array([2.,1.,0.]) if task==1 else np.array([.8,.2])
    ax._summary_artists=[]
    for position,color,(_,row) in zip(positions,config["colors"],frame.iterrows()):
        interval=ax.errorbar(row.rate*100,position,
            xerr=np.array([[(row.rate-row.lower)*100],[(row.upper-row.rate)*100]]),
            fmt="none",ecolor=INK,elinewidth=.8,capsize=2,capthick=.7,zorder=2)
        marker=ax.scatter(row.rate*100,position,s=18,color=color,edgecolor=INK,linewidth=.35,zorder=3)
        ax._summary_artists.append((marker,interval[2][0]))
    ax.set(xlim=(0,100),ylim=(-.5,2.5) if task==1 else (-.5,1.5),
           xticks=[0,25,50,75,100],yticks=positions,yticklabels=config["labels"],xlabel="Full score (%)")
    nature_task1_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y",length=0)


def risk_difference(ax,task=1):
    """Risk differences in percentage points, preserving participant pairing."""
    frame=read("fig1_risk_differences.csv")
    frame=frame[frame.task.eq(task)]
    positions=np.arange(len(frame))[::-1]
    ax.axvline(0,color=GRAY,lw=.65,ls=(0,(2,2)),zorder=0)
    ax._summary_artists=[]
    for position,(_,row) in zip(positions,frame.iterrows()):
        interval=ax.errorbar(row.difference*100,position,
            xerr=np.array([[(row.difference-row.lower)*100],[(row.upper-row.difference)*100]]),
            fmt="none",ecolor=INK,elinewidth=.8,capsize=2,capthick=.7,zorder=2)
        marker=ax.scatter(row.difference*100,position,s=17,color=COLORS[1] if task==1 else "#E67E28",
                          edgecolor=INK,linewidth=.35,zorder=3)
        ax._summary_artists.append((marker,interval[2][0]))
    ax.set(xlim=(-20,60) if task==1 else (-30,50),ylim=(-.5,len(frame)-.5),
           xticks=[-20,0,20,40,60] if task==1 else [-20,0,20,40],
           yticks=positions,yticklabels=frame.label.tolist(),xlabel="Difference (pp)")
    nature_task1_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y",length=0)


def cumulative_completion(ax,task=1):
    """All-cohort first-full-score process; failures are administratively censored."""
    config=TASK_CONFIG[task]
    frame=read("fig1_all_cohort_cumulative.csv")
    frame=frame[frame.task.eq(task)]
    ax._cdf_lines=[];ax._cdf_bands=[];ax._cdf_censors=[]
    for group,color,label in zip(config["groups"],config["colors"],config["labels"]):
        part=frame[frame.group.eq(group)]
        band=ax.fill_between(part.trial.to_numpy(),part.lower.to_numpy()*100,part.upper.to_numpy()*100,
                             step="post",color=color,alpha=.13,lw=0,zorder=1)
        line,=ax.step(part.trial,part.cumulative_proportion*100,where="post",color=color,lw=1.10,label=label,zorder=2)
        censor,=ax.plot(part.trial.iloc[-1],part.cumulative_proportion.iloc[-1]*100,
                       marker="|",markersize=5,markeredgewidth=.8,color=color,ls="none",clip_on=False,zorder=3)
        ax._cdf_lines.append(line);ax._cdf_bands.append(band);ax._cdf_censors.append(censor)
    horizon=int(frame.trial.max())
    ax.set(xlim=(0,horizon),ylim=(0,100),xticks=[0,15,30,50] if task==1 else [0,20,40,60],
           yticks=[0,25,50,75,100],xlabel="Trial",ylabel="Reached full score (%)")
    nature_task1_axis(ax)
    ax.legend(loc="upper left",frameon=False,handlelength=1.5,handletextpad=.4,
              labelspacing=.2,borderpad=.15,fontsize=6)
    test=next(t for t in outcome_statistics()["tests"] if t["id"]==f"survival_task{task}")
    comparison="4D-E vs 4D-NE" if task==1 else "Obs vs NObs"
    ax.text(.97,.98,comparison+"\n"+r"$P$ (Holm) = "+f"{test['p_holm']:.3f}",
            transform=ax.transAxes,ha="right",va="top",fontsize=6)


def audit_efg_style(fig):
    """Inspect actual native artists, full denominators, uncertainty and text geometry."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    axes={k:fig._panels[k] for k in ["e","f","f_delta","g","h","i","i_delta","j"]}
    stats=outcome_statistics()
    display=read("fig1_learning_display.csv")
    cumulative=read("fig1_all_cohort_cumulative.csv")
    differences=read("fig1_risk_differences.csv")
    learning_limits={};denominators={};events={}
    for task,learning_key,proportion_key,difference_key,cdf_key in [(1,"e","f","f_delta","g"),(2,"h","i","i_delta","j")]:
        config=TASK_CONFIG[task]
        curve=display[display.task.eq(task)]
        for group,line,band in zip(config["groups"],axes[learning_key].lines,axes[learning_key].collections):
            part=curve[curve.group.eq(group)]
            assert np.allclose(line.get_xdata(),part.trial)
            assert np.allclose(line.get_ydata(),part.mean_smooth)
            values=band.get_paths()[0].vertices[:,1]
            assert np.isclose(values.min(),(part.mean_smooth-part["sem"]).min())
            assert np.isclose(values.max(),(part.mean_smooth+part["sem"]).max())
            assert values.min()>axes[learning_key].get_ylim()[0] and values.max()<axes[learning_key].get_ylim()[1]
        assert axes[learning_key].get_xlabel()=="Trial" and not axes[learning_key].texts
        learning_limits[learning_key]=list(axes[learning_key].get_ylim())
        rates=read(f"fig1_task{task}_perfect_rate.csv")
        for (marker,interval),(_,row) in zip(axes[proportion_key]._summary_artists,rates.iterrows()):
            assert np.isclose(marker.get_offsets()[0,0],row.rate*100)
            segment=interval.get_segments()[0]
            assert np.allclose(segment[:,0],[row.lower*100,row.upper*100])
        effect=differences[differences.task.eq(task)]
        for (marker,interval),(_,row) in zip(axes[difference_key]._summary_artists,effect.iterrows()):
            assert np.isclose(marker.get_offsets()[0,0],row.difference*100)
            assert np.allclose(interval.get_segments()[0][:,0],[row.lower*100,row.upper*100])
        denominators[cdf_key]=[];events[cdf_key]=[]
        for group,line,band,censor in zip(config["groups"],axes[cdf_key]._cdf_lines,axes[cdf_key]._cdf_bands,axes[cdf_key]._cdf_censors):
            part=cumulative[cumulative.task.eq(task)&cumulative.group.eq(group)]
            assert line.get_drawstyle()=="steps-post"
            assert np.allclose(line.get_xdata(),part.trial)
            assert np.allclose(line.get_ydata(),part.cumulative_proportion*100)
            assert np.isclose(censor.get_xdata()[0],part.trial.iloc[-1])
            assert np.isclose(censor.get_ydata()[0],part.cumulative_proportion.iloc[-1]*100)
            denominators[cdf_key].append(int(part.n.iloc[0]));events[cdf_key].append(int(part.n_fullscore.iloc[-1]))
            values=band.get_paths()[0].vertices[:,1]
            assert values.min()>=0 and values.max()<=100
        assert axes[cdf_key].get_xlabel()=="Trial"
        # Check annotation clearance against the displayed CI, not just other text.
        for annotation in axes[cdf_key].texts:
            bounds=axes[cdf_key].transData.inverted().transform(annotation.get_window_extent(renderer).get_points())
            for band in axes[cdf_key]._cdf_bands:
                vertices=band.get_paths()[0].vertices
                within=vertices[(vertices[:,0]>=bounds[0,0])&(vertices[:,0]<=bounds[1,0])]
                if len(within):assert bounds[0,1]>within[:,1].max()+.5
    assert denominators=={"g":[51,51,54],"j":[40,40]}
    assert events=={"g":[17,32,22],"j":[18,12]}
    assert learning_limits=={"e":[50.,100.],"h":[45.,90.]}
    collisions=[];all_texts=[]
    for key,ax in axes.items():
        texts=[ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels(),*ax.texts]
        legend=ax.get_legend()
        if legend is not None:texts.extend(legend.get_texts())
        texts=[t for t in texts if t.get_visible() and t.get_text().strip()]
        all_texts.extend((key,t) for t in texts)
    all_texts.extend(("heading",t) for t in fig.texts if t.get_position()[1]<1-95/fig._height_mm)
    for idx,(key,text) in enumerate(all_texts):
        a=text.get_window_extent(renderer)
        for key2,text2 in all_texts[idx+1:]:
            b=text2.get_window_extent(renderer)
            if min(a.x1,b.x1)-max(a.x0,b.x0)>.5 and min(a.y1,b.y1)-max(a.y0,b.y0)>.5:
                collisions.append([key,key2,text.get_text(),text2.get_text()])
    assert not collisions,collisions
    geometry={key:dict(left_mm=ax.get_position().x0*180,top_mm=(1-ax.get_position().y1)*fig._height_mm,
        width_mm=ax.get_position().width*180,height_mm=ax.get_position().height*fig._height_mm) for key,ax in axes.items()}
    assert all(np.isclose(v["width_mm"],39) for v in geometry.values())
    assert all(np.isclose(geometry[k]["height_mm"],43) for k in ["e","g","h","j"])
    assert all(np.isclose(geometry[k]["height_mm"],18) for k in ["f","i"])
    assert all(np.isclose(geometry[k]["height_mm"],10) for k in ["f_delta","i_delta"])
    report=dict(status="PASS",learning_actual_trials=True,learning_limits=learning_limits,
        learning_mean_values_preserved=True,learning_uncertainty="SEM after participant-level smoothing",
        proportion_display="points and Wilson95%CI",risk_difference_display="points and paired/independent bootstrap95%CI",
        cumulative_display="all-cohort first-full-score proportions, fixed-horizon right censoring",
        cumulative_denominators=denominators,cumulative_event_counts=events,
        right_censoring_not_completed_events=True,geometry_mm=geometry,
        semantic_layout="e/g/h/j main axes; f/i each have proportion and difference subpanels",
        tests=stats["tests"],annotation_text_collisions=collisions,cumulative_annotation_ci_clearance=True)
    (QA/"fig1_efg_native_style.json").write_text(json.dumps(report,indent=2))
    (QA/"fig1_hij_native_style.json").write_text(json.dumps(report,indent=2))


def render_figure1():
    fig=make_canvas(232)
    figure_text(fig,3,2,"Task structure, learning and full-score attainment",color=GRAY)
    for letter,x,title in [("a",3,"Task 1 | meal ranking"),("b",94,"Task 2 | scene selection")]:
        panel_heading(fig,letter,x,7,title)
    a=axis(fig,"a",3,14,83,45,schematic=True);task1_design(a)
    b=axis(fig,"b",94,14,83,45,schematic=True);task2_design(b)
    panel_heading(fig,"c",3,63,"Task 1 | transfer from prior practice")
    panel_heading(fig,"d",94,63,"Task 2 | prior observation")
    c=axis(fig,"c",3,70,83,23,schematic=True);task1_groups(c)
    d=axis(fig,"d",94,70,83,23,schematic=True);task2_groups(d)
    xs=[13,68,123];heading_xs=[x-10 for x in xs]
    titles=["Task 1 | learning","Full-score outcomes","First full score"]
    for letter,x,title in zip("efg",heading_xs,titles):panel_heading(fig,letter,x,99,title)
    e=axis(fig,"e",xs[0],108,39,43);learning(e)
    f=axis(fig,"f",xs[1],108,39,18);perfect(f)
    fd=axis(fig,"f_delta",xs[1],142,39,10);risk_difference(fd)
    g=axis(fig,"g",xs[2],108,39,43);cumulative_completion(g)
    titles=["Task 2 | learning","Full-score outcomes","First full score"]
    for letter,x,title in zip("hij",heading_xs,titles):panel_heading(fig,letter,x,166,title)
    h=axis(fig,"h",xs[0],175,39,43);learning(h,task=2)
    i=axis(fig,"i",xs[1],175,39,18);perfect(i,task=2)
    idelta=axis(fig,"i_delta",xs[1],209,39,10);risk_difference(idelta,task=2)
    j=axis(fig,"j",xs[2],175,39,43);cumulative_completion(j,task=2)
    fig._rows=[["a","b"],["c","d"],["e","g"],["h","j"]]
    fig._columns=[["e","h"],["f","i"],["f_delta","i_delta"],["g","j"]]
    audit_efg_style(fig)
    finish(fig,"fig1_rebuilt")


















def render_figure2():
    """Compatibility route: the legacy Fig2 implementation lives in waste."""
    import subprocess
    archived=ROOT.parent.parent/'discarded_figures/fig2_rebuilt/code/render_figures.py'
    subprocess.run([sys.executable,str(archived),'--figure','2'],check=True)


def finish(fig,name,width_exemptions=None,nominal_font_sizes=(6,7,8),alignment_exemptions=None):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=[artist for artist in fig.findobj(Text) if artist.get_visible() and artist.get_text().strip()]
    sizes=sorted({round(t.get_fontsize(),3) for t in texts})
    # Figure-specific typography can reuse this export path without changing
    # the established default for the original Fig. 1/2 renderers.
    assert sizes == list(nominal_font_sizes), sizes
    bounds=[]
    canvas=fig.bbox
    for text in texts:
        bbox=text.get_window_extent(renderer)
        if bbox.x0 < -.5 or bbox.y0 < -.5 or bbox.x1 > canvas.x1+.5 or bbox.y1 > canvas.y1+.5:
            bounds.append(dict(text=text.get_text(),bbox=[bbox.x0,bbox.y0,bbox.x1,bbox.y1]))
    assert not bounds, bounds
    kwargs=dict(axes=list(fig._panels.values()), row_groups=fig._rows, column_groups=fig._columns,
                json_out=QA/f"{name}_alignment.json")
    # Pass explicit groups; standalone add_axes has no implicit grid metadata.
    if width_exemptions:
        kwargs["exemptions"]=[dict(panel=key,reason="Asymmetric draft panel allocation; semantic half-panels retained.") for key in width_exemptions]
    if alignment_exemptions:
        kwargs["exemptions"]=kwargs.get("exemptions",[])+alignment_exemptions
    report=require_matplotlib_panel_alignment(fig,**kwargs)
    temp=QA/f"{name}_render.png"
    fig.savefig(temp,dpi=600,facecolor="white")
    with Image.open(temp) as source:
        source.convert("RGB").save(OUT/f"{name}.png",dpi=(600,600))
    temp.unlink()
    with Image.open(OUT/f"{name}.png") as result:
        dimensions=result.size;dpi=result.info["dpi"]
        assert result.mode=="RGB" and all(abs(v-600)<.05 for v in dpi)
    verification=dict(status="full_source_quantitative_panels" if name=="fig1_rebuilt" else "draft_missing_source_data",dpi=dpi,pixel_dimensions=dimensions,
                      nominal_font_sizes_pt=sizes,visible_text_count=len(texts),text_inside_canvas=True,
                      measured_width_mm=fig.get_figwidth()*25.4,
                      panel_alignment=report["verdict"],sha256=hashlib.sha256((OUT/f"{name}.png").read_bytes()).hexdigest())
    (QA/f"{name}_verification.json").write_text(json.dumps(verification,indent=2))
    preview=OUT/f"{name}.png"
    with Image.open(preview) as im:
        im.thumbnail((1200,1500));im.save(QA/f"{name}_preview.png")
    plt.close(fig)
    print(name,dimensions,sizes,report["verdict"])


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure",choices=["1","2","all"],default="1")
    selection=parser.parse_args().figure
    if selection in ["1","all"]:
        render_figure1()
    if selection in ["2","all"]:
        render_figure2()
