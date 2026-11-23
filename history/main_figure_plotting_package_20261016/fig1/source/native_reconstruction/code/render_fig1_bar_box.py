#!/usr/bin/env python3
"""Current compact Fig1 entry point; historical f/i rates moved to S1.

The reusable source-backed drawing functions remain here. The six-plot layout
is explicitly historical; the default CLI invokes render_fig1_compact.py.
The differently named all-cohort Fig1 is not overwritten.
"""
from __future__ import annotations
from axis_break_marks import mark_nonzero_yaxis, audit_yaxis_breaks

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import render_figures as base
from build_reference_panels import calculate
from matplotlib.collections import Collection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

ROOT=base.ROOT
NAME="fig1_rebuilt_bar_box"
STYLE_REFERENCE=ROOT.parent/"fig5_nhb_evidence_layout.png"
INK="#252B30"
BOX_EDGE="#4C535A"
OBS_COLORS={"#E67E28":"#D57224","#F7BA00":"#B68F00"}
PLOT_HEIGHT_MM=30.0
COMPARISON_WIDTH_MM=27.9
LEARNING_WIDTH_MM=39.0
CANVAS_HEIGHT_MM=206.0
HEIGHT_REFERENCE=ROOT.parent/"fig5_nhb_evidence_layout/height_compaction_verification.json"
WIDTH_REFERENCE=ROOT.parent/"fig5_nhb_evidence_layout/fig5_nhb_evidence_layout_manifest.json"
MARK_WIDTH_MM=4.95
TASK1_LEARNING_TICKS=np.arange(0,51,10)
CONTENT_COLUMN_GAP_MM=12.0
CONTENT_ROW_GAP_MM=6.0
FIG6_WIDTH_REFERENCE=ROOT.parent/"fig6_full_joint90_free_beta_consistent/fig6_full_joint90_free_beta_consistent_manifest.json"
COMPARISON_LEFTS_MM=[68+(LEARNING_WIDTH_MM-COMPARISON_WIDTH_MM)/2,
                    123+(LEARNING_WIDTH_MM-COMPARISON_WIDTH_MM)/2]


def recolor(value):
    rgba=base.matplotlib.colors.to_rgba(value)
    for old,new in OBS_COLORS.items():
        if np.allclose(rgba[:3],base.matplotlib.colors.to_rgb(old),atol=1e-6):
            return (*base.matplotlib.colors.to_rgb(new),rgba[3])
    return rgba


def match_fig5_style(fig):
    """Apply the retained evidence Fig. 5's typography and display palette.

    Axis domains, measured values and uncertainty definitions remain fixed.
    Data-source icons are specific to Fig. 5.
    """
    for artist in fig.findobj():
        if isinstance(artist,Line2D):
            artist.set_color(recolor(artist.get_color()))
            artist.set_markerfacecolor(recolor(artist.get_markerfacecolor()))
            artist.set_markeredgecolor(recolor(artist.get_markeredgecolor()))
        elif isinstance(artist,Collection):
            for getter,setter in [(artist.get_facecolors,artist.set_facecolors),
                                  (artist.get_edgecolors,artist.set_edgecolors)]:
                colors=getter()
                if len(colors):setter([recolor(color) for color in colors])
        elif isinstance(artist,Patch):
            artist.set_facecolor(recolor(artist.get_facecolor()))
            artist.set_edgecolor(recolor(artist.get_edgecolor()))
        elif isinstance(artist,Text):
            artist.set_color(recolor(artist.get_color()))
    letters=set("abcdefghij")
    for text in fig.texts:
        is_letter=text.get_text() in letters
        text.set_fontsize(8 if is_letter else 5 if text.get_fontsize()==6 else 6)
        text.set_fontweight("bold" if is_letter else "normal")
    for name,ax in fig._panels.items():
        if not ax.axison:
            for text in ax.texts:
                text.set_fontsize(5 if np.allclose(base.matplotlib.colors.to_rgba(text.get_color())[:3],
                                                  base.matplotlib.colors.to_rgb(base.GRAY)) else 6)
            continue
        for spine in ax.spines.values():
            if spine.get_visible():spine.set_linewidth(.70);spine.set_color(INK)
        ax.tick_params(which="major",width=.70,labelsize=5,colors=INK)
        for text in [ax.xaxis.label,ax.yaxis.label]:
            text.set_fontsize(6);text.set_fontweight("normal");text.set_color(INK)
        for text in [*ax.get_xticklabels(),*ax.get_yticklabels()]:
            text.set_fontweight("normal")
        for text in ax.texts:
            if text.get_text() not in ("*","**","***","ns"):
                text.set_fontsize(5);text.set_fontweight("normal")
        legend=ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(5);text.set_fontweight("normal")


def reference_statistics():
    return json.loads((base.CSV/"fig1_reference_panel_statistics.json").read_text())


def quantitative_headings(fig,key):
    for index,text in enumerate(fig.texts):
        if text.get_text()==key:
            return [text,fig.texts[index+1]]
    raise KeyError(key)


def quantitative_spacing(fig):
    """Measure visible bounds including ticks, labels, legends and headings."""
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    factor=25.4/fig.dpi
    bounds={}
    for key in "efghij":
        boxes=[fig._panels[key].get_tightbbox(renderer),
               *[text.get_window_extent(renderer) for text in quantitative_headings(fig,key)]]
        bounds[key]=dict(left_mm=min(box.x0 for box in boxes)*factor,
            right_mm=max(box.x1 for box in boxes)*factor,
            bottom_mm=min(box.y0 for box in boxes)*factor,
            top_mm=max(box.y1 for box in boxes)*factor)
    columns=[dict(left_mm=min(bounds[key]["left_mm"] for key in column),
                  right_mm=max(bounds[key]["right_mm"] for key in column))
             for column in ["eh","fi","gj"]]
    rows=[dict(bottom_mm=min(bounds[key]["bottom_mm"] for key in row),
               top_mm=max(bounds[key]["top_mm"] for key in row))
          for row in ["efg","hij"]]
    return dict(bounds_mm=bounds,columns_mm=columns,
        horizontal_gaps_mm=[columns[index+1]["left_mm"]-columns[index]["right_mm"] for index in range(2)],
        vertical_gap_mm=rows[0]["bottom_mm"]-rows[1]["top_mm"])


def balance_quantitative_spacing(fig):
    """Align two rows while balancing actual content gaps at fixed plot sizes."""
    before=quantitative_spacing(fig)
    previous_right=before["columns_mm"][0]["right_mm"]
    shifts={"eh":0.0}
    for keys,column in zip(["fi","gj"],before["columns_mm"][1:]):
        dx=previous_right+CONTENT_COLUMN_GAP_MM-column["left_mm"]
        shifts[keys]=dx
        for key in keys:
            axis=fig._panels[key]
            x,y,width,height=axis.get_position().bounds
            axis.set_position((x+dx/180,y,width,height))
            for text in quantitative_headings(fig,key):
                x,y=text.get_position();text.set_position((x+dx/180,y))
        previous_right=column["right_mm"]+dx
    dy=before["vertical_gap_mm"]-CONTENT_ROW_GAP_MM
    for key in "hij":
        axis=fig._panels[key]
        x,y,width,height=axis.get_position().bounds
        axis.set_position((x,y+dy/fig._height_mm,width,height))
        for text in quantitative_headings(fig,key):
            x,y=text.get_position();text.set_position((x,y+dy/fig._height_mm))
    after=quantitative_spacing(fig)
    assert np.allclose(after["horizontal_gaps_mm"],CONTENT_COLUMN_GAP_MM,atol=1e-7)
    assert np.isclose(after["vertical_gap_mm"],CONTENT_ROW_GAP_MM,atol=1e-7)
    fig._quantitative_spacing=dict(before=before,after=after,
        target_column_gap_mm=CONTENT_COLUMN_GAP_MM,target_row_gap_mm=CONTENT_ROW_GAP_MM,
        column_shifts_mm=shifts,lower_row_shift_up_mm=dy,
        measurement="Final visible bounds including headings, labels, ticks, legends and significance")


def mark_width_data(ax,domain):
    """Match physical bar/box width across two- and three-category axes.

    Fig.5 paired boxes: .33 / 1.86 * 27.9 = 4.95 mm.
    Fig.6 phase boxes: .45 / 2.90 * 31.9 = 4.95 mm.
    """
    axes_width_mm=ax.get_position().width*ax.figure.get_figwidth()*25.4
    return MARK_WIDTH_MM*(domain[1]-domain[0])/axes_width_mm


def reference_comparison(ax,left,right,p):
    """One selected reference contrast in the project's reserved annotation band."""
    line,=ax.plot([left,left,right,right],[1.045,1.070,1.070,1.045],
                 transform=ax.get_xaxis_transform(),color="#252B30",lw=.70,
                 solid_capstyle="butt",clip_on=False)
    text=ax.text((left+right)/2,1.088,base.stars(p),transform=ax.get_xaxis_transform(),
                 ha="center",va="bottom",fontsize=6,fontweight="bold",color=INK)
    ax._reference_comparisons=[(line,text)]


def perfect_bar(ax,task=1):
    config=base.TASK_CONFIG[task]
    count=len(config["groups"])
    positions=np.asarray(config["positions"])
    frame=base.read(f"fig1_task{task}_perfect_rate.csv")
    ink="#252B30"
    fills=[base.matplotlib.colors.to_rgba(color,.58) for color in config["colors"]]
    domain=(-.5,count-.5)
    ax.bar(positions,frame.rate*100,width=mark_width_data(ax,domain),
           color=fills,edgecolor=BOX_EDGE,linewidth=.75,zorder=2)
    intervals=ax.errorbar(positions,frame.rate*100,
        yerr=np.vstack([(frame.rate-frame.lower)*100,(frame.upper-frame.rate)*100]),
        fmt="none",ecolor=ink,elinewidth=.75,capsize=2,capthick=.75,zorder=3)
    ax._restored_intervals=intervals[2][0]
    ax._reference_comparisons=[]
    ax.set(ylim=(0,100),yticks=[0,25,50,75,100],xticks=positions,xticklabels=config["labels"],
           xlim=domain,ylabel="Full-score rate (%)")
    base.nature_task1_axis(ax,categorical=True,categories=count,positions=positions)
    if task==1:
        reference_comparison(ax,positions[1],positions[2],reference_statistics()["tests"]["f"]["p_raw"])


def personal_best_box(ax,task=1):
    config=base.TASK_CONFIG[task]
    count=len(config["groups"])
    positions=np.asarray(config["positions"])
    data=base.read("fig1_reference_personal_best.csv")
    data=data[data.task.eq(task)]
    values=[data.loc[data.group.eq(group),"first_best_trial"].to_numpy()
            for group in config["groups"]]
    ink="#252B30"
    domain=(-.5,count-.5)
    bp=ax.boxplot(values,positions=positions,widths=mark_width_data(ax,domain),
        patch_artist=True,showfliers=False,
        medianprops={"color":ink,"linewidth":1.10},boxprops={"edgecolor":BOX_EDGE,"linewidth":.75},
        whiskerprops={"color":ink,"linewidth":.75},capprops={"color":ink,"linewidth":.75})
    ax._restored_boxes=bp
    for color,patch in zip(config["colors"],bp["boxes"]):
        patch.set_facecolor(color);patch.set_alpha(.58)
    ax.set(ylim=(0,58 if task==1 else 70),
           yticks=[10,20,30,40,50] if task==1 else [15,30,45,60],
           xticks=positions,xticklabels=config["labels"],
           xlim=domain,ylabel="Round of personal best")
    base.nature_task1_axis(ax,categorical=True,categories=count,positions=positions)
    pair=positions[1:] if task==1 else positions
    panel="g" if task==1 else "j"
    reference_comparison(ax,*pair,reference_statistics()["tests"][panel]["p_raw"])


def check_statistics():
    """Recalculate frozen reference endpoints, counts and selected tests."""
    personal,calculated=calculate()
    frozen=base.read("fig1_reference_personal_best.csv")
    base.pd.testing.assert_frame_equal(frozen.assign(subject=frozen.subject.astype(str)),
                                      personal.assign(subject=personal.subject.astype(str)))
    stats=reference_statistics()
    assert stats==json.loads(json.dumps(calculated))
    return stats


def audit(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    axes={k:fig._panels[k] for k in "efghij"}
    stats=check_statistics()
    curves=base.read("fig1_learning_display.csv")
    sample_counts={};all_texts=[]
    for task,learning,bar,box in [(1,"e","f","g"),(2,"h","i","j")]:
        config=base.TASK_CONFIG[task]
        for group,line,band in zip(config["groups"],axes[learning].lines,axes[learning].collections):
            part=curves[curves.task.eq(task)&curves.group.eq(group)]
            assert np.allclose(line.get_xdata(),part.trial) and np.allclose(line.get_ydata(),part.mean_smooth)
            y=band.get_paths()[0].vertices[:,1]
            assert np.isclose(y.min(),(part.mean_smooth-part["sem"]).min())
            assert np.isclose(y.max(),(part.mean_smooth+part["sem"]).max())
        assert axes[learning].get_xlabel()=="Trial" and not axes[learning].texts
        rates=base.read(f"fig1_task{task}_perfect_rate.csv")
        assert np.allclose([p.get_height() for p in axes[bar].patches],rates.rate*100)
        for segment,(_,row) in zip(axes[bar]._restored_intervals.get_segments(),rates.iterrows()):
            assert np.allclose(segment[:,1],[row.lower*100,row.upper*100])
        for patch,color in zip(axes[bar].patches,config["colors"]):
            assert np.allclose(patch.get_facecolor(),recolor(base.matplotlib.colors.to_rgba(color,.58)))
            assert np.isclose(patch.get_linewidth(),.75)
        data=base.read("fig1_reference_personal_best.csv")
        data=data[data.task.eq(task)]
        assert len(axes[box].collections)==0  # No individual-dot or mean markers.
        assert all(len(line.get_xdata())==0 for line in axes[box]._restored_boxes["fliers"])
        sample_counts[box]=[]
        for index,group in enumerate(config["groups"]):
            values=data.loc[data.group.eq(group),"first_best_trial"].to_numpy()
            bp=axes[box]._restored_boxes
            vertices=bp["boxes"][index].get_path().vertices
            assert np.allclose([vertices[:,1].min(),vertices[:,1].max()],np.percentile(values,[25,75]))
            assert np.allclose(bp["medians"][index].get_ydata(),np.median(values))
            assert np.isclose(bp["medians"][index].get_linewidth(),1.10)
            assert np.isclose(bp["boxes"][index].get_facecolor()[-1],.58)
            sample_counts[box].append(len(values))
        labels=[]
        if task==1:labels.append(base.stars(stats["tests"][bar]["p_raw"]))
        assert [t.get_text() for t in axes[bar].texts]==labels
        assert [t.get_text() for t in axes[box].texts]==[base.stars(stats["tests"][box]["p_raw"])]
        for key in [bar,box]:
            for line,text in axes[key]._reference_comparisons:
                window=line.get_transform().transform(np.column_stack((line.get_xdata(),line.get_ydata())))
                assert text.get_window_extent(renderer).y0>window[:,1].max()+1
                assert text.get_fontweight()=="bold" and text.get_fontsize()==6
        if task==2:
            for line,color in zip(axes[learning].lines,config["colors"]):
                assert np.allclose(base.matplotlib.colors.to_rgba(line.get_color()),recolor(color))
    assert sample_counts=={"g":[51,51,54],"j":[40,40]}
    assert list(axes["e"].get_ylim())==[50.,100.] and list(axes["h"].get_ylim())==[45.,90.]
    assert np.allclose(axes["e"].get_xlim(),[0,50])
    assert np.allclose(axes["e"].get_xticks(),TASK1_LEARNING_TICKS)
    tick_positions=axes["e"].transData.transform(np.column_stack((TASK1_LEARNING_TICKS,np.zeros(len(TASK1_LEARNING_TICKS)))))[:,0]
    assert np.allclose(np.diff(tick_positions),np.diff(tick_positions)[0])
    for key,ax in axes.items():
        texts=[ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels(),*ax.texts]
        if ax.get_legend() is not None:texts.extend(ax.get_legend().get_texts())
        all_texts.extend((key,t) for t in texts if t.get_visible() and t.get_text().strip())
    all_texts.extend(("heading",t) for t in fig.texts if t.get_position()[1]<1-95/fig._height_mm)
    collisions=[]
    for index,(key,text) in enumerate(all_texts):
        a=text.get_window_extent(renderer)
        for key2,text2 in all_texts[index+1:]:
            b=text2.get_window_extent(renderer)
            if min(a.x1,b.x1)-max(a.x0,b.x0)>.5 and min(a.y1,b.y1)-max(a.y0,b.y0)>.5:
                collisions.append([key,key2,text.get_text(),text2.get_text()])
    assert not collisions,collisions
    geometry={k:dict(left_mm=ax.get_position().x0*180,top_mm=(1-ax.get_position().y1)*fig._height_mm,
        width_mm=ax.get_position().width*180,
        height_mm=ax.get_position().height*fig._height_mm) for k,ax in axes.items()}
    ref_height=json.loads(HEIGHT_REFERENCE.read_text())["after_plot_height_mm"]
    ref_width=json.loads(WIDTH_REFERENCE.read_text())["edition_audit"]["statistical_panel_geometry"]["plot_widths_mm"]["c"]
    assert np.isclose(ref_height,PLOT_HEIGHT_MM)
    assert np.isclose(ref_width,COMPARISON_WIDTH_MM)
    for key,row in geometry.items():
        assert np.isclose(row["width_mm"],LEARNING_WIDTH_MM if key in "eh" else ref_width)
        assert np.isclose(row["height_mm"],ref_height)
    assert np.isclose(geometry["e"]["left_mm"],13)
    for column in ["eh","fi","gj"]:
        assert np.isclose(geometry[column[0]]["left_mm"],geometry[column[1]]["left_mm"])
    for row in ["efg","hij"]:
        assert np.ptp([geometry[key]["top_mm"] for key in row])<1e-7
    spacing=quantitative_spacing(fig)
    assert np.allclose(spacing["horizontal_gaps_mm"],CONTENT_COLUMN_GAP_MM,atol=1e-7)
    assert np.isclose(spacing["vertical_gap_mm"],CONTENT_ROW_GAP_MM,atol=1e-7)
    # Verify actual exported patch geometry, independent of the data-unit
    # arguments: categorical domains and group counts differ by task.
    marks={key:[float(patch.get_window_extent(renderer).width*25.4/fig.dpi)
                for patch in axes[key].patches] for key in "fgij"}
    for widths in marks.values():
        assert np.allclose(widths,MARK_WIDTH_MM,atol=1e-7),widths
    fig6_box=json.loads(FIG6_WIDTH_REFERENCE.read_text())["edition_audit"]["horizontal_spacing_qa"]["after"]["areas"]["f"]["plot_mm"]
    fig5_box_width=.33*ref_width/1.86
    fig6_box_width=.45*(fig6_box[2]-fig6_box[0])/2.90
    assert np.isclose(MARK_WIDTH_MM,fig5_box_width)
    assert np.isclose(MARK_WIDTH_MM,fig6_box_width)
    for key,ax in axes.items():
        assert all(np.isclose(ax.spines[s].get_linewidth(),.70) for s in ["left","bottom"])
        for text in [ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]:
            assert text.get_fontweight()=="normal"
        assert all(text.get_fontsize()==5 for text in [*ax.get_xticklabels(),*ax.get_yticklabels()])
        assert all(np.isclose(tick.tick1line.get_markeredgewidth(),.70) for tick in ax.yaxis.get_major_ticks())
        if key in "figj":
            assert ax.spines["bottom"].get_position()==("outward",3)
    font_paths=sorted({base.matplotlib.font_manager.findfont(t.get_fontproperties()) for _,t in all_texts})
    assert all("DejaVuSans" in path for path in font_paths)
    report=dict(status="PASS",variant=NAME,restored_panels=list("figj"),learning_actual_trials_preserved=True,
        task1_learning_ticks=dict(xlim=[0,50],positions=TASK1_LEARNING_TICKS.tolist(),
            physical_intervals_mm=(np.diff(tick_positions)*25.4/fig.dpi).tolist(),
            observed_first_trial=1,zero_trial_imputed=False),
        learning_sem="Across individually smoothed participant trajectories",geometry_mm=geometry,
        quantitative_spacing=fig._quantitative_spacing,
        personal_best_counts=sample_counts,time_excluded_noncompleters={"g":0,"j":0},
        noncompleters_included={"g":85,"j":50},
        time_estimand="First observed trial at participant's own maximum; not first-full-score time",
        proportion_intervals="Wilson 95%CI",statistics_source="csv/fig1_reference_panel_statistics.json",
        nonparametric_tests_verified=True,tests=stats["tests"],annotation_text_collisions=collisions,
        significance_text_clear_of_brackets=True,bar_value_annotations=False,box_scatter_markers=False,
        project_typography=dict(font_paths=font_paths,axis_tick_label_weight="normal",axis_width_pt=.70),
        style_reference=str(STYLE_REFERENCE),style_reference_sha256=hashlib.sha256(STYLE_REFERENCE.read_bytes()).hexdigest(),
        fig5_style=dict(font_sizes_pt=[5,6,8],tick_size_pt=5,label_size_pt=6,legend_size_pt=5,
                       box_alpha=.58,box_scatter_markers=False,box_edge=BOX_EDGE,box_line_width_pt=.75,
                       median_width_pt=1.10,observation_palette=OBS_COLORS,
                       annotation="Same bold star convention and annotation band as retained Fig. 5"),
        height_reference=dict(path=str(HEIGHT_REFERENCE),panel="Fig.5c",height_mm=ref_height),
        width_reference=dict(path=str(WIDTH_REFERENCE),panel="Fig.5c",width_mm=ref_width,
                             comparison_panels=list("fgij"),learning_width_mm=LEARNING_WIDTH_MM),
        mark_width_reference=dict(target_mm=MARK_WIDTH_MM,measured_widths_mm=marks,
            fig5=dict(path=str(WIDTH_REFERENCE),panel="paired comparison boxes",width_mm=fig5_box_width,
                      data_width=.33,x_domain=[-.43,1.43],axes_width_mm=ref_width),
            fig6=dict(path=str(FIG6_WIDTH_REFERENCE),panel="f",width_mm=fig6_box_width,
                      data_width=.45,x_domain=[-.45,2.45],axes_width_mm=fig6_box[2]-fig6_box[0])),
        reference_style_adaptations="Reference endpoints; comparison axes27.9x30mm; bar/box marks4.95mm matched to Fig.5/6; no bar labels or box dots")
    (base.QA/f"{NAME}_native_style.json").write_text(json.dumps(report,indent=2)+"\n")


def render_legacy(output_dir,qa_dir,replace=False):
    output_dir.mkdir(parents=True,exist_ok=True);qa_dir.mkdir(parents=True,exist_ok=True)
    output=output_dir/f"{NAME}.png"
    if output.exists() and not replace:raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory")
    base.OUT=output_dir;base.QA=qa_dir
    fig=base.make_canvas(CANVAS_HEIGHT_MM)
    base.figure_text(fig,3,2,"Task structure, learning and full-score attainment",color=base.GRAY)
    for letter,x,title in [("a",3,"Task 1 | meal ranking"),("b",94,"Task 2 | scene selection")]:
        base.panel_heading(fig,letter,x,7,title)
    a=base.axis(fig,"a",3,14,83,45,schematic=True);base.task1_design(a)
    b=base.axis(fig,"b",94,14,83,45,schematic=True);base.task2_design(b)
    base.panel_heading(fig,"c",3,63,"Task 1 | transfer from prior practice")
    base.panel_heading(fig,"d",94,63,"Task 2 | prior observation")
    c=base.axis(fig,"c",3,70,83,23,schematic=True);base.task1_groups(c)
    d=base.axis(fig,"d",94,70,83,23,schematic=True);base.task2_groups(d)
    # Preserve the first quantitative row's top edge and original inter-row gap;
    # remove the 13mm freed by each shorter row rather than adding whitespace.
    for task,letters,heading_top,top in [(1,"efg",99,108),(2,"hij",153,162)]:
        for letter,x,title in zip(letters,[3,*[left-10 for left in COMPARISON_LEFTS_MM]],
                                   [f"Task {task} | learning","Full-score rate","Personal best"]):
            base.panel_heading(fig,letter,x,heading_top,title)
        learning=base.axis(fig,letters[0],13,top,LEARNING_WIDTH_MM,PLOT_HEIGHT_MM);base.learning(learning,task)
        if task==1:
            # Show a conventional origin with equally spaced true trial ticks;
            # leave all line/band data at their observed trial coordinates.
            learning.set(xlim=(0,50),xticks=TASK1_LEARNING_TICKS)
        bar=base.axis(fig,letters[1],COMPARISON_LEFTS_MM[0],top,COMPARISON_WIDTH_MM,PLOT_HEIGHT_MM);perfect_bar(bar,task)
        box=base.axis(fig,letters[2],COMPARISON_LEFTS_MM[1],top,COMPARISON_WIDTH_MM,PLOT_HEIGHT_MM);personal_best_box(box,task)
    fig._rows=[["a","b"],["c","d"],["e","f","g"],["h","i","j"]]
    fig._columns=[["e","h"],["f","i"],["g","j"]]
    match_fig5_style(fig)
    for key in "efghij":
        mark_nonzero_yaxis(fig._panels[key],panel=key)
    balance_quantitative_spacing(fig)
    audit(fig)
    # Plot widths differ by evidence type; visible-content gaps are uniform.
    # Check row edges/heights and corresponding columns, exempting only
    # intentional plot-width/gutter differences relative to learning plots.
    exemptions=[dict(panels=list(row),checks=["panel-width","horizontal-gutter"],
                      reason="Learning axes39mm and endpoint axes27.9mm; user requests equal12mm visible-content gaps including labels/headings")
                for row in ["efg","hij"]]
    base.finish(fig,NAME,nominal_font_sizes=(5,6,8),alignment_exemptions=exemptions)
    # Shared finish uses the original names to classify draft status; this variant is Fig1.
    metadata=qa_dir/f"{NAME}_verification.json"
    report=json.loads(metadata.read_text());report["status"]="source_backed_reference_bar_personal_best_panels"
    report["shared_renderer_sha256"]=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()
    report["y_axis_breaks"]=audit_yaxis_breaks({k:fig._panels[k] for k in "efghij"})
    report["plot_input_hashes"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in base.CSV.iterdir() if p.is_file()}
    metadata.write_text(json.dumps(report,indent=2)+"\n")
    return output


def render(output_dir,qa_dir,replace=False):
    from render_fig1_compact import render as render_compact
    return render_compact(output_dir,qa_dir,replace)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT.parent)
    parser.add_argument("--qa-dir",type=Path,default=ROOT/"qa/bar_box")
    parser.add_argument("--replace",action="store_true",help="Replace only this named variant, never fig1_rebuilt.png")
    args=parser.parse_args()
    render(args.output_dir,args.qa_dir,args.replace)
