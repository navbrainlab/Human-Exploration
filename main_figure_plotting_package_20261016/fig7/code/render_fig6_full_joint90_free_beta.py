"""Render a separate edition: performance, phase slopes, DIS trajectories."""
from __future__ import annotations
from axis_break_marks import mark_nonzero_yaxis, audit_yaxis_breaks
from detached_axes import detach_axes, audit_detached_axes
from category_label_spacing import compact_category_margin
from tick_bounded_spines import bound_spines_to_ticks,audit_tick_bounded_spines

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
MAIN = PACKAGE.parent
PANEL_C_YLABEL = 'Max. dim. attention weight'
ROOT = PACKAGE
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load('fig6_free_beta_base', PACKAGE/'code/full_joint90_native.py')
slope = load('fig6_free_beta_fitted_slope', PACKAGE/'code/phase_slope.py')
layout = base.layout
layout.CSV_ROOT = PACKAGE/'data'
layout.DATA_PATHS = {key: PACKAGE/'data'/path.name for key,path in layout.DATA_PATHS.items()
                     if key != 'temperature_entities'}
layout.DATA_PATHS.update(participant_parameters=PACKAGE/'data/participant_fitted_parameters.csv',
    slope_participants=PACKAGE/'data/participant_temperature_slope.csv',
    slope_tests=PACKAGE/'data/slope_phase_tests.csv')
layout.DATA_PATHS['best_performance']=PACKAGE/'data/performance_task1_4d.csv'
layout.DATA_PATHS['task2_fit_tests']=PACKAGE/'data/task2_fit_tests.csv'
layout.DATA_PATHS['task2_fit_display_significance']=PACKAGE/'data/task2_fit_display_significance.csv'
layout.STATISTICS_PATH = PACKAGE/'data/panel_statistics.csv'
layout.CAPTION_PATH = PACKAGE/'caption.md'
layout.METHODS_PATH = PACKAGE/'methods.md'
layout.SOURCE_PROVENANCE_PATH = PACKAGE/'source_provenance.json'
layout.MANIFEST_DIR = PACKAGE
layout.OUTPUT_DIR = MAIN
base.AUDIT = PACKAGE
slope.PACKAGE = PACKAGE
slope.layout = layout
layout._draw_temperature_comparison = slope.draw_slope
layout.PANEL_MAPPING.update(b='Task1 retained fits; Task2 seven-parameter DGEM versus saved V2 ActionScore-softmax alternatives; ACL gaze-conditional with uniform fallback',
    c='Free-beta fitted-data replay: effective attention temperature and dimension weight',
    d='Free-beta fitted-data replay: maximum attention and observed DIS',
    e='Task1 retained 4D-only performance; Task2 V2 alternatives (80 saved episodes, deterministic feedback) versus retained DGEM (noisy feedback) and Human references; Bayesian Task1 historical',
    f='Seven-parameter free-beta fitted temperature slopes: 3D, 4D-E, 4D-NE; boxes only, without individual points or paired lines',
    g='Free-beta fitted-parameter simulations: DIS; lower trace is fitted-data replay temperature',
    h='Dimension-guided movie recommendation: semantic tag clusters, selected tags and candidates; Movie B highlighted as selected')


def statistics():
    frame = pd.read_csv(layout.STATISTICS_PATH)
    assert set(frame.panel_comparison) == {'d_DIS_vs_Non-DIS'}
    assert frame.p_value.between(0,1).all()
    return frame


layout._compute_panel_statistics = statistics
original_b = layout._draw_fitting_comparison


def draw_fitting(figure):
    before = len(figure.axes)
    original_b(figure)
    first, second = figure.axes[before:]
    values = pd.read_csv(layout.DATA_PATHS['fit_summary']).set_index('model')
    dgem = values.loc['DGEM']
    end = 1000*(dgem.mean_geom_likelihood+dgem.sem_geom_likelihood)+.20
    bracket = end+1.0
    step = float(np.ceil((end+3.0)/3))
    first.set_xlim(0,3*step)
    first.set_xticks(np.arange(4)*step)
    for line in first.lines:
        x,y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        if len(x)==4 and y[0]==4 and y[1]==4 and y[2]==y[3]:
            line.set_xdata([end,bracket,bracket,x[-1]])
    for text in first.texts:
        if text.get_text() in ('ns','*','**','***'):
            text.set_x(bracket+1.0)
    task2 = pd.read_csv(layout.DATA_PATHS['task2_fit']).set_index('model')
    end2 = 1000*(task2.loc['DGEM','mean']+task2.loc['DGEM','se'])+.20
    bracket2 = end2+1.
    step2 = float(np.ceil((end2+3.)/3))
    second.set_xlim(0,3*step2)
    second.set_xticks(np.arange(4)*step2)
    for line in second.lines:
        x,y=np.asarray(line.get_xdata()),np.asarray(line.get_ydata())
        if len(x)==4 and y[0]==4 and y[1]==4 and y[2]==y[3]:
            line.set_xdata([end2,bracket2,bracket2,x[-1]])
    for text in second.texts:
        if text.get_text() in ('ns','*','**','***'):text.set_x(bracket2+1.)
    second.set_title('Task 2', loc='left', pad=4.0, fontsize=6.0)
    # The current ACL row is a real comparison at0. Give its y-spine
    # endpoint a major tick, retaining the sole existing manual ACL label.
    second.set_yticks([4,3,2,1,0],['','','','',''])


layout._draw_fitting_comparison = draw_fitting


def finalize(figure, main, extension, tests):
    # Creation order stays stable for the established typography helpers.
    fitting1,fitting2=figure.axes[1:3]
    attention=figure.axes[3]
    attention.set_ylabel(PANEL_C_YLABEL)
    _,bottom,_,height=attention.get_position().bounds
    old_bottom=fitting1.get_position().y0
    for axis in (fitting1,fitting2):
        left,_,width,_=axis.get_position().bounds
        axis.set_position((left,bottom,width,height))
    for text in main.texts:
        if text.get_text().startswith('Mean likelihood per trial'):
            text.set_y(text.get_position()[1]+bottom-old_bottom)
    performance1,performance2=figure.axes[6:8]
    dis,temperature=figure.axes[8:10]
    phase=figure.axes[10]
    phase.set_position((.495,.075,.215,.215))
    dis.set_position((.760,.140,.215,.150))
    temperature.set_position((.760,.075,.215,.040))
    performance1.set_position((.073,.075,.152,.215))
    performance2.set_position((.273,.075,.162,.215))
    for text in main.texts:
        label=text.get_text()
        if label=='e':text.set_position((.008,.324))
        elif label=='f':text.set_position((.469,.324))
        elif label=='g':text.set_position((.732,.324))
        elif label=='DIS ratio':text.set_x(.254)
    result = base.finalize(figure,main,extension,tests)
    # Restore6pt tick labels without clipping b1's longest model name.
    # Its right/top/bottom edges and b1–b2 gutter stay fixed; only its left
    # display margin/width change. All likelihood/SEM values remain native.
    b1_before=list(fitting1.get_position().bounds)
    left,bottom,width,height=b1_before
    fitting1.set_position((left+1.7/180,bottom,width-1.7/180,height))
    result['b1_font_layout_allowance_20261014']={
        'left_edge_rightward_mm':1.7,'before_plot_bounds_fraction':b1_before,
        'after_plot_bounds_fraction':list(fitting1.get_position().bounds),
        'right_top_bottom_edges_unchanged':True,'b1_b2_plot_gutter_unchanged':True,
        'reason':'Allow6pt fRL-decay tick to retain a1mm canvas margin with normal positive2pt tick padding',
        'numeric_domains_source_likelihoods_and_SEMs_unchanged':True}
    figure.canvas.draw()
    # Match actual visible star-to-line clearance in physical units, using
    # Task2 as the reference after final font and axis-size adjustments.
    renderer=figure.canvas.get_renderer()
    fit_artists=[]
    for axis in (fitting1,fitting2):
        connector=next(line for line in axis.lines if len(line.get_xdata())==4
            and line.get_ydata()[0]==4 and line.get_ydata()[1]==4)
        star=next(text for text in axis.texts if text.get_text() in ('ns','*','**','***'))
        vx=connector.get_transform().transform(connector.get_xydata())[1,0]
        fit_artists.append((axis,connector,star,vx))
    task2_box=fit_artists[1][2].get_window_extent(renderer)
    clearance=max(task2_box.x0-fit_artists[1][3],figure.dpi*.5/25.4)
    for axis,connector,star,vx in fit_artists:
        box=star.get_window_extent(renderer)
        px,py=star.get_transform().transform(star.get_position())
        shift=vx+clearance-box.x0
        star.set_position(star.get_transform().inverted().transform((px+shift,py)))
    figure.canvas.draw()
    clearances=[(star.get_window_extent(figure.canvas.get_renderer()).x0-vx)*25.4/figure.dpi
                for axis,connector,star,vx in fit_artists]
    assert abs(clearances[0]-clearances[1])<1e-8
    assert 0 not in fitting1.get_yticks()
    assert 0 in fitting2.get_yticks()
    shared=next(line for line in fitting2.lines if len(line.get_xdata())==2
        and np.ptp(line.get_xdata())==0 and np.ptp(line.get_ydata())>3.)
    assert np.allclose(shared.get_ydata(),[-.34,3.34])
    result['b_revision']={'star_clearance_mm':clearances,'ACL_label':'ACL',
        'Task1_ACL_tick_absent':True,'Task2_ACL_major_tick':0,
        'Task2_ACL_tick_label':'','Task2_ACL_label':'Existing manual ACL label, no duplicate',
        'Task2_shared_line_y':list(shared.get_ydata()),
        'ACL_included_in_display_connector':True,'ACL_test':'saved V2 separate gaze-conditional/fallback Wilcoxon, 80 pairs'}
    # Shared legend sits in the row gutter, outside both performance data areas.
    center=(performance1.bbox.x0+performance2.bbox.x1)/2
    y=max(performance1.bbox.y1,performance2.bbox.y1)+figure.dpi*3.0/25.4
    anchor=main.transSubfigure.inverted().transform((center,y))
    from matplotlib.lines import Line2D
    handles=[]
    for model,container in zip(main._performance_model_order,performance2.containers):
        line=container.lines[0]
        handles.append(Line2D([],[],linestyle='none',marker=line.get_marker(),
            markersize=line.get_markersize(),markerfacecolor=line.get_markerfacecolor(),
            fillstyle=line.get_fillstyle(),markerfacecoloralt=line.get_markerfacecoloralt(),
            markeredgecolor=line.get_markeredgecolor(),markeredgewidth=line.get_markeredgewidth(),
            color=line.get_color(),label='Bayesian‡' if model=='Bayesian' else model))
    legend=main.legend(handles=handles,loc='lower center',
        bbox_to_anchor=anchor,frameon=False,ncol=3,fontsize=6.,borderaxespad=0,
        handlelength=.9,handletextpad=.35,columnspacing=.85,labelspacing=.25)
    base.final._recolor(figure)
    qa=base.final._render_qa(figure,base.AUDIT/'fig6_full_joint90_render_qa.json')
    figure.canvas.draw()
    bounds=legend.get_window_extent(figure.canvas.get_renderer())
    for axis in (performance1,performance2):
        assert bounds.y0>axis.bbox.y1,'Performance legend overlaps data area'
    result['performance_legend']={'rows':2,'columns':3,'font_pt':6.,
        'placement':'shared reserved row gutter above e;outside data areas',
        'bbox_pixels':list(bounds.extents),'no_point_jitter':True,
        'model_representation':'Task1 retained feature+item fRL / item-only whole-Q naive; Task2 V2 ActionScore-softmax alternatives; no jitter',
        'source_role_display':'Original filled style retained at user request; Task2 V2/reference distinction in caption'}
    numeric=[]
    for axis,key,models in [(performance1,'best_performance',['Human','DGEM','Bayesian','fRL-decay','naiveRL']),
                            (performance2,'task2_performance',['Human','DGEM','Bayesian','fRL-decay','naiveRL','ACL'])]:
        table=pd.read_csv(layout.DATA_PATHS[key]).replace({'model':{'fRL':'fRL-decay','DGEM (winner)':'DGEM'}}).set_index('model')
        assert len(axis.containers)==len(models)
        for container,model in zip(axis.containers,models):
            row=table.loc[model]
            x=float(row['fds_ratio'] if key=='best_performance' else row['DIS_rate_mean'])
            y=float(row['full_score_rate'] if key=='best_performance' else row['full_score_rate_mean']/100.)
            np.testing.assert_allclose(np.asarray(container.lines[0].get_xdata(),dtype=float),[x],atol=1e-14)
            np.testing.assert_allclose(np.asarray(container.lines[0].get_ydata(),dtype=float),[y],atol=1e-14)
            numeric.append(dict(task=key,model=model,DIS_ratio=x,full_score_proportion=y))
            line=container.lines[0]
            px,py=axis.transData.transform((x,y))
            radius=(line.get_markersize()+line.get_markeredgewidth())/2*figure.dpi/72
            assert axis.bbox.x0<=px-radius and px+radius<=axis.bbox.x1,(key,model,'horizontal marker clipping')
            assert axis.bbox.y0<=py-radius and py+radius<=axis.bbox.y1,(key,model,'vertical marker clipping')
            index=list(main._performance_model_order).index(model)
            handle=legend.legend_handles[index];line=container.lines[0]
            for getter in ['get_marker','get_markersize','get_markeredgewidth','get_fillstyle']:
                assert getattr(handle,getter)()==getattr(line,getter)(),(key,model,getter)
            from matplotlib.colors import to_rgba
            for getter in ['get_markerfacecolor','get_markerfacecoloralt','get_markeredgecolor','get_color']:
                np.testing.assert_allclose(to_rgba(getattr(handle,getter)()),to_rgba(getattr(line,getter)()),atol=1e-14)
    result['performance_numeric_coordinates']=numeric
    result['performance_legend']['all_11_marker_styles_match_legend']=True
    result['performance_marker_clipping_qa']={'status':'PASS','series':11,
        'right_display_padding':1.15,'maximum_ratio_tick':1.0}
    result['qa_status']='PASS'
    result['y_axis_labels'] = {'c': attention.get_ylabel()}
    result.pop('full_dgem_90both',None)
    result['scientific_scope'] = {
        'b_Task1_c_d_f_g': 'DGEM with individually fitted beta_init and six other parameters',
        'b_Task2': 'retained seven-parameter DGEM; V2 baselines; all 80 people/4111 trials; ACL five parameters and real-gaze/uniform fallback, separate saved test',
        'e': 'Task1 retained 4D-only, Human4D-NE54; DGEM5000, RL1000, historical Bayesian100. Task2 V2 alternatives80 episodes with deterministic feedback; retained DGEM1000 noisy-feedback episodes and Human80. Not an equal-environment benchmark. DGEM performance fixed beta20; g historical RL recovery unchanged.',
        'f': 'seven-parameter free-beta fitted slopes, new Holm(3) phase tests, without paired visual lines',
    }
    figure.canvas.draw()
    factor=25.4/figure.dpi
    fitting_attention={'status':'PASS',
        'plot_heights_mm':{name:axis.bbox.height*factor for name,axis in
                           [('b1',fitting1),('b2',fitting2),('c',attention)]},
        'top_spread_mm':float(np.ptp([axis.bbox.y1 for axis in (fitting1,fitting2,attention)]))*factor,
        'bottom_spread_mm':float(np.ptp([axis.bbox.y0 for axis in (fitting1,fitting2,attention)]))*factor}
    assert fitting_attention['top_spread_mm']<.01 and fitting_attention['bottom_spread_mm']<.01
    result['fitting_attention_height_qa']=fitting_attention
    top=[phase.bbox.y1,dis.bbox.y1,performance1.bbox.y1,performance2.bbox.y1]
    bottom=[phase.bbox.y0,temperature.bbox.y0,performance1.bbox.y0,performance2.bbox.y0]
    assert max(top)-min(top)<figure.dpi*1.5/72
    assert max(bottom)-min(bottom)<figure.dpi*1.5/72
    result['third_row_edges_aligned']=True
    from matplotlib.collections import PathCollection
    counts=[len(item.get_offsets()) for item in phase.collections if isinstance(item,PathCollection)]
    assert not counts, 'No individual scatter points should appear in panel f'
    spec=phase._slope_display_spec
    assert spec['source_counts']==[51,51,54] and len(phase.patches)==3
    diagonal_pairs=[line for line in phase.lines if len(line.get_xdata())==2
                    and np.ptp(line.get_xdata())>0 and np.ptp(line.get_ydata())>0]
    assert not diagonal_pairs, 'Participant connecting lines must be absent'
    # Audit annotation text against every other symbol and bracket in the
    # final rendering, after the global typography/physical-size adjustment.
    from matplotlib.path import Path as MplPath
    renderer=figure.canvas.get_renderer()
    adjacent_heights=[line.get_transform().transform(line.get_xydata())[1,1]
                      for line in spec['brackets'][:2]]
    assert abs(adjacent_heights[0]-adjacent_heights[1])<1e-6
    boxes=[text.get_window_extent(renderer) for text in spec['stars']]
    for i,box in enumerate(boxes):
        for other in boxes[i+1:]:
            assert min(box.x1,other.x1)<=max(box.x0,other.x0) or min(box.y1,other.y1)<=max(box.y0,other.y0)
        for line in spec['brackets']:
            assert not MplPath(line.get_transform().transform(line.get_xydata())).intersects_bbox(box,filled=False)
    result['slope_panel']={'source_counts':spec['source_counts'],'displayed_point_counts':counts,
                           'box_count':len(phase.patches),'connecting_subject_lines':0,
                           'statistics_unchanged':False,'variant':'free_bonus','beta_init_fitted':True,
                           'display':'boxes only; individual points and fliers hidden',
                           'significance_annotation_collisions':0,
                           'adjacent_comparison_brackets_same_height':True}
    spec=figure.axes[11]._movie_selection_spec
    assert len(spec['containers'])==12 and len(spec['strokes'])==3
    result['h_schematic']={'display':'selected Movie B within candidates; no separate Action box',
                           'boxes':12,'arrows':3,'selected_movie':'Movie B','action_box':False,
                           'other_tags_muted_block':True,
                           'text_labels':len(spec['texts']),'details_in_methods':True}
    result['public_panel_order']=['e: performance tasks','f: slope distributions','g: DIS and temp.']
    quantitative=dict(zip(('c','d_upper','d_lower','e1','e2','g_upper','g_lower','f','i'),
        [figure.axes[index] for index in (3,4,5,6,7,8,9,10,12)]))
    # The Full range inset intentionally omits break marks (user request).
    for name,axis in quantitative.items():
        mark_nonzero_yaxis(axis,panel=name)
    figure.canvas.draw()
    result['y_axis_breaks']=audit_yaxis_breaks(quantitative)
    base.final._render_qa(figure,base.AUDIT/'fig6_full_joint90_render_qa.json')
    result['y_axis_break_exclusions']={name:{'scale':axis.get_yscale(),'ylim':list(axis.get_ylim())}
        for name,axis in quantitative.items() if not hasattr(axis,'_publication_y_break')}
    for index,axis in enumerate(attention.child_axes):
        assert not hasattr(axis,'_publication_y_break')
        assert not any(str(line.get_gid()).startswith('publication-y-break') for line in axis.lines)
        result['y_axis_break_exclusions'][f'c_inset{index+1}']={
            'scale':axis.get_yscale(),'ylim':list(axis.get_ylim()),
            'reason':'User requested no double slashes in the Full range inset'}
    # 2026-10-14 style-only revision. The Full range inset retains its
    # compact axes and the user's explicit no-omission-marker requirement.
    cartesian = dict(zip(('b1','b2','c','d_upper','d_lower','e1','e2',
                          'g_upper','g_lower','f','i'),
                         [figure.axes[index] for index in (1,2,3,4,5,6,7,8,9,10,12)]))
    def scientific_artist_snapshot():
        from matplotlib.collections import PathCollection
        state = []
        for axis in cartesian.values():
            lines = [line for line in axis.lines
                     if not str(line.get_gid()).startswith('publication-y-break')]
            state.append({
                'xlim':list(axis.get_xlim()),'ylim':list(axis.get_ylim()),
                'xscale':axis.get_xscale(),'yscale':axis.get_yscale(),
                'xticks':list(axis.get_xticks()),'yticks':list(axis.get_yticks()),
                'position':list(axis.get_position().bounds),
                'line_xy':[np.asarray(line.get_xydata()).tolist() for line in lines],
                'collection_offsets':[np.asarray(item.get_offsets()).tolist()
                      for item in axis.collections if isinstance(item,PathCollection)],
                'collection_paths':[[path.vertices.tolist() for path in item.get_paths()]
                                    for item in axis.collections],
                'patch_vertices':[item.get_path().vertices.tolist() for item in axis.patches],
                'annotations':[item.get_text() for item in axis.texts],
            })
        return sha256(json.dumps(state,sort_keys=True,default=lambda item:item.item()).encode()).hexdigest()
    frozen_artist_hash = scientific_artist_snapshot()
    for axis in cartesian.values():
        detach_axes(axis)
    figure.canvas.draw()
    def rendered_tick_boxes():
        renderer=figure.canvas.get_renderer()
        return [{'axis':name,'orientation':orientation,'text':text.get_text(),
                 'fontsize_pt':text.get_fontsize(),
                 'bbox_pixels':text.get_window_extent(renderer).extents.tolist()}
                for name,axis in cartesian.items()
                for orientation,labels in (('x',axis.get_xticklabels()),('y',axis.get_yticklabels()))
                for text in labels if text.get_visible() and text.get_text().strip()]
    tick_boxes_before=rendered_tick_boxes()
    # Preserve existing visible label anchors while increasing tick glyphs.
    # Otherwise Matplotlib's automatic label placement would move ylabels
    # left and xlabels down, broadening this tick-only repair.
    fixed_label_anchors={}
    for name,axis in cartesian.items():
        for orientation,coordinate_axis in (('x',axis.xaxis),('y',axis.yaxis)):
            label=coordinate_axis.label
            if label.get_visible() and label.get_text():
                pixels=label.get_transform().transform(label.get_position())
                coordinates=axis.transAxes.inverted().transform(pixels)
                coordinate_axis.set_label_coords(*coordinates)
                fixed_label_anchors[f'{name}_{orientation}']=pixels.tolist()
    # Spine repositioning rebuilds Matplotlib ticks from the original5pt
    # defaults. Restore the documented6pt tick tier after those resets.
    for axis in cartesian.values():
        axis.tick_params(axis='both',which='major',labelsize=6.0)
    # This shared label is figure text, so it does not follow the two bottom
    # spines automatically. Match their 3-pt shift without changing its words.
    for text in main.texts:
        if text.get_text().startswith('Mean likelihood per trial'):
            px,py=text.get_transform().transform(text.get_position())
            text.set_position(text.get_transform().inverted().transform((px,py-3*figure.dpi/72)))
    figure.canvas.draw()
    assert scientific_artist_snapshot() == frozen_artist_hash, 'Axis styling changed scientific artists'
    # A short real ACL endpoint tick preserves clear separation from its
    # existing manual label; all other major tick lengths are unchanged.
    acl_tick=next(tick for tick in fitting2.yaxis.get_major_ticks() if np.isclose(tick.get_loc(),0))
    acl_tick.tick1line.set_markersize(1.0)
    selected_y_axes={'b1':fitting1,'b2':fitting2,'f':phase}
    for axis in selected_y_axes.values():
        bound_spines_to_ticks(axis,sides=('left',))
    figure.canvas.draw()
    result['feedback_y_tick_endpoints_20261014']={
        'axes':audit_tick_bounded_spines(selected_y_axes,sides=('left',)),
        'b2_ACL_major_tick':0,'b2_ACL_tick_label':'','b2_ACL_tick_length_pt':1.0,
        'f_scale':'log','f_view_limits':list(phase.get_ylim()),
        'f_no_omitted_zero_marks':not hasattr(phase,'_publication_y_break'),
        'all_numeric_limits_and_source_artists_preserved':True,
        'scope':'Only b1/b2/f left y-spine endpoints and the real ACL endpoint tick; no fitting/statistical changes'}
    gap_qa = audit_detached_axes(cartesian)
    # i intentionally uses a native bracket from the first to last category
    # instead of the hidden ordinary bottom spine. Verify its actual endpoint
    # and the omission strokes as a separate categorical-baseline exception.
    reference_axis=cartesian['i']
    reference_bracket=next(line for line in reference_axis.lines
        if len(line.get_xdata())==4 and np.allclose(line.get_xdata(),[0,0,2,2])
        and np.allclose(line.get_ydata(),[-.045,-.015,-.015,-.045]))
    category_vertices=reference_bracket.get_transform().transform(reference_bracket.get_xydata())
    left_vertices=reference_axis.spines['left'].get_transform().transform(reference_axis.spines['left'].get_path().vertices)
    slashes=[line for line in reference_axis._publication_y_break_artists
             if str(line.get_gid()).startswith('publication-y-break-slash')]
    assert len(slashes)==2
    slash_vertices=[line.get_transform().transform(line.get_xydata()) for line in slashes]
    assert all(np.isclose(vertices[:,0].mean(),left_vertices[:,0].mean()) for vertices in slash_vertices)
    scale=25.4/figure.dpi
    category_gap=(category_vertices[:,0].min()-max(vertices[:,0].max() for vertices in slash_vertices))*scale
    category_gap-=(reference_bracket.get_linewidth()+slashes[0].get_linewidth())/2*25.4/72
    assert category_gap>=.5
    gap_qa['i'].update(reason='Original custom categorical bracket replaces hidden bottom spine; retains category bounds',
        actual_left_spine_position=list(reference_axis.spines['left'].get_position()),
        actual_omission_marks_follow_left_spine=True,
        custom_category_baseline_to_omission_visible_horizontal_gap_mm=float(category_gap))
    # Ordinary text/tick labels move with the true spines. Run the same native
    # clipping/collision check after those transforms have settled.
    base.final._render_qa(figure,base.AUDIT/'fig6_full_joint90_render_qa.json')
    from matplotlib.text import Text
    sizes=sorted({text.get_fontsize() for text in figure.findobj(Text)
                  if text.get_visible() and text.get_text().strip()})
    assert sizes==[6.0,7.0,8.0],sizes
    result['typography']['font_sizes_pt']=sizes
    result['tick_font_repair_20261014']={
        'visible_tick_fontsize_pt':6.0,'font_sizes_pt':sizes,
        'reason':'Outward spine changes recreate ticks from old5pt defaults; restore current6/7/8 tier',
        'b1_y_tick_pad_pt':2.0,
        'b1_long_tick_reason':'6pt fRL-decay retains full label, positive2pt tick padding and a1mm canvas margin via the recorded b1 layout allowance',
        'axis_label_display_anchors_preserved':fixed_label_anchors,
        'rendered_tick_boxes_before_pixels':tick_boxes_before,
        'rendered_tick_boxes_after_pixels':rendered_tick_boxes(),
        'categorical_views_unchanged':True,'axis_endpoints_unchanged_by_font_repair':True,
        'no_statistical_tests_fits_or_simulations':True}
    from matplotlib.patches import PathPatch
    f_boxes=[patch for patch in phase.patches if isinstance(patch,PathPatch)]
    result['category_label_spacing_20261014']={
        'f':compact_category_margin(phase,f_boxes),
        'i':{'status':'EXEMPT','reason':'Digitized seed scatter, mean diamonds and dashed source baseline lines; no categorical boxes/bars',
             'custom_category_baseline_unchanged':True}}
    assert result['category_label_spacing_20261014']['f']['categorical_view_translation']==0
    assert 6<=result['category_label_spacing_20261014']['f']['after']['ylabel_to_box_mm']<=12
    # Keep all established row/column checks at1.5pt. The approved b1 font
    # allowance intentionally narrows b1 relative to e1; verify their retained
    # right edge as a shared boundary instead of asserting equal left edges.
    from audit_panel_alignment import matplotlib_layout_manifest,audit_layout_manifest,write_json_report
    selection={name:cartesian[name] for name in ('b1','b2','c','d_upper','d_lower','e1','e2','f','g_upper','g_lower')}
    alignment_manifest=matplotlib_layout_manifest(figure,axes=list(selection.values()),
        panel_ids={axis:key for key,axis in selection.items()},
        row_groups=[{'id':'fit-tasks','panels':['b1','b2']},
                    {'id':'fit-and-attention-height','panels':['b1','b2','c']},
                    {'id':'third-row-performance-and-slopes','panels':['e1','e2','f']}],
        column_groups=[{'id':'dis-association','panels':['d_upper','d_lower']},
                       {'id':'round-trajectories','panels':['g_upper','g_lower']},
                       {'id':'attention-and-parameter-column','panels':['c','f']}],
        exemptions=[{'panels':['e2','f'],'checks':['panel-width','horizontal-gutter'],
                     'reason':'Two-task performance comparison and phase-distribution panel have intentionally different spans'},
                    {'panels':['b1','b2','c'],'checks':['panel-width','horizontal-gutter'],
                     'reason':'Task-specific fitting plots and attention panel retain distinct widths; match top/bottom/height'}])
    alignment_manifest['boundary_groups']=[{'id':'task1-retained-right-edge','panels':['b1','e1'],'edge':'right'}]
    alignment=audit_layout_manifest(alignment_manifest,tolerance_pt=1.5,gutter_tolerance_pt=1.5)
    assert alignment['verdict']=='PASS',alignment
    alignment['b1_e1_left_width_exception']=result['b1_font_layout_allowance_20261014']
    write_json_report(alignment,base.AUDIT/'fig6_full_joint90_alignment.json')
    result['alignment_status']=alignment['verdict']
    names=('a','b1','b2','c','d_upper','d_lower','e1','e2','g_upper','g_lower','f','h','i')
    final_content=base.measure_horizontal_spacing(figure,dict(zip(names,figure.axes)))
    for pair,gap in final_content['content_gaps_mm'].items():
        assert gap>=(1.0 if pair=='b1-b2' else 2.0),(pair,gap)
    result['horizontal_spacing_qa']['after']=final_content
    result['horizontal_spacing_qa']['font_sizes_changed']=True
    result['horizontal_spacing_qa']['b1_font_layout_allowance']=result['b1_font_layout_allowance_20261014']
    result['horizontal_spacing_qa']['method']+=' Final6pt tick restoration is measured after detachment; b1 gains1.7mm left margin while retaining its right edge and b1–b2 gutter.'
    (base.AUDIT/'fig6_full_joint90_horizontal_spacing.json').write_text(json.dumps(result['horizontal_spacing_qa'],indent=2)+'\n')
    result['detached_axes'] = {
        'status':'PASS','reference':'Fig4 tight outcomes 5 pt left / 3 pt bottom',
        'axes':gap_qa,'scientific_artist_hash_unchanged':True,
        'scientific_artist_sha256':frozen_artist_hash,
        'plot_area_positions_limits_ticks_values_unchanged':True,
        'exceptions':{'a':'Model architecture: no quantitative Cartesian axes',
                      'h':'Movie-recommendation schematic: no quantitative Cartesian axes',
                      'c_Full_range_inset':'Compact inset retains original corner transforms and no omission marks, as requested by the user'},
        'shared_fitting_xlabel_shift_pt':3.0,
        'data_sources':'Frozen existing CSVs; no fitting, simulation or statistical test executed',
    }
    (base.AUDIT/'detached_axes_20261014.json').write_text(json.dumps(result['detached_axes'],indent=2)+'\n')
    acl=next(text for text in fitting2.texts if text.get_text()=='ACL')
    renderer=figure.canvas.get_renderer()
    left_spine=fitting2.spines['left']
    spine_x=left_spine.get_transform().transform(left_spine.get_path().vertices)[:,0].mean()
    acl_gap=(spine_x-acl.get_window_extent(renderer).x1)*25.4/figure.dpi
    acl_gap-=left_spine.get_linewidth()/2*25.4/72
    assert acl_gap>=.8
    middle_tick=next(line for line in reference_axis.lines if line.get_gid()=='category-tick-1')
    np.testing.assert_array_equal(middle_tick.get_xdata(),[1,1])
    np.testing.assert_array_equal(middle_tick.get_ydata(),[-.015,-.045])
    e_gap=(performance2.bbox.x0-performance1.bbox.x1)*25.4/figure.dpi
    np.testing.assert_allclose(e_gap,7.36,atol=1e-10)
    b_gap=(fitting2.bbox.x0-fitting1.bbox.x1)*25.4/figure.dpi
    np.testing.assert_allclose(b_gap,9.36,atol=1e-10)
    xlabel=next(text.get_text() for text in main.texts if text.get_text().startswith('Mean likelihood per trial'))
    result['detail_polish_20261014']=dict(acl_to_yaxis_stroke_gap_mm=float(acl_gap),
        neural_ucb_middle_tick=True, fitting_xlabel=xlabel,
        fitting_plot_gap_mm=float(b_gap),previous_fitting_plot_gap_mm=10.36,
        fitting_task2_shift_left_mm=1,
        performance_plot_gap_mm=float(e_gap),previous_performance_plot_gap_mm=10.36,
        performance_task2_shift_left_mm=3, a_architecture_unchanged=True,
        a_internal_subplot_request='User clarified: tighten Task1/Task2 in b and e, not a')
    # Apply display wording after legacy geometry hooks keyed by old labels.
    from matplotlib.text import Text
    from matplotlib.ticker import FuncFormatter
    for item in figure.findobj(Text):
        if item.get_text()=='DIS ratio':item.set_text('% of DIS trials')
        elif item.get_text()=='Round':item.set_text('Trial')
    for axis in (performance1,performance2):
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{100*value:g}'))
    dis.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{100*value:g}'))
    figure.axes[4].yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{100*value:.0f}'))
    for key in 'cdfg':
        anchor=next(t for t in main.texts if t.get_text()==key)
        x,y=anchor.get_position()
        main.text(x+.034,y,'Task 1',fontsize=6,ha='left',va='top',color=base.final.INK)
    # Group headers sit above the plot, as in panel c; the range reflects data.
    fitting1.set_title('Task 1',loc='left',pad=4.,fontsize=6.)
    for ax,label in [(performance1,'Task 1 (4D)'),(performance2,'Task 2')]:
        for text in list(ax.texts):
            if text.get_text() in ('Task 1','Task 2'):text.remove()
        ax.set_title(label,loc='left',pad=4.,fontsize=6.)
    result['task_phase_scope']={'b1':'3D/4D-E/4D-NE','c':'3D/4D-E/4D-NE',
      'd':'3D/4D-E/4D-NE','e1':'4D; Human 4D-NE only',
      'f':'3D/4D-E/4D-NE','g':'3D/4D-E/4D-NE; available phases pooled'}
    figure.canvas.draw()
    result['display_revision_20261015']={
        'DIS_label':'% of DIS trials','percent_display_factor':100,
        'raw_artist_coordinates_and_intervals_unchanged':True,
        'time_axis':'Trial','others_color':'#8D9399'}
    base.final._render_qa(figure,base.AUDIT/'fig6_full_joint90_render_qa.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        layout.OUTPUT_DIR = args.output_dir
        layout.MANIFEST_DIR = args.output_dir/PACKAGE.name
        base.AUDIT = layout.MANIFEST_DIR
        base.AUDIT.mkdir(parents=True, exist_ok=True)
    protected = []  # Original archives are outside this portable plotting package.
    before = {str(p):sha256(p.read_bytes()).hexdigest() for p in protected}
    output, manifest = layout.assemble(output_stem=PACKAGE.name,figure_finalize=finalize)
    assert output.resolve() not in {p.resolve() for p in protected}
    after = {str(p):sha256(p.read_bytes()).hexdigest() for p in protected}
    preservation = {'same_before_and_after':before == after, 'before':before,'after':after,
                    'scope':'Original PNG hashes during this renderer invocation; later upstream edits are separate.'}
    assert before == after, 'Original artwork changed concurrently during rendering'
    (layout.MANIFEST_DIR/'original_artwork_preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
    record = json.loads(manifest.read_text())
    record['vector_panel_a'].update(bounds='Full-width180 x49 mm native header',
        structure='Original full-DGEM stimulus-to-action architecture; beta initial strength is fitted for b Task1 and Task2/c/d/f/g',
        standalone_image_dependency=False)
    record['panel_statistics']['methods'] = ['d: participant-paired two-sided Wilcoxon',
        'f: seven-parameter free-beta phase comparisons, Wilcoxon/Mann-Whitney with Holm(3)']
    record['panel_statistics']['f_tests'] = str(layout.DATA_PATHS['slope_tests'])
    record['notes'] = [note for note in record['notes'] if 'Panel f shows' not in note]
    record['notes'].append('Panel f boxes use all156 seven-parameter free-beta participant-phase slopes; individual points/fliers and paired visual lines are hidden; new tests retain participant pairing.')
    record['notes'].append('Panel h highlights selected Movie B within the candidate list using the feature accent; no separate Action box. Learning details remain in Methods.')
    record['notes'].append('Task2 fits use the unordered four-of-sixteen event (chance 1/1820): retained DGEM grouped PL versus V2 ActionScore-softmax alternatives. Saved Holm(3) tests compare three behavioural alternatives; the separate ACL test includes all80 participants/4111 trials with real gaze or uniform fallback. Shared connector does not pool tests.')
    record['v2_be_revision']=json.loads((PACKAGE/'v2_be_revision.json').read_text())
    record['notes'].append('Original filled marks retained at user request. Task2 e: V2 deterministic-feedback simulations,80 episodes/model; retained DGEM noisy-feedback and Human references; distinction in caption. No equal-environment inference; no tests, fits or simulations recomputed.')
    manifest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'output':str(output),'manifest':str(manifest)},indent=2))


if __name__=='__main__':
    main()
