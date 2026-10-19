"""Render a separate evidence-led Fig. 5; keep the current primary PNG intact."""
from pathlib import Path
from axis_break_marks import mark_nonzero_yaxis, audit_yaxis_breaks
from detached_axes import detach_axes, audit_detached_axes
from tick_bounded_spines import bound_spines_to_ticks, audit_tick_bounded_spines
from category_label_spacing import compact_category_margin
import sys,json,hashlib
from matplotlib.text import Text
ROOT=Path(__file__).resolve().parents[1]
import native_style as final
import figure5_evidence_layout as layout
PKG=ROOT/'data/old5'
layout.CAPTION_PATH=PKG/'caption.md'
layout.METHODS_PATH=PKG/'methods.md'
layout.original.DPI=600
STEM='fig5_nhb_evidence_layout'
# Keep statistical panels nearly square, preserving both rows' existing top edges.
STAT_PLOT_HEIGHT_MM=30.0
Y_LABELS={
    'b':'Fixated dimension',
    'c':'Gaze area\n(deg²)',
    'd':'Mean dwell time\nper image (ms)',
    'e':'Gaze-shift speed\n(deg s⁻¹)',
    'f':'Fixated dimensions',
    'g':'Pre-task dimensions\nfixated',
    'h':'Initially noticed\ndimensions',
    'i':'Dimensions explored\nvia DIS',
    'j':'Score-relevant\ndimensions',
}
previous_height=layout.PLOT_HEIGHT
layout.PLOT_HEIGHT=STAT_PLOT_HEIGHT_MM/141
layout.ROW_BOTTOMS=tuple(bottom if row==0 else bottom+previous_height-layout.PLOT_HEIGHT
                        for row,bottom in enumerate(layout.ROW_BOTTOMS))

def scientific_marks(panels):
    """Fingerprint data-space marks, independent of spine and layout transforms."""
    import numpy as np
    def digest(values):
        a = np.asarray(values, dtype=np.float64)
        return {'shape':list(a.shape),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
    result = {}
    for name, axis in panels.items():
        lines = [digest(line.get_xydata()) for line in axis.lines
                 if not str(line.get_gid()).startswith('publication-y-break')]
        collections = [{'offsets':digest(c.get_offsets()),
                        'paths':[digest(path.vertices) for path in c.get_paths()],
                        'segments':[digest(seg) for seg in c.get_segments()] if hasattr(c,'get_segments') else []}
                       for c in axis.collections]
        result[name] = {'lines':lines,'collections':collections,
            'patches':[digest(patch.get_path().vertices) for patch in axis.patches],
            'xlim':list(axis.get_xlim()),'ylim':list(axis.get_ylim()),
            'xticks':axis.get_xticks().tolist(),'yticks':axis.get_yticks().tolist()}
    return result


def hide_box_fliers_and_adapt_view(panels):
    """Hide only native flier artists; retain every array and summary artist."""
    import numpy as np
    settings = {'g': ((0., 8.), [0, 2, 4, 6, 8]),
                'h': ((0., 20.), [0, 5, 10, 15, 20]),
                'i': ((1.5, 5.3), [2, 3, 4, 5])}
    reports = {}
    for name, (limits, ticks) in settings.items():
        axis = panels[name]
        artists = axis._publication_boxplot_artists
        groups = axis._publication_boxplot_source_groups
        before_limits, before_ticks = list(axis.get_ylim()), axis.get_yticks().tolist()
        records = []
        for index, (values, flier) in enumerate(zip(groups, artists['fliers'])):
            coordinates = np.asarray(flier.get_xydata(), dtype=float).copy()
            flier.set_visible(False)
            np.testing.assert_array_equal(flier.get_xydata(), coordinates)
            whiskers = [line.get_xydata().tolist() for line in artists['whiskers'][2*index:2*index+2]]
            visible_y = np.concatenate([np.asarray(points)[:, 1] for points in whiskers])
            records.append({'group': ('Obs', 'NObs')[index], 'n_in_all_summaries_and_tests': len(values),
                'raw_values_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
                'box_vertices': artists['boxes'][index].get_path().vertices.tolist(),
                'whisker_coordinates': whiskers,
                'whisker_range': [float(visible_y.min()), float(visible_y.max())],
                'median_coordinates': artists['medians'][index].get_xydata().tolist(),
                'flier_coordinates_retained': coordinates.tolist(), 'hidden_flier_count': len(coordinates),
                'flier_visible': flier.get_visible()})
        axis._publication_showfliers = False
        axis.set_ylim(limits)
        axis.set_yticks(ticks)
        for text in axis.get_yticklabels():
            text.set_fontsize(5)
            text.set_fontweight('normal')
        reports[name] = {'showfliers': False, 'implementation': 'Existing native flierLine2D visibility=False; arrays retained',
            'before_ylim': before_limits, 'before_yticks': before_ticks,
            'ylim': list(limits), 'yticks': ticks, 'groups': records,
            'native_mean_coordinates': axis._publication_boxplot_mean_artist.get_xydata().tolist(),
            'range_reason': ('Visible source whiskers still require the original rounded zero-based range'
                if name in 'gh' else 'Whiskers2–5 retained with lower/upper display padding; omitted zero marked'),
            'all_observations_in_summaries_and_tests': True}
    return reports


def finalize(figure,panels,labels,statistics):
    scientific_before = scientific_marks(panels)
    for name in "cdef":
        pos = panels[name].get_position()
        panels[name].set_position((pos.x0,pos.y0-2.1/141,pos.width,pos.height))
    final._recolor(figure)
    from matplotlib.collections import PathCollection
    import numpy as np
    analysis_path=PKG/'fig5j_change_analysis.json'
    change=json.loads(analysis_path.read_text())
    assert change['input_sha256']==hashlib.sha256(layout.original.DATA_PATHS['timecourse_values'].read_bytes()).hexdigest()
    within_path=PKG/'fig5j_within_group_analysis.json'
    within=json.loads(within_path.read_text())
    assert within['input_sha256']==change['input_sha256']
    assert within['paired_subject_changes_sha256']==hashlib.sha256((PKG/'fig5j_subject_changes.csv').read_bytes()).hexdigest()
    within_marks={}
    legend=panels['j'].get_legend()
    # Two columns identify the group-specific Early–Final comparisons directly.
    for text,name in zip(legend.get_texts(),('Obs','NObs')):
        p=within['groups'][name]['p_value_adjusted']
        mark=layout.original._stars(p) if p<.05 else f'P = {p:.3g}'
        within_marks[name]=mark
        display=rf'$\mathbf{{{mark}}}$' if p<.05 else mark
        text.set_text(f'{name}\n{display}')
    handles,_=panels['j'].get_legend_handles_labels()
    legend=panels['j'].legend(handles,[text.get_text() for text in legend.get_texts()],
                             frameon=False,loc='lower center',bbox_to_anchor=(.5,1.02),
                             ncol=2,fontsize=5,handlelength=.8,handletextpad=.25,
                             columnspacing=.65,borderaxespad=0,borderpad=0)
    change_row={'panel':'j-change','comparison':change['contrast'],
                'n_left':change['n_obs'],'n_right':change['n_nobs'],
                'left_mean':change['obs_mean_change'],'right_mean':change['nobs_mean_change'],
                'p_value':change['p_value'],'p_value_displayed':change['p_value'],
                'test':change['test'],'p_adjustment':change['p_adjustment'],
                'source':'timecourse_subject_values.csv','display_significance':'Caption and Results only',
                'change_difference_obs_minus_nobs':change['effect_obs_minus_nobs'],
                'change_difference_ci95_low':change['ci95'][0],
                'change_difference_ci95_high':change['ci95'][1],
                't_statistic':change['t_statistic'],'degrees_of_freedom':change['degrees_of_freedom']}
    for key in change_row:
        if key not in statistics.columns:statistics[key]=np.nan
    assert not statistics['panel'].eq('j-change').any()
    statistics.loc[len(statistics)]=change_row
    for name in ('Obs','NObs'):
        group=within['groups'][name]
        row={'panel':f'j-{name}','comparison':f'Final versus Early within {name}',
             'n_left':group['n'],'n_right':group['n'],'left_mean':group['early_mean'],
             'right_mean':group['final_mean'],'p_value':group['p_value_raw'],
             'p_value_displayed':group['p_value_adjusted'],'test':within['test'],
             'p_adjustment':within['p_adjustment'],'display_significance':within_marks[name],
             'source':'timecourse_subject_values.csv','effect_mean_right_minus_left':group['mean_change'],
             'effect_ci95_low':group['mean_change_ci95'][0],'effect_ci95_high':group['mean_change_ci95'][1],
             't_statistic':group['t_statistic'],'degrees_of_freedom':group['degrees_of_freedom']}
        statistics.loc[len(statistics)]=row
    for name in 'cdefghi':
        axis=panels[name]
        for patch in axis.patches:
            rgb=patch.get_facecolor()[:3]
            patch.set_alpha(None)
            patch.set_facecolor((*rgb,.40 if name in 'cdef' else .58))
            patch.set_edgecolor('#4C535A')
            patch.set_linewidth(.75)
        if name in 'cdef':
            for line in axis.lines:
                if line.get_alpha()==.065:
                    line.set_alpha(.14)
                    line.set_linewidth(.40)
                elif np.isclose(line.get_linewidth(),.55):
                    line.set_linewidth(.75)
                elif np.isclose(line.get_linewidth(),1.05):
                    line.set_linewidth(1.10)
            for collection in axis.collections:
                if isinstance(collection,PathCollection) and collection.get_alpha()==.32:
                    collection.set_alpha(.48)
    for line in panels['j'].lines:
        if np.isclose(line.get_linewidth(),.85):line.set_linewidth(1.0)
    for collection in panels['j'].collections:collection.set_linewidth(.90)
    # Keep units and distinguishing context; counts/report modality are in the
    # caption and source cue instead of repeating them in every axis title.
    for name,label in Y_LABELS.items():
        panels[name].set_ylabel(label,labelpad=2.5)
    for text in figure.texts:
        if text.get_text() in ('DIS trials','others'):text.set_color(final.INK)
    final._readable_axes([axis for axis in figure.axes if axis.axison])
    figure.canvas.draw()
    for text in figure.findobj(Text):
        text.set_fontsize(8 if text.axes is None and text.get_text().strip() in tuple('abcdefghij')
                          else 6 if text.get_fontsize()>=6 else 5)
    figure.canvas.draw()
    renderer=figure.canvas.get_renderer()
    top=.944
    width=44.0/180
    height=(44.0*577/770)/141
    target_bottom=top-height
    image_axes=[axis for axis in figure.axes if axis.images and axis is not panels['b']]
    assert len(image_axes)==2
    for axis,left in zip(image_axes,(.070,.330)):
        axis.set_position((left,target_bottom,width,height))
    for text in figure.texts:
        if text.get_text() in ('DIS trials','others'):
            left=.070 if text.get_text()=='DIS trials' else .330
            text.set_x(left+width/2)
            if text.get_text()=='DIS trials':
                text.set_text('DIS trials (Plant)')
    density=next(axis for axis in figure.axes if axis.get_label()=='fig5a_shared_colorbar')
    density.set_position((.330+width+.010,target_bottom,.008,height))
    density.yaxis.set_label_position('right')
    density.yaxis.label.set_rotation(270)
    density.yaxis.set_label_coords(2.3,.5)
    density.yaxis.label.set_fontsize(5)
    # Put the qualitative endpoints in the shared vertical gap to the right.
    heatmap_side_mm=30.6
    heatmap_left=.710+(37.0-heatmap_side_mm)/360  # Preserve the previous centre.
    panels['b'].set_aspect('equal',adjustable='box')
    panels['b'].set_position((heatmap_left,top-heatmap_side_mm/141,
                              heatmap_side_mm/180,heatmap_side_mm/141))
    diagonal=[patch for patch in panels['b'].patches if patch.get_gid()=='diagonal-highlight']
    assert len(diagonal)==5
    for patch in diagonal:
        patch.set_edgecolor('white')
        np.testing.assert_allclose(patch.get_edgecolor(),(1,1,1,1))
    selectivity_bar=next(axis for axis in figure.axes if axis.get_label()=='fig5c_colorbar')
    selectivity_bar.set_position((heatmap_left+heatmap_side_mm/180+.010,
        panels['b'].get_position().y0,.008,panels['b'].get_position().height))
    # Match a's vertical label; keep this numerical key beyond all four tick labels.
    selectivity_bar.set_title('',loc='right')
    selectivity_bar.yaxis.set_label_position('right')
    selectivity_bar.set_ylabel('Dwell-time proportion',fontsize=5,fontweight='normal',
                              rotation=270,labelpad=6,va='center')
    for text in figure.texts:
        if text.get_text()=='a':text.set_x(.050)
        if text.get_text()=='b':text.set_x(.635)
    for axis in figure.axes:
        if axis.get_label().startswith('source_eye_tracking_') and abs(axis.get_position().y1-.985)<.01 and axis.get_position().x0<.2:
            pos=axis.get_position()
            axis.set_position((.050+3.8/180,pos.y0,pos.width,pos.height))
    for axis in figure.axes:
        if axis.get_label().startswith('source_eye_tracking_') and abs(axis.get_position().y1-.985)<.01 and axis.get_position().x0>.6:
            pos=axis.get_position()
            axis.set_position((.635+3.8/180,pos.y0,pos.width,pos.height))
    figure.canvas.draw()
    renderer=figure.canvas.get_renderer()
    for axis in image_axes:
        assert abs(axis.get_window_extent(renderer).y1-panels['b'].get_window_extent(renderer).y1)<.5
    image_audit={'reference_proportions':True,'top_aligned_to_b':True,'height_mm':height*141,
                 'bounds':[list(axis.get_position().bounds) for axis in image_axes],
                 'titles':['DIS trials (Plant)','others'],
                 'source_aspect_ratio_preserved':True,'b_size_mm':[heatmap_side_mm,heatmap_side_mm]}
    matrix_geometry={'size_mm':[heatmap_side_mm,heatmap_side_mm],'cell_aspect_ratio':1.0,
                     'diagonal_outline_rgba':[list(p.get_edgecolor()) for p in diagonal],
                     'outline_width_pt':.65,'unchanged_colour_range':[0,.3]}
    measured=panels['b'].get_window_extent(renderer)
    assert abs(measured.width-measured.height)<.1
    label_box=selectivity_bar.yaxis.label.get_window_extent(renderer)
    tick_boxes=[text.get_window_extent(renderer) for text in selectivity_bar.get_yticklabels()
                if text.get_visible() and text.get_text().strip()]
    assert all(label_box.x0>bbox.x1 for bbox in tick_boxes)
    assert label_box.x1<figure.bbox.x1
    colorbar_audit={'label':'Dwell-time proportion','placement':'Right of tick labels, vertically centred',
                   'rotation_degrees':270,'fontsize_pt':5,
                   'tick_label_gap_mm':(label_box.x0-max(b.x1 for b in tick_boxes))*25.4/figure.dpi,
                   'right_canvas_margin_mm':(figure.bbox.x1-label_box.x1)*25.4/figure.dpi,
                   'normalization':[0,.3],'tick_values':[0,.1,.2,.3]}
    annotation=layout._align_bottom_annotations(figure,panels)
    legend_bounds=legend.get_window_extent(figure.canvas.get_renderer())
    frame_bounds=panels['j'].get_window_extent(figure.canvas.get_renderer())
    assert legend_bounds.y0>frame_bounds.y1
    plot_bounds={name:list(panels[name].get_position().bounds) for name in 'cdefghij'}
    heights={name:bounds[3]*141 for name,bounds in plot_bounds.items()}
    widths={name:bounds[2]*180 for name,bounds in plot_bounds.items()}
    np.testing.assert_allclose(list(heights.values()),STAT_PLOT_HEIGHT_MM,atol=.01)
    np.testing.assert_allclose(list(widths.values()),27.9,atol=.01)
    renderer=figure.canvas.get_renderer()
    icons=[axis for axis in figure.axes if axis.get_label().startswith('source_')]
    label_icon_overlaps=[{'panel':name,'icon':icon.get_label()} for name in 'cdefghij' for icon in icons
                         if panels[name].yaxis.label.get_window_extent(renderer).overlaps(icon.get_window_extent(renderer))]
    assert not label_icon_overlaps,label_icon_overlaps
    geometry={'plot_heights_mm':heights,'plot_widths_mm':widths,
              'row_top_edges_mm_from_bottom':{row:[panels[name].get_position().y1*141 for name in row]
                                              for row in ('cdef','ghij')},
              'top_edges_preserved':False,'middle_row_downward_translation_mm':2.1,'all_eight_equal_height':True,
              'ylabel_icon_overlaps':label_icon_overlaps,
              'wrapped_ylabels':{name:label for name,label in Y_LABELS.items() if '\n' in label},
              'y_axis_labels':{name:panels[name].get_ylabel() for name in Y_LABELS}}
    box_flier_display = hide_box_fliers_and_adapt_view(panels)
    for name in 'cdefghij':
        mark_nonzero_yaxis(panels[name],panel=name)
        detach_axes(panels[name])
    # Use the user's exact sparse tick set, retaining the100..620 viewport
    # and the existing100/600 numerical spine endpoints.
    panels['c'].set_yticks([100,350,600])
    for text in panels['c'].get_yticklabels():
        text.set_fontsize(5)
        text.set_fontweight('normal')
    from matplotlib.patches import PathPatch
    category_spacing={}
    for name in 'cdefghi':
        boxes=[patch for patch in panels[name].patches if isinstance(patch,PathPatch)]
        category_spacing[name]=compact_category_margin(panels[name],boxes)
        assert 6<=category_spacing[name]['after']['ylabel_to_box_mm']<=12
    for name in 'cdefghij':
        bound_spines_to_ticks(panels[name])
    figure.canvas.draw()
    tick_endpoints=audit_tick_bounded_spines({name:panels[name] for name in 'cdefghij'})
    breaks=audit_yaxis_breaks({name:panels[name] for name in 'cdefghij'})
    detached = audit_detached_axes({name:panels[name] for name in 'cdefghij'})
    # Left-outward spines move the top y ticks left, toward the source cues.
    # Keep the same physical letter/icon offset and lift the middle-row header
    # into a proper annotation band, matching the existing bottom-row spacing.
    icons = [axis for axis in figure.axes if axis.get_label().startswith('source_')]
    assert len(icons) == 10
    icon_by_panel = dict(zip('abcdefghij', icons))
    header_gap_mm = 5.898
    for name in 'cdefghij':
        target_top = panels[name].get_position().y1 + header_gap_mm/141
        letter = next(t for t in figure.texts if t.get_text() == name)
        letter.set_y(target_top)
        labels[name] = (labels[name][0], target_top)
        icon = icon_by_panel[name]
        pos = icon.get_position()
        icon.set_position((pos.x0, target_top-pos.height, pos.width, pos.height))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    icon_text_collisions = []
    for name, icon in icon_by_panel.items():
        if name not in 'cdefghij':
            continue
        rect = icon.get_window_extent(renderer)
        for text in figure.findobj(Text):
            if not text.get_visible() or not text.get_text().strip():
                continue
            if rect.overlaps(text.get_window_extent(renderer)):
                icon_text_collisions.append((name, text.get_text()))
    if icon_text_collisions:
        (PKG/'icon_collision_diagnostic.json').write_text(json.dumps({
            'icons':{name:(icon.get_window_extent(renderer).extents*25.4/figure.dpi).tolist()
                     for name,icon in icon_by_panel.items()},
            'text_boxes':[{'text':t.get_text(),'box_mm':(t.get_window_extent(renderer).extents*25.4/figure.dpi).tolist()}
                          for t in figure.findobj(Text) if t.get_visible() and t.get_text().strip()
                          and t.get_text() in {text for _,text in icon_text_collisions}]
        },indent=2))
    assert not icon_text_collisions, icon_text_collisions
    header_audit = {'header_above_plot_mm': header_gap_mm,
        'letter_to_icon_left_mm':3.8, 'icon_text_collisions':icon_text_collisions,
        'source_icon_bounds_fraction':{name:list(icon.get_position().bounds)
            for name,icon in icon_by_panel.items()}}

    from matplotlib.patches import PathPatch
    box_mark_geometry={}
    for name in 'cfgh':
        axis=panels[name]
        box_widths=[patch.get_window_extent(figure.canvas.get_renderer()).width*25.4/figure.dpi
                    for patch in axis.patches if isinstance(patch,PathPatch)]
        assert len(box_widths)==2
        np.testing.assert_allclose(box_widths,4.95 if name in 'cf' else 4.35,atol=.01)
        category_gap=float(axis.transData.transform((1,0))[0]-axis.transData.transform((0,0))[0])*25.4/figure.dpi
        np.testing.assert_allclose(category_gap,15.,atol=.01)
        box_mark_geometry[name]={'box_widths_mm':box_widths,'category_gap_mm':category_gap}
    scientific_after = scientific_marks(panels)
    # Native source coordinates, physical widths, intervals and statistics stay.
    # Category translations and explicit g/h/i display-y exceptions are audited.
    for name in panels:
        before_marks=dict(scientific_before[name])
        after_marks=dict(scientific_after[name])
        if name in category_spacing:
            np.testing.assert_allclose(after_marks.pop('xlim'),category_spacing[name]['after']['xlim'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(before_marks.pop('xlim'),category_spacing[name]['before']['xlim'],rtol=0,atol=1e-12)
        if name=='c':
            assert before_marks.pop('yticks')==[150,300,450,600]
            assert after_marks.pop('yticks')==[100,350,600]
        if name in box_flier_display:
            assert before_marks.pop('ylim') == box_flier_display[name]['before_ylim']
            assert after_marks.pop('ylim') == box_flier_display[name]['ylim']
            assert before_marks.pop('yticks') == box_flier_display[name]['before_yticks']
            assert after_marks.pop('yticks') == box_flier_display[name]['yticks']
        assert before_marks==after_marks,(name,'Style revision changed data-space marks')
    qa=final._render_qa(figure,ROOT/'output/qa/old5_native_render_qa.json')
    sizes=sorted({t.get_fontsize() for t in figure.findobj(Text) if t.get_visible() and t.get_text().strip()})
    assert sizes==[5,6,8]
    return {'scientific_marks_unchanged':True,'all_box_whisker_median_mean_and_flier_arrays_unchanged':True,'box_flier_display':box_flier_display,'scientific_mark_fingerprints':scientific_after,'category_label_spacing':category_spacing,'tick_bounded_spines':tick_endpoints,'endpoint_tick_addition':{'c':{'before':[150,300,450,600],'after':[100,350,600],'reason':'User-selected exact sparse ticks; numerical view100..620 and source marks unchanged'}},'statistical_header_layout':header_audit,'detached_axes':detached,'detached_axes_exceptions':{'a':'Source-image frames and qualitative colorbar retain their native geometry','b':'Matrix frame, square cells and numerical colorbar retain native geometry'},'category_spacing_exceptions':{'a':'Source images and qualitative colourbar','b':'Matrix frame and numerical colourbar','j':'Mean/SEM trajectory with stage coordinates; no categorical boxes/bars'},'box_mark_geometry':box_mark_geometry,'panel_b_matrix_geometry':matrix_geometry,'y_axis_breaks':breaks,'panel_a_alignment':image_audit,'panel_b_colorbar_label':colorbar_audit,'statistical_panel_geometry':geometry,'primary_style_matched':{'paired_box_alpha':.40,'observation_box_alpha':.58,'paired_line_alpha':.14,'paired_point_alpha':.48},'font_sizes_pt':sizes,'font_floor_pt':qa['font_floor_pt'],'bottom_row_height':annotation,
            'panel_order_from_primary':{'a':'a','b':'c','c':'b','d':'d','e':'e','f':'f','g':'h','h':'i','i':'g','j':'j'},
            'panel_j_change_contrast':{'analysis_path':str(analysis_path.resolve()),
                                     'analysis_sha256':hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
                                     'annotation':'Reported in caption and Results; no panel annotation','p_value':change['p_value'],
                                     'effect_obs_minus_nobs':change['effect_obs_minus_nobs'],'ci95':change['ci95'],
                                     'analysis_status':change['analysis_status']},
            'panel_j_within_group_annotations':{'analysis_path':str(within_path.resolve()),
                                              'analysis_sha256':hashlib.sha256(within_path.read_bytes()).hexdigest(),
                                              'annotations':within_marks,'correction':within['p_adjustment'],
                                              'legend_anchor_axes':panels['j'].transAxes.inverted().transform(legend.get_bbox_to_anchor().p0).tolist(),
                                              'legend_clearance_mm':(legend_bounds.y0-frame_bounds.y1)*25.4/figure.dpi,
                                              'brackets':0}}
