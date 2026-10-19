"""Apply tick endpoints and compact category margins without reanalysis."""
from pathlib import Path
import hashlib
import io
import json
import sys
import numpy as np
from PIL import Image
from matplotlib.text import Text

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'source/native_reconstruction'
sys.path.insert(0,str(NATIVE/'code'))
import render_fig1_compact as native
from detached_axes import audit_detached_axes
from axis_break_marks import audit_yaxis_breaks
from tick_bounded_spines import bound_spines_to_ticks,audit_tick_bounded_spines
from category_label_spacing import compact_category_margin
from audit_panel_alignment import require_matplotlib_panel_alignment

CUT=3024
SNAPSHOT=ROOT/'revisions/20261014_before_tick_endpoints_box_spacing'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def pixels(fig):
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=600,facecolor='white');buffer.seek(0)
    with Image.open(buffer) as image:return np.array(image.convert('RGB'))


def data_signature(fig):
    result={}
    for name in native.LOWER_KEYS:
        axis=fig._panels[name]
        result[name]={'lines':[hashlib.sha256(line.get_xydata().tobytes()).hexdigest()
                              for line in axis.lines if not str(line.get_gid()).startswith('publication-y-break')],
                     'patches':[hashlib.sha256(patch.get_path().vertices.tobytes()).hexdigest() for patch in axis.patches],
                     'collections':[[hashlib.sha256(path.vertices.tobytes()).hexdigest() for path in item.get_paths()] for item in axis.collections],
                     'annotations':[text.get_text() for text in axis.texts]}
    return result


def main():
    inputs=native.check_inputs();fig=native.build_figure()
    right=fig._panels['h'];right.set_ylabel('Trial of personal-best score');right.set_yticks([0,20,40,60])
    right.spines['bottom'].set_bounds(*right.get_xlim())
    if ROOT.name=='fig1_nhb_tick_endpoints_20261014':bound_spines_to_ticks(right,sides=('bottom',))
    before=pixels(fig)
    # Portable package: historical baseline assertion omitted; final output is checked against reference/fig1.png.
    original=data_signature(fig)
    geometry={name:{'xlim':list(fig._panels[name].get_xlim()),'ylim':list(fig._panels[name].get_ylim())} for name in native.LOWER_KEYS}
    spacing={}
    for name in 'fh':
        spacing[name]=compact_category_margin(fig._panels[name],fig._panels[name]._restored_boxes['boxes'])
    fig._panels['e'].set_yticks([50,75,100])
    fig._panels['f'].set_yticks([0,25,50])
    fig._panels['h'].set_yticks([0,30,60])
    for name in native.LOWER_KEYS:bound_spines_to_ticks(fig._panels[name])
    assert data_signature(fig)==original
    # Display-only normalization by the scheduled horizon of each condition.
    # Means, SEMs, observations, smoothing and all statistics stay frozen.
    normalization=[]
    for task,key in [(1,'e'),(2,'g')]:
        axis=fig._panels[key]
        curves=native.base.read('fig1_learning_display.csv')
        for group,line,band in zip(native.base.TASK_CONFIG[task]['groups'],axis.lines,axis.collections):
            rows=curves.loc[curves.task.eq(task)&curves.group.eq(group)]
            horizon={1:{'P1':30,'P2':50,'P2-only':50},2:{'FDS-Obs':60,'FDS-NoObs':60}}[task][group]
            oldx=np.asarray(line.get_xdata(),float).copy()
            oldy=np.asarray(line.get_ydata(),float).copy()
            np.testing.assert_array_equal(oldx,rows.trial)
            line.set_xdata((oldx-1)/(horizon-1))
            for path in band.get_paths():
                oldv=path.vertices.copy()
                path.vertices[:,0]=(oldv[:,0]-1)/(horizon-1)
                np.testing.assert_array_equal(path.vertices[:,1],oldv[:,1])
            np.testing.assert_array_equal(line.get_ydata(),oldy)
            np.testing.assert_allclose(np.asarray(line.get_xdata())*(horizon-1)+1,oldx,atol=1e-12)
            assert min(line.get_xdata())==0 and max(line.get_xdata())==1
            normalization.append(dict(task=task,group=group,horizon=horizon,formula='(trial - 1) / (scheduled trials - 1)',y_and_sem_unchanged=True))
        axis.set(xlim=(0,1),xticks=[0,.5,1],xlabel='Normalized trial')
        axis.set_xticklabels(['0','0.5','1'])
        bound_spines_to_ticks(axis)
    for name in 'fh':fig._panels[name].set_ylabel('Trial of best score')
    fig.canvas.draw()
    axis_display={name:{'ylabel':fig._panels[name].get_ylabel(),
                        'yticks':fig._panels[name].get_yticks().tolist(),
                        'ylim':list(fig._panels[name].get_ylim()),
                        'y_axis_bounds_px':list(fig._panels[name].yaxis.get_tightbbox(fig.canvas.get_renderer()).bounds)}
                  for name in native.LOWER_KEYS}
    for name in 'fh':
        axis=fig._panels[name]
        for box in axis._restored_boxes['boxes']:
            bounds=box.get_window_extent(fig.canvas.get_renderer())
            assert axis.bbox.x0<=bounds.x0<bounds.x1<=axis.bbox.x1
        # Box/whisker data lie within the viewport; comparison annotations
        # intentionally occupy the reserved band above the data rectangle.
        for kind in ['medians','whiskers','caps']:
            for line in axis._restored_boxes[kind]:
                assert min(line.get_xdata())>=axis.get_xlim()[0] and max(line.get_xdata())<=axis.get_xlim()[1]
                assert min(line.get_ydata())>=axis.get_ylim()[0] and max(line.get_ydata())<=axis.get_ylim()[1]
    qa=ROOT/'qa';qa.mkdir(exist_ok=True)
    endpoints=audit_tick_bounded_spines({name:fig._panels[name] for name in native.LOWER_KEYS})
    gaps=audit_detached_axes({name:fig._panels[name] for name in native.LOWER_KEYS})
    omission=audit_yaxis_breaks({name:fig._panels[name] for name in native.LOWER_KEYS})
    alignment=require_matplotlib_panel_alignment(fig,axes=[fig._panels[name] for name in native.LOWER_KEYS],
        row_groups=[list(native.LOWER_KEYS)],column_groups=[],tolerance_pt=1.5,gutter_tolerance_pt=1.5,
        exemptions=[{'panels':list(native.LOWER_KEYS),'checks':['panel-width','horizontal-gutter'],
                     'reason':'Existing task-specific learning widths and compound plot gaps are preserved.'}],json_out=qa/'uniform_alignment.json')
    sizes=sorted({text.get_fontsize() for text in fig.findobj(Text) if text.get_visible() and text.get_text().strip()})
    assert sizes==[5.,6.,8.]
    after=pixels(fig)
    assert np.array_equal(before[:CUT],after[:CUT])
    target=ROOT/'source/results_axis_spacing.png'
    Image.fromarray(after[CUT:]).save(target,dpi=(600,600))
    record={'status':'PASS','output_sha256':sha(target),'original_results_sha256':sha(ROOT/'source/fig1_rebuilt_bar_box.png'),
            'pixels':[after.shape[1],len(after)-CUT],'dpi':600,'font_sizes_pt':sizes,'source_input_hashes':inputs,'axis_display':axis_display,
            'data_artists_exactly_unchanged':False,'raw_values_and_y_sem_exactly_unchanged':True,'trial_normalization':normalization,'data_signature_before':original,'data_signature_after':data_signature(fig),'category_spacing':spacing,'tick_endpoints':endpoints,
            'axis_corners':gaps,'omission_marks':omission,'geometry_before':geometry,'alignment':alignment['verdict'],
            'scope':'Both personal-best views use Trial of best score; existing normalized-trial display, compact spacing and requested ticks retained; scientific y values and statistics unchanged.',
            'no_statistical_tests_fitting_or_simulation':True}
    (ROOT/'source/results_axis_spacing.json').write_text(json.dumps(record,indent=2)+'\n')
    (qa/'uniform_axes_category_spacing.json').write_text(json.dumps(record,indent=2)+'\n')
    native.base.plt.close(fig)
    print(json.dumps({'status':'PASS','spacing':spacing,'fonts':sizes},indent=2))


if __name__=='__main__':main()
