"""Explicit release validation entry point; never runs on ordinary launch."""
import json
import time
from pathlib import Path
from PySide6.QtCore import QTimer
from .models import ReconMethod
from .runtime import APP_ROOT

def validate_release(app,window,config_path):
    config=json.loads(Path(config_path).read_text(encoding='utf-8-sig'));root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    if config.get('manual_only'):return validate_manual_release(app,window,config,root)
    report={'manual':False,'automatic':False,'data_root':str(APP_ROOT)}
    def finish():
        (root/'frozen_validation.json').write_text(json.dumps(report,indent=2));window.close();QTimer.singleShot(800,app.quit)
    def manual_done(ok,message):
        report['manual']=bool(ok and window.results and window.result_view._pixmap is not None);report['manual_message']=message
        report['manual_log']=window.log_box.toPlainText()
        if not report['manual']:finish();return
        window.main_tabs.setCurrentIndex(1);window.analysis_input.setText(config['input'])
        window.settings.update(output_root=str(root/'automatic'),gpu=config.get('gpu',0),start_row=1021,end_row=1026,
            range_low=1027.,range_high=1030.,initial_step=1.,target_step=.25,repo=str(APP_ROOT/'sam3_repo'))
        window.segment_all.setChecked(True);window.analysis_start.click()
    def analysis_done(ok,message):
        report['automatic']=bool(ok and window.analysis_view._pixmap is not None);report['automatic_message']=message
        report['automatic_log']=window.log_box.toPlainText()
        report['automatic_output']=str(window.analysis_output)
        if ok:
            data=json.loads((window.analysis_output/'analysis_summary.json').read_text())
            report['center']=data['center_search']['recommended_center'];report['segmented_slices']=data['segmented_slices']
            report['automatic']=report['automatic'] and data['segmented_slices']==5
            window.grab().save(str(root/'frozen_analysis.png'))
        finish()
    window.runner.finished.connect(manual_done);window.analysis_runner.finished.connect(analysis_done)
    window.input_edit.setText(config['input']);window.output_edit.setText(str(root/'manual'))
    window.gpu_spin.setValue(config.get('gpu',0));window.center_spin.setValue(1028.75);window.start_spin.setValue(1021);window.end_spin.setValue(1026)
    window.method_combo.setCurrentIndex(window.method_combo.findData(ReconMethod.FBP_CUDA))
    QTimer.singleShot(800,window.run_btn.click);QTimer.singleShot(600000,finish)
    return report


def validate_manual_release(app,window,config,root):
    """Exercise the shipped GUI without invoking intelligent analysis."""
    from .models import GeometryMode
    from .hdf5_inspector import inspect_hdf5
    report={'manual':False,'automatic':False,'manual_only':True,'cases':[]}
    steps=config.get('steps',['center','angle','full','stop']);current=[None];stop_sent=[False]
    def finish(error=None):
        if error:report['error']=str(error)
        report['manual']=not error and len(report['cases'])==len(steps) and all(x['passed'] for x in report['cases'])
        report['log']=window.log_box.toPlainText()
        (root/'frozen_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        window.close();QTimer.singleShot(500,app.quit)
    def next_step():
        index=len(report['cases'])
        if index==len(steps):finish();return
        current[0]=steps[index];stop_sent[0]=False
        try:
            if current[0] in ('full','stop'):window.run_btn.click()
            else:window._start_search(current[0])
            if not window.runner.running:finish('Manual job did not start')
        except Exception as error:finish(error)
    def progress(value):
        if current[0]=='stop' and not stop_sent[0] and value.get('stage')=='Saving' and value.get('completed',0)>=2:
            stop_sent[0]=True;window.stop_btn.click()
    def done(ok,message):
        kind=current[0];output=window.active_output_dir;count=len(list(output.glob('*.tiff')))
        row={'kind':kind,'message':message,'files':count,'output':str(output)}
        try:
            if kind=='stop':
                row['passed']=bool(not ok and stop_sent[0] and 0<count<config.get('expected_slices',8) and not (output/'.tofu_work').exists())
            elif kind in ('center','angle'):
                target=config['center'] if kind=='center' else config['angle']
                index=min(range(len(window.center_candidates)),key=lambda i:abs(window.center_candidates[i][0]-target))
                window.center_candidate_slider.setValue(index);window.apply_center_btn.click()
                value=window.center_spin.value() if kind=='center' else window.lamino_spin.value()
                row.update(selected=value,passed=bool(ok and abs(value-target)<1e-6 and window.center_search_view._pixmap is not None))
            else:
                row['passed']=bool(ok and count==config.get('expected_slices',8) and window.result_view._pixmap is not None)
                window.grab().save(str(root/'frozen_manual.png'))
            report['cases'].append(row)
            if not row['passed']:finish('Manual workflow check failed');return
            QTimer.singleShot(200,next_step)
        except Exception as error:finish(error)
    try:
        window.input_edit.setText(config['input']);window.output_edit.setText(str(root/'runs'))
        window.mode_combo.setCurrentIndex(window.mode_combo.findData(GeometryMode.LAMINOGRAPHY))
        window.method_combo.setCurrentIndex(window.method_combo.findData(ReconMethod.LAMINO_TOFU))
        window.lamino_spin.setValue(config['angle']);window._apply_manual_defaults(inspect_hdf5(config['input']))
        window.center_spin.setValue(config['center']);window.gpu_spin.setValue(config.get('gpu',0))
        window.start_spin.setValue(config['start_row']);window.end_spin.setValue(config['end_row'])
        window.try_row_spin.setValue(config['try_row'])
        window.search_width_spin.setValue(config.get('center_radius',1));window.search_step_spin.setValue(config.get('center_step',.25))
        window.lamino_search_width_spin.setValue(config.get('angle_radius',.5));window.lamino_search_step_spin.setValue(config.get('angle_step',.25))
        window.settings['remove_stripe_method']=config.get('remove_stripe_method','none')
        original=window._job
        def job(*args,**kwargs):
            value=original(*args,**kwargs);value.tofu_block_slices=config.get('block_slices',2);return value
        window._job=job
        window.runner.finished.connect(done);window.runner.progress_changed.connect(progress)
        QTimer.singleShot(500,next_step);QTimer.singleShot(600000,lambda:finish('Validation timeout'))
    except Exception as error:QTimer.singleShot(0,lambda error=str(error):finish(error))
    return report
