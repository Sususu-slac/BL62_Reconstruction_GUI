"""Explicit release validation entry point; never runs on ordinary launch."""
import json
import time
from pathlib import Path
from PySide6.QtCore import QTimer
from .models import ReconMethod
from .runtime import APP_ROOT

def validate_release(app,window,config_path):
    config=json.loads(Path(config_path).read_text());root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
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
