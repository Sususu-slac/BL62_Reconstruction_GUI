"""Real-data GUI integration test; run explicitly in tomocupy_astra.

Exercises six manual methods, direct center/angle buttons, laminography,
safe stop, optional automatic segmentation and UI heartbeat responsiveness.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import tifffile
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication,QMessageBox
from ct_recon_gui.main_window import MainWindow
from ct_recon_gui.models import GeometryMode,ReconMethod

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--manual-only',action='store_true');parser.add_argument('--auto-only',action='store_true');parser.add_argument('--global-search',action='store_true');parser.add_argument('--segment-all',action='store_true');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);app=QApplication([]);window=MainWindow();window.show()
    source_before=(args.input.stat().st_size,args.input.stat().st_mtime_ns)
    results=[];heartbeat=[];last=[time.perf_counter()];active={};queue=[]
    clock=QTimer();clock.setInterval(50)
    def beat():
        now=time.perf_counter();heartbeat.append(now-last[0]);last[0]=now
    clock.timeout.connect(beat);clock.start()
    window.input_edit.setText(str(args.input));window.output_edit.setText(str(args.output/args.input.stem));window.center_spin.setValue(1028.75)
    window.start_spin.setValue(1021);window.end_spin.setValue(1026);window.try_row_spin.setValue(1023)
    window.gpu_spin.setValue(args.gpu)
    for method in ('fourierrec','lprec','linerec','fbp_cuda','sirt_cuda','cgls_cuda'):queue.append(('full',method,'tomography'))
    queue += [('center','fourierrec','tomography'),('full','lamino_fourierrec','laminography'),('full','lamino_linerec','laminography'),('angle','lamino_linerec','laminography'),('stop','sirt_cuda','tomography')]
    if args.auto_only:queue=[]
    if not args.manual_only:queue += [('automatic','fourierrec','tomography'),('auto_stop','fourierrec','tomography')]
    def fail(message):
        results.append({'case':active.get('name','unknown'),'passed':False,'error':str(message)})
        window.runner.stop();window.analysis_runner.stop();QTimer.singleShot(1000,finish)
    def dialog_error(*items):fail(items[-1]);return QMessageBox.StandardButton.Ok
    def finish():
        report={'cases':results,'all_passed':bool(results) and all(r['passed'] for r in results),
                'heartbeat_max_ms':round(max(heartbeat or [0])*1000,1),'heartbeat_p99_ms':round(float(np.percentile(heartbeat or [0],99))*1000,1),
                'input_unchanged':source_before==(args.input.stat().st_size,args.input.stat().st_mtime_ns)}
        (args.output/'integration_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
        window.close();QTimer.singleShot(1000,app.quit)
    def next_case():
        if not queue:finish();return
        kind,method,mode=queue.pop(0);active.clear();active.update(kind=kind,name=f'{mode}_{kind}_{method}',started=time.perf_counter())
        print('CASE '+active['name'],flush=True)
        window.mode_combo.setCurrentIndex(window.mode_combo.findData(GeometryMode(mode)))
        window.method_combo.setCurrentIndex(window.method_combo.findData(ReconMethod(method)))
        window.gpu_spin.setValue(args.gpu);window.iter_spin.setValue(5);window.binning_combo.setCurrentIndex(0)
        window.start_spin.setValue(1021);window.end_spin.setValue(1026)
        if mode=='laminography':
            window.binning_combo.setCurrentIndex(3);window.start_spin.setValue(1024);window.end_spin.setValue(1032)
        window.search_width_spin.setValue(.5);window.search_step_spin.setValue(.5)
        window.lamino_search_width_spin.setValue(.25);window.lamino_search_step_spin.setValue(.25)
        if kind in ('automatic','auto_stop'):
            window.main_tabs.setCurrentIndex(1);window.analysis_input.setText(str(args.input))
            window.settings.update(output_root=str(args.output/'automatic'),gpu=args.gpu,start_row=1021,end_row=1026,
                range_low=1024.,range_high=1032.,initial_step=2.,target_step=.25)
            if args.global_search:window.settings.update(range_low=None,range_high=None,initial_step=None)
            window.segment_all.setChecked(args.segment_all);window.analysis_start.click()
            if kind=='auto_stop':QTimer.singleShot(1500,window._stop_analysis)
        elif kind=='center':window.find_center_btn.click()
        elif kind=='angle':window.find_angle_btn.click()
        else:
            if kind=='stop':window.iter_spin.setValue(10000)
            window.run_btn.click()
            if kind=='stop':QTimer.singleShot(1500,window.runner.stop)
    def manual_done(ok,message):
        try:
            kind=active['kind'];root=window.active_output_dir
            if kind=='stop':
                assert not ok and window.state_label.text()=='Stopped',message
                results.append({'case':active['name'],'passed':True,'seconds':time.perf_counter()-active['started']})
            else:
                assert ok,message
                images=sorted(root.rglob('*.tiff'));assert images,'No output TIFF'
                for path in images:
                    array=tifffile.imread(path);assert array.ndim==2 and np.isfinite(array).all() and np.std(array)>0,path
                if kind in ('center','angle'):
                    assert window.center_candidates,'Candidates not visible'
                    window._apply_selected_center();assert (root/('selected_lamino_angle.json' if kind=='angle' else 'selected_rotation_center.json')).is_file()
                else:assert window.results,'Results not listed'
                results.append({'case':active['name'],'passed':True,'slices':len(images),'shape':list(array.shape),'seconds':time.perf_counter()-active['started']})
                window.grab().save(str(args.output/(active['name']+'.png')))
            QTimer.singleShot(200,next_case)
        except Exception as error:fail(error)
    def auto_done(ok,message):
        try:
            if active['kind']=='auto_stop':assert not ok and message=='Stopped',message
            else:
                assert ok,message
                summary=json.loads((window.analysis_output/'analysis_summary.json').read_text())
                assert summary['reconstructed_slices']==5 and summary['segmented_slices']==(5 if args.segment_all else 0)
                assert abs(summary['center_search']['recommended_center']-1028.75)<=.5,summary['center_search']['recommended_center']
                assert window.analysis_view._pixmap is not None,'Best image not displayed'
                assert '102' in window.analysis_center.text(),'Center not displayed'
                assert window.filtered_curve.getData()[0] is not None,'Curve not displayed'
                window.grab().save(str(args.output/'automatic_complete.png'))
            results.append({'case':active['name'],'passed':True,'seconds':time.perf_counter()-active['started']})
            QTimer.singleShot(200,next_case)
        except Exception as error:fail(error)
    window.runner.finished.connect(manual_done);window.analysis_runner.finished.connect(auto_done)
    QTimer.singleShot(700,next_case);QTimer.singleShot(1800000,lambda:fail('Integration test timeout'))
    with patch.object(QMessageBox,'critical',side_effect=dialog_error),patch.object(QMessageBox,'warning',side_effect=dialog_error):app.exec()
    report=json.loads((args.output/'integration_report.json').read_text())
    return 0 if report['all_passed'] and report['input_unchanged'] else 1

if __name__=='__main__':raise SystemExit(main())
