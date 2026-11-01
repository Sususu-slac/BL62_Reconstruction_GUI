"""Explicit real-data GUI test: text and painted cross-image instance workflows.

Uses existing NMC662_Rod6 reference/selected mask. The search is intentionally
limited to 1013..1021, with real refinement to .25 and only TWO output slices.
This validates the plumbing, not a new global center estimate or ground truth.
"""
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import tifffile
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication,QMessageBox
from ct_recon_gui.main_window import MainWindow
from sam3_rotation_center.exemplar import InstanceExample,read_reference,load_example

SOURCE=Path('D:/1_Datasets/2_Tomography/hdf5_collection/NMC662_Rod6.hdf5')
REFERENCE=Path('D:/BL62_Assistant_Chat/outputs/jobs/20261027_171855_3298bd4f_analyze/results/reconstruction/recon_01026.tiff')
MASK=Path('D:/BL62_Assistant_Chat/test_outputs/cross_image_instance_1793314713932328900/reference_mask.npy')
OUT=ROOT/'test_outputs'/('instance_integration_'+str(time.time_ns()))
OUT.mkdir(parents=True)
print('OUTPUT '+str(OUT),flush=True)
app=QApplication([])
window=MainWindow()
window.show()
window.main_tabs.setCurrentIndex(1)
window.analysis_input.setText(str(SOURCE))
window.settings.update(gpu=0,start_row=1023,end_row=1025,range_low=1013.,range_high=1021.,
    initial_step=2.,target_step=.25,search_row=1023,output_root=str(OUT),
    sam3_python='C:/Users/subo/AppData/Local/miniforge3/envs/SAM3p1/python.exe',
    recon_python='C:/Users/subo/AppData/Local/miniforge3/envs/tomocupy_astra/python.exe',
    repo='D:/SAM3p1/sam3',checkpoint='D:/Chcekpoints/checkpoint_demo_05142026.pt')
window.gpu_spin.setValue(0)
selected=InstanceExample(read_reference(REFERENCE),np.load(MASK).astype(bool),str(REFERENCE))
window._set_instance_example(selected)
window.segment_all.setChecked(True)
source_stats={str(path):(path.stat().st_size,path.stat().st_mtime_ns) for path in (SOURCE,REFERENCE,MASK)}
queue=['text','instance','instance_stop'];results=[];events=[];active={};beat_times=[];last_beat=[time.perf_counter()]
timer=QTimer();timer.setInterval(50)
def beat():
    now=time.perf_counter();beat_times.append(now-last_beat[0]);last_beat[0]=now
timer.timeout.connect(beat);timer.start()
def finish():
    timer.stop()
    report=dict(cases=results,passed=len(results)==3 and all(r['passed'] for r in results),
        heartbeat_max_ms=max(beat_times or [0])*1000,heartbeat_p99_ms=float(np.percentile(beat_times or [0],99))*1000,
        input_files_unchanged=all(stats==(Path(path).stat().st_size,Path(path).stat().st_mtime_ns) for path,stats in source_stats.items()),
        scope='Local 1013..1021 search, step 2 -> .25; 2 reconstructed and segmented slices. Not a global center accuracy study.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    window.close();QTimer.singleShot(100,app.quit)
def fail(message):
    results.append(dict(mode=active.get('mode'),passed=False,error=str(message)))
    print('FAIL '+str(message),flush=True)
    window.analysis_runner.stop()
    if not window.analysis_runner.running:finish()
def next_case():
    if not queue:finish();return
    mode=queue.pop(0);active.clear();active.update(mode=mode,started=time.perf_counter())
    window.analysis_prompt_mode.setCurrentIndex(0 if mode=='text' else 1)
    window.analysis_prompt.setText('particle')
    print('START '+mode,flush=True)
    window.analysis_start.click()
    if mode=='instance_stop':QTimer.singleShot(500,window._stop_analysis)
def updated(status):
    events.append(dict(mode=active.get('mode'),**status))
    if status.get('stage') in ('Loading SAM3','Preparing Instance'):
        print('STATUS '+status['stage'],flush=True)
def done(ok,message):
    if results and not results[-1]['passed']:QTimer.singleShot(100,finish);return
    try:
        mode=active['mode']
        if mode=='instance_stop':
            assert not ok and message=='Stopped',message
            assert not window.analysis_prompt_mode.isEnabled() is False
            assert not (window.analysis_output/'analysis_summary.json').exists()
            results.append(dict(mode=mode,passed=True,seconds=time.perf_counter()-active['started']))
        else:
            assert ok,message
            root=window.analysis_output
            summary=json.loads((root/'analysis_summary.json').read_text())
            assert summary['reconstructed_slices']==2 and summary['segmented_slices']==2
            search=summary['center_search']
            assert search['final_step']==.25 and search['unique_evaluations']>5
            assert 1013<=search['recommended_center']<=1021
            assert search['prompt_mode']==mode
            assert window.analysis_view._pixmap is not None
            assert window.filtered_curve.getData()[0] is not None
            slices=sorted((root/'reconstruction').rglob('recon_*.tiff'))
            for path in slices:
                pixels=tifffile.imread(path)
                assert np.isfinite(pixels).all() and pixels.std()>0
                labels=tifffile.imread(root/'segmentation'/(path.stem+'.tiff'))
                assert labels.shape==pixels.shape and labels.max()>0
            if mode=='instance':
                job=json.loads((root/'analysis_job.json').read_text())
                frozen=load_example(job['instance_reference'])
                np.testing.assert_array_equal(frozen.mask,selected.mask)
                encoding=json.loads((root/'instance_reference/encoding.json').read_text())
                assert encoding['mask_pixels']==int(selected.mask.sum())
                assert encoding['encoding']=='mask_isolated_reference_box_features'
                assert any(e.get('mode')==mode and e.get('stage')=='Preparing Instance' for e in events)
                assert search['instance_reference']['token_shape']==encoding['token_shape']
            window.grab().save(str(OUT/(mode+'_complete.png')))
            results.append(dict(mode=mode,passed=True,center=search['recommended_center'],
                rounds=search['rounds'],points=search['unique_evaluations'],slices=2,output=str(root),
                elapsed_seconds=summary['elapsed_seconds'],seconds=time.perf_counter()-active['started']))
        print('PASS '+mode,flush=True)
        QTimer.singleShot(100,next_case)
    except Exception as error:fail(error)
window.analysis_runner.updated.connect(updated)
window.analysis_runner.finished.connect(done)
QTimer.singleShot(200,next_case)
QTimer.singleShot(900000,lambda:fail('Integration test exceeded 15 minutes'))
with patch.object(QMessageBox,'critical',side_effect=lambda *a:fail(a[-1])),patch.object(QMessageBox,'warning',side_effect=lambda *a:fail(a[-1])):
    app.exec()
report=json.loads((OUT/'report.json').read_text())
raise SystemExit(not (report['passed'] and report['input_files_unchanged']))
