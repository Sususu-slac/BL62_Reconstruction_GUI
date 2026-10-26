"""Resumable native-Windows reconstruction + persistent-WSL SAM3 search."""
import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
from .adaptive import grid, plan, coverage_missing
from .data import normalize_reconstruction, parse_center
from .inference import Sam3ParticleSegmenter, Prediction
from .scoring import score_prediction, assign_scores
from .visualization import save_overlay
from .reporting import round_record, recommendation, plot_history

WINDOWS_PYTHON = '/mnt/c/Users/subo/AppData/Local/miniforge3/envs/tomocupy_astra/python.exe'
HELPER = '/mnt/d/BL62_Auto_Rotation_Center_Method/reconstruct_candidates.py'

def windows_path(path):
    value = str(Path(path).resolve())
    match = re.match(r'/mnt/([a-z])/(.*)', value)
    if not match: raise ValueError(f'Native reconstruction needs a Windows drive path: {value}')
    return match[1].upper()+':\\'+match[2].replace('/', '\\')

def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)

def fingerprint(path):
    p = Path(path).resolve(); st=p.stat()
    return {'path':str(p), 'size':st.st_size, 'mtime_ns':st.st_mtime_ns}

def reconstruct(a, centers, number):
    root = a.scratch / f'round_{number:02d}'
    root.mkdir(parents=True, exist_ok=True)
    files = {parse_center(p):p for p in root.rglob('*.tiff')}
    missing = [c for c in centers if c not in files]
    if missing:
        manifest = root/'requested_centers.json'
        write_json(manifest, missing)
        dest = root / f'batch_{len(files):04d}'
        dest.mkdir(exist_ok=True)
        cmd = [WINDOWS_PYTHON, windows_path(a.recon_helper), '--input', windows_path(a.input),
               '--output', windows_path(dest), '--centers-file', windows_path(manifest),
               '--row', str(a.row), '--gpu', str(a.gpu), '--method', a.method]
        with (root/'reconstruction.log').open('a') as log:
            child_env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8', 'PYTHONNOUSERSITE':'1'}
            child_env.pop('PYTHONPATH',None)
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True,env=child_env)
        files.update({parse_center(p):p for p in dest.rglob('*.tiff')})
    if any(c not in files for c in centers):raise RuntimeError('Missing reconstructed candidates')
    if any(f'recon_slice{a.row:04d}_' not in files[c].name for c in centers):
        raise RuntimeError('Reconstruction returned the wrong detector row')
    return {c:files[c] for c in centers}

def run(a, segmenter=None):
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=True)
    a.scratch.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.input, 'r') as h:
        _, height, width = h['exchange/data'].shape
    if a.row is None:a.row=(height-1)//2
    if not 0<=a.row<height:raise ValueError('Invalid detector row')
    upper = float(width-1)
    initial_step = float(max(1, int(width/100+0.5)))
    configuration = {'input':fingerprint(a.input), 'checkpoint':fingerprint(a.checkpoint),
                     'row':a.row, 'width':width, 'method':a.method, 'gpu':a.gpu,
                     'prompt':a.prompt, 'confidence':0.2, 'nms':0.5,
                     'thresholds':[0.2,0.3,0.4,0.5,0.6], 'target_step':a.target_step,
                     'initial_step':initial_step, 'bounds':[0,upper], 'binning':1,
                     'algorithm_version':'adaptive_v3_cumulative_global_bounded_refinement', 'max_peaks':a.max_peaks,
                     'score_method':a.score,
                     'filter_input':'all cumulative raw scores', 'filter_axis':'global original-pixel coordinates',
                     'filter_grid_spacing':a.target_step, 'gaussian_sigma_min_pixels':0.75,
                     'refinement_radius_steps':6,'minimum_refinement_radius_pixels':3.0}
    conf_path = a.output/'run_config.json'
    if conf_path.exists() and json.loads(conf_path.read_text()) != configuration:
        raise ValueError('Resume parameters differ from the saved run')
    if (a.output/'summary.json').exists():
        raise FileExistsError('Completed result already exists; choose a new output directory')
    write_json(conf_path, configuration)
    cache = {}
    journal = a.output/'evaluations.jsonl'
    if journal.exists():
        lines = journal.read_text().splitlines()
        for i, line in enumerate(lines):
            try:r=json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines)-1:raise
                journal.write_text('\n'.join(lines[:i])+'\n'); break
            cache[r['center']] = r
    load_seconds=0.0
    def ensure_segmenter():
        nonlocal segmenter,load_seconds
        if segmenter is None:
            os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
            print('Loading SAM3 once for remaining candidates...',flush=True)
            t=time.perf_counter()
            segmenter=Sam3ParticleSegmenter(a.checkpoint,prompt=a.prompt,device='cuda')
            load_seconds=time.perf_counter()-t
        write_json(a.output/'loading.json', {'load_seconds':load_seconds, **segmenter.memory_snapshot()})
    intervals, step = [[0.0,upper]], initial_step
    history=[]
    visible_centers=set()
    for number in range(1, a.max_rounds+1):
        centers=grid(intervals,step,upper)
        missing=[c for c in centers if c not in cache]
        t=time.perf_counter()
        print(f'ROUND {number} step={step:g} intervals={intervals} new={len(missing)} cached={len(centers)-len(missing)}',flush=True)
        recon_started=time.perf_counter()
        paths=reconstruct(a,missing,number) if missing else {}
        recon_seconds=time.perf_counter()-recon_started
        if missing:ensure_segmenter()
        for index,center in enumerate(missing,1):
            status='evaluated'
            image_started=time.perf_counter()
            try:
                image, norm=normalize_reconstruction(paths[center])
            except ValueError as error:
                if not str(error).startswith('Invalid intensity range'):raise
                # A constant image at an extreme candidate has no evidence.
                image=None; norm={'low':None,'high':None};status='flat_image'
            image_seconds=time.perf_counter()-image_started
            if image is None:
                prediction=Prediction(np.zeros((0,1,1),dtype=bool),np.zeros(0,dtype=np.float32))
                telemetry={**segmenter.memory_snapshot(),'inference_seconds':0.0}
            else:
                prediction=segmenter.predict(image)
                telemetry=segmenter.last_telemetry
            score_started=time.perf_counter()
            metrics,_=score_prediction(prediction)
            metrics_seconds=time.perf_counter()-score_started
            row={'center':center,'file':str(paths[center]),'round':number,'step':step,
                 **metrics, **telemetry, 'metrics_seconds':metrics_seconds, 'status':status,
                 'image_read_normalize_seconds':image_seconds,
                 'normalization_low':norm['low'],'normalization_high':norm['high']}
            with journal.open('a',encoding='utf-8') as f:
                f.write(json.dumps(row)+'\n'); f.flush()
            cache[center]=row
            del prediction, image
            print(f'  {index}/{len(missing)} center={center:g} count={metrics["nms_count"]} '
                  f'infer={row["inference_seconds"]:.2f}s score={metrics_seconds:.2f}s '
                  f'allocated={row["cuda_allocated_gib"]:.2f} reserved={row["cuda_reserved_gib"]:.2f}GiB',flush=True)
        # Replay exactly the same planning decisions after interruption: future
        # cached measurements must not influence an earlier round's denominator.
        visible_centers.update(centers)
        planning_started=time.perf_counter()
        cumulative=assign_scores([cache[c] for c in visible_centers])
        decision=plan(cumulative,step,upper,a.target_step,a.max_peaks,a.score)
        record=round_record(number,step,intervals,decision,len(centers),len(missing),
                            recon_seconds,time.perf_counter()-t)
        record['planning_seconds']=time.perf_counter()-planning_started
        history.append(record)
        write_json(a.output/'history.json',history)
        print('PEAKS '+json.dumps(decision['peaks']),flush=True)
        if step<=a.target_step+1e-9:
            missing_support=coverage_missing(visible_centers,decision['peaks'],step,upper)
            if missing_support:
                print(f'Globally selected peaks need {len(missing_support)} additional fine samples',flush=True)
                intervals=decision['intervals']
                continue
            break
        intervals,step=decision['intervals'],decision['next_step']
    else:raise RuntimeError('Maximum rounds reached before a bounded final peak; inspect history.json')
    winner=decision['peaks'][0]
    finalization_started=time.perf_counter()
    center=winner['center']
    rows=sorted(assign_scores(list(cache.values())),key=lambda r:r['center'])
    with (a.output/'scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])+['multithreshold_score'] if 'multithreshold_score' not in rows[0] else list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    plot_history(history,rows,center,upper,a.output/'adaptive_search.png')
    raw_best=max(rows,key=lambda r:r['multithreshold_score'])
    bestfile=Path(cache[center]['file'])
    if not bestfile.exists():bestfile=reconstruct(a,[center],99)[center]
    ensure_segmenter()
    image,_=normalize_reconstruction(bestfile)
    prediction=segmenter.predict(image)
    _,kept=score_prediction(prediction)
    image.save(a.output/'best_input.png')
    save_overlay(image,prediction.masks,kept,a.output/'best_overlay.png')
    summary={'config':configuration,'recommended_center':center,'final_step':step,'rounds':len(history),
             'score_method':a.score,
             'unique_evaluations':len(cache),'wall_seconds':time.perf_counter()-started,
             'model_load_seconds':load_seconds,'final_peaks':decision['peaks'],
             'raw_multithreshold_best':raw_best,'recommended_metrics':cache[center],
             'raw_selected_score_best':max(rows,key=lambda r:r[a.score]),
             'particle_count_best':max(rows,key=lambda r:r['nms_count']),
             'confidence_reward_best':max(rows,key=lambda r:r['confidence_reward']),
             'peak_allocated_gib':max(r['cuda_peak_allocated_gib'] for r in rows),
             'max_reserved_gib_after_image':max(r['cuda_reserved_gib'] for r in rows),
             'allocated_gib_range':[min(r['cuda_allocated_gib'] for r in rows),max(r['cuda_allocated_gib'] for r in rows)],
             'mean_inference_seconds':float(np.mean([r['inference_seconds'] for r in rows])),
             'mean_metrics_seconds':float(np.mean([r['metrics_seconds'] for r in rows])),
             'sampling_note':'0.25 px is the sampling interval, not a calibrated physical accuracy',
             **recommendation(decision['peaks'],decision['score_scale'])}
    timings={'model_load_seconds':load_seconds,
             'reconstruction_seconds':sum(h['reconstruction_seconds'] for h in history),
             'image_read_normalize_seconds':sum(r['image_read_normalize_seconds'] for r in rows),
             'sam3_inference_seconds':sum(r['inference_seconds'] for r in rows),
             'score_statistics_seconds':sum(r['metrics_seconds'] for r in rows),
             'global_filter_planning_seconds':sum(h['planning_seconds'] for h in history),
             'result_finalization_seconds':time.perf_counter()-finalization_started}
    timings['other_seconds']=max(0.,summary['wall_seconds']-sum(timings.values()))
    timings['engine_total_seconds']=summary['wall_seconds']
    summary['timings']=timings
    write_json(a.output/'summary.json',summary)
    print('COMPLETE '+json.dumps({k:summary[k] for k in ['recommended_center','final_step','unique_evaluations','wall_seconds','peak_allocated_gib','max_reserved_gib_after_image','ambiguity']}),flush=True)
    return summary

def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('/mnt/d/Chcekpoints/checkpoint_demo_05142026.pt'))
    p.add_argument('--recon-helper',type=Path,default=Path(HELPER))
    p.add_argument('--row',type=int)
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--method',choices=['fourierrec','linerec'],default='fourierrec')
    p.add_argument('--prompt',default='particle')
    p.add_argument('--score',choices=['multithreshold_score','confidence_reward'],default='multithreshold_score')
    p.add_argument('--target-step',type=float,default=0.25,choices=[0.25,0.5,1.0])
    p.add_argument('--max-peaks',type=int,default=3)
    p.add_argument('--max-rounds',type=int,default=14)
    return p.parse_args()

def main():
    a=arguments()
    run(a)

if __name__=='__main__':main()
