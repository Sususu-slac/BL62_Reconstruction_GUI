"""Native Windows automatic center search, reconstruction and optional SAM3.

The planner, normalization, NMS and confidence reward are the validated method.
Only process orchestration, progress reporting and optional bounds are adapted.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent

class Cancelled(Exception): pass

def atomic_json(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    # Windows readers/antivirus can hold a short-lived delete-sharing lock.
    for attempt in range(100):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.02)

class Pipeline:
    def __init__(self, config):
        self.cfg=config;self.output=Path(config['output']);self.output.mkdir(parents=True,exist_ok=True)
        self.started=time.perf_counter();self.status={};self.model=None;self.last_preview=0;self.last_best=None
        self.search=self.output/'center_search';self.search.mkdir(exist_ok=True)
        self.scratch=self.output/'candidates';self.scratch.mkdir(exist_ok=True)

    def check_cancel(self):
        if (self.output/'CANCEL').exists():raise Cancelled()

    def emit(self, **data):
        self.status.update(data);self.status['elapsed_seconds']=time.perf_counter()-self.started
        atomic_json(self.output/'status.json',self.status)

    def child(self, argv, log_path, poll=None):
        from ct_recon_gui.process_utils import terminate_tree,subprocess_flags
        self.check_cancel()
        env={**os.environ,'PYTHONNOUSERSITE':'1','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'}
        for key in ('PYTHONPATH','PYTHONHOME','CUDA_VISIBLE_DEVICES'):env.pop(key,None)
        with Path(log_path).open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,**subprocess_flags())
            try:
                while proc.poll() is None:
                    self.check_cancel()
                    if poll:poll()
                    time.sleep(.3)
                self.check_cancel()
                if proc.returncode:
                    tail=Path(log_path).read_text(encoding='utf-8',errors='replace')[-4000:]
                    raise RuntimeError(f'Reconstruction exited {proc.returncode}:\n{tail}')
            finally:
                if proc.poll() is None:terminate_tree(proc.pid)

    def load_model(self):
        if self.model is not None:return
        self.emit(stage='Loading SAM3',indeterminate=True)
        print('Loading SAM3 once; it remains resident until this analysis ends.',flush=True)
        from sam3_rotation_center.inference import Sam3ParticleSegmenter
        t=time.perf_counter()
        self.model=Sam3ParticleSegmenter(Path(self.cfg['checkpoint']),prompt=self.cfg['prompt'],
            confidence_threshold=self.cfg['confidence'],device='cuda')
        self.load_seconds=time.perf_counter()-t;self.check_cancel()

    def candidates(self, centers, number):
        from sam3_rotation_center.data import parse_center
        folder=self.scratch/f'round_{number:02d}';folder.mkdir()
        manifest=folder/'centers.json';atomic_json(manifest,centers)
        argv=[self.cfg['recon_python'],str(ROOT/'reconstruct_candidates.py'),
            '--input',self.cfg['input'],'--output',str(folder),'--centers-file',str(manifest),
            '--row',str(self.row),'--gpu',str(self.cfg['gpu']),'--method',self.cfg['search_method']]
        self.emit(stage='Reconstructing Candidates',indeterminate=False,completed=0,total=len(centers))
        self.child(argv,folder/'reconstruction.log',lambda:self.emit(completed=len(list(folder.rglob('*.tiff')))))
        result={parse_center(p):p for p in folder.rglob('*.tiff')}
        if set(result)!=set(centers):raise RuntimeError('Reconstruction did not return exactly the requested centers')
        if any(f'recon_slice{self.row:04d}_' not in p.name for p in result.values()):
            raise RuntimeError('Candidate reconstruction returned the wrong detector row')
        return result

    def decision(self, rows, step):
        from sam3_rotation_center.adaptive import plan
        # With default [0,width-1] this is exactly the original planner.
        shifted=[{**row,'center':row['center']-self.low} for row in rows]
        result=plan(shifted,step,self.high-self.low,self.cfg['target_step'],3,'confidence_reward')
        for peak in result['peaks']:
            for key in ('center','low','high'):peak[key]+=self.low
        result['intervals']=[[lo+self.low,hi+self.low] for lo,hi in result['intervals']]
        result['cumulative_centers']=[c+self.low for c in result['cumulative_centers']]
        result['smoothed_scores']={c+self.low:v for c,v in result['smoothed_scores'].items()}
        result['filter_curve']['start']+=self.low
        result['filter_bounds']=[self.low,self.high]
        return result

    def sample_grid(self, intervals, step):
        from sam3_rotation_center.adaptive import grid
        return [round(c+self.low,6) for c in grid([(lo-self.low,hi-self.low) for lo,hi in intervals],step,self.high-self.low)]

    def preview(self, rows, step, force=False):
        from sam3_rotation_center.scoring import assign_scores
        from sam3_rotation_center.data import normalize_reconstruction
        if not force and time.perf_counter()-self.last_preview < self.cfg['refresh_seconds']:return
        self.last_preview=time.perf_counter()
        try:decision=self.decision(assign_scores(list(rows.values())),step)
        except ValueError:return
        center=decision['peaks'][0]['center'];best=rows[center]
        folder=self.output/'previews';folder.mkdir(exist_ok=True)
        png=folder/f'center_{center:.6f}.png'
        if not png.exists():
            image,_=normalize_reconstruction(Path(best['file']))
            tmp=png.with_suffix('.tmp');image.save(tmp,format='PNG');tmp.replace(png)
        graph={'centers':sorted(rows),'scores':[rows[c]['confidence_reward'] for c in sorted(rows)],
               'curve':decision['filter_curve'],'best_center':center}
        atomic_json(self.search/'plot.json',graph)
        self.emit(best_center=center,best_image=str(png),curve_file=str(self.search/'plot.json'))

    def find_center(self):
        import h5py
        import numpy as np
        from sam3_rotation_center.data import normalize_reconstruction
        from sam3_rotation_center.inference import Prediction
        from sam3_rotation_center.scoring import score_prediction,assign_scores
        from sam3_rotation_center.adaptive import coverage_missing
        from sam3_rotation_center.reporting import round_record,recommendation,plot_history
        from sam3_rotation_center.visualization import save_overlay
        with h5py.File(self.cfg['input'],'r') as h:
            _,self.height,self.width=h['exchange/data'].shape
        self.row=(self.height-1)//2 if self.cfg['search_row'] is None else int(self.cfg['search_row'])
        self.low=0. if self.cfg['range_low'] is None else float(self.cfg['range_low'])
        self.high=float(self.width-1) if self.cfg['range_high'] is None else float(self.cfg['range_high'])
        if not 0<=self.row<self.height:raise ValueError('Invalid center-search slice')
        if not 0<=self.low<self.high<=self.width-1:raise ValueError('Initial search range is outside the detector')
        step=float(self.cfg['initial_step'] or max(1,int(self.width/100+.5)))
        if step<self.cfg['target_step']:raise ValueError('Initial step is smaller than target step')
        intervals=[[self.low,self.high]];cache={};history=[]
        for number in range(1,15):
            self.check_cancel();centers=self.sample_grid(intervals,step);missing=[c for c in centers if c not in cache]
            self.emit(stage='Finding Center',round=number,step=step,slice=self.row,completed=0,total=len(missing),indeterminate=False)
            print(f'ROUND {number} step={step:g} intervals={intervals} new={len(missing)}',flush=True)
            began=time.perf_counter();paths=self.candidates(missing,number) if missing else {};recon_seconds=time.perf_counter()-began
            if missing:self.load_model()
            self.emit(stage='Finding Center',indeterminate=False,completed=0,total=len(missing))
            for index,center in enumerate(missing,1):
                self.check_cancel();t=time.perf_counter()
                try:image,norm=normalize_reconstruction(paths[center])
                except ValueError as error:
                    if not str(error).startswith('Invalid intensity range'):raise
                    image=None;norm={'low':None,'high':None}
                image_seconds=time.perf_counter()-t
                if image is None:
                    prediction=Prediction(np.zeros((0,1,1),bool),np.zeros(0,np.float32));telemetry={**self.model.memory_snapshot(),'inference_seconds':0.}
                else:prediction=self.model.predict(image);telemetry=self.model.last_telemetry
                t=time.perf_counter();metrics,_=score_prediction(prediction,nms_iou=self.cfg['nms_iou'])
                row=dict(center=center,file=str(paths[center]),round=number,step=step,**metrics,**telemetry,
                         metrics_seconds=time.perf_counter()-t,image_read_normalize_seconds=image_seconds,
                         normalization_low=norm['low'],normalization_high=norm['high'])
                cache[center]=row
                with (self.search/'evaluations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                del prediction,image
                self.preview(cache,step)
                self.emit(completed=index,total=len(missing),evaluated=len(cache),current_center=center)
                print(f'  {index}/{len(missing)} center={center:g} score={metrics["confidence_reward"]:.4f}',flush=True)
            decision=self.decision(assign_scores(list(cache.values())),step)
            record=round_record(number,step,intervals,decision,len(centers),len(missing),recon_seconds,time.perf_counter()-began)
            history.append(record);atomic_json(self.search/'history.json',history)
            self.preview(cache,step,True)
            if step<=self.cfg['target_step']+1e-9:
                shifted=[{**p,'center':p['center']-self.low} for p in decision['peaks']]
                support=coverage_missing([c-self.low for c in cache],shifted,step,self.high-self.low)
                if not support:break
                intervals=decision['intervals'];continue
            intervals,step=decision['intervals'],decision['next_step']
        else:raise RuntimeError('Maximum rounds reached before a supported final peak')
        center=decision['peaks'][0]['center'];rows=sorted(assign_scores(list(cache.values())),key=lambda r:r['center'])
        with (self.search/'scores.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        plot_history(history,rows,center,self.width-1,self.search/'adaptive_search.png')
        image,_=normalize_reconstruction(Path(cache[center]['file']));prediction=self.model.predict(image)
        _,kept=score_prediction(prediction,nms_iou=self.cfg['nms_iou'])
        image.save(self.search/'best_input.png');save_overlay(image,prediction.masks,kept,self.search/'best_overlay.png')
        shutil.copy2(cache[center]['file'],self.search/'best_reconstruction.tiff')
        summary=dict(recommended_center=center,final_step=step,rounds=len(history),unique_evaluations=len(cache),
                     score_method='confidence_reward',model_load_seconds=self.load_seconds,
                     recommended_metrics=cache[center],final_peaks=decision['peaks'],
                     **recommendation(decision['peaks'],decision['score_scale']))
        atomic_json(self.search/'summary.json',summary)
        self.emit(best_center=center,best_image=str(self.search/'best_input.png'))
        return summary

    def reconstruct(self, center):
        factor=2**self.cfg['binning'];start=self.cfg['start_row'];end=self.cfg['end_row']
        start=0 if start is None else int(start);end=self.height if end is None else int(end)
        if not 0<=start<end<=self.height or (end-start)//factor<1:raise ValueError('Invalid reconstruction row range')
        self.reconstruction=self.output/'reconstruction';self.reconstruction.mkdir(exist_ok=True)
        total=(end-start)//factor
        metadata=self.reconstruction/'run_metadata.json'
        existing=sorted(self.reconstruction.rglob('recon_*.tiff'))
        if metadata.exists() and len(existing)==total:
            saved=json.loads(metadata.read_text())
            if saved.get('original_center')==center and saved.get('original_rows')==[start,end] and saved.get('method')==self.cfg['method']:
                print('Reusing the completed reconstruction from this interrupted job.',flush=True)
                return existing
        argv=[self.cfg['recon_python'],str(ROOT/'backend_runner.py'),self.cfg['method'],self.cfg['input'],str(self.reconstruction),
            '--reconstruction-type','full','--start-row',str(start),'--end-row',str(end),'--center',str(center),
            '--gpu',str(self.cfg['gpu']),'--chunk-size',str(self.cfg['chunk_size']),'--iterations',str(self.cfg['iterations']),
            '--binning',str(self.cfg['binning'])]
        if self.cfg['method']=='lprec':argv.append('--accept-uniform-angle-approximation')
        self.emit(stage='Reconstructing',completed=0,total=total,indeterminate=False)
        self.child(argv,self.reconstruction/'reconstruction.log',lambda:self.emit(completed=len(list(self.reconstruction.rglob('recon_*.tiff')))))
        files=sorted(self.reconstruction.rglob('recon_*.tiff'))
        if len(files)!=total:raise RuntimeError(f'Expected {total} reconstructed slices, received {len(files)}')
        self.emit(completed=total)
        return files

    def segment(self, files):
        import numpy as np
        import tifffile
        from sam3_rotation_center.data import normalize_reconstruction
        from sam3_rotation_center.scoring import score_prediction
        output=self.output/'segmentation';output.mkdir(exist_ok=True)
        self.emit(stage='Segmenting',completed=0,total=len(files),indeterminate=False)
        with (output/'instances.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['slice','instance_id','confidence','area_pixels'])
            for index,path in enumerate(files,1):
                self.check_cancel();image,_=normalize_reconstruction(path)
                prediction=self.model.predict(image);_,keep=score_prediction(prediction,nms_iou=self.cfg['nms_iou'])
                labels=np.zeros((image.height,image.width),np.uint16 if len(keep)<65535 else np.uint32)
                for instance,mask_index in enumerate(keep,1):
                    mask=prediction.masks[mask_index];labels[mask&(labels==0)]=instance
                    writer.writerow([path.stem,instance,float(prediction.scores[mask_index]),int(mask.sum())])
                tmp=output/(path.stem+'.tmp');tifffile.imwrite(tmp,labels);tmp.replace(output/(path.stem+'.tiff'))
                # Preserve every instance mask, including overlapping pixels.
                packed=np.packbits(prediction.masks[keep].reshape(len(keep),image.height*image.width),axis=1) if keep else np.zeros((0,(image.height*image.width+7)//8),np.uint8)
                np.savez_compressed(output/(path.stem+'_instances.npz'),masks_packed=packed,scores=prediction.scores[keep],shape=np.array(labels.shape))
                f.flush();self.emit(completed=index)
                print(f'SEGMENT {index}/{len(files)} {path.name} instances={len(keep)}',flush=True)
        return len(files)

    def run(self):
        self.check_cancel();self.emit(stage='Preparing',indeterminate=True)
        from ct_recon_gui.hdf5_inspector import inspect_hdf5
        info=inspect_hdf5(self.cfg['input']);_,self.height,self.width=info.data_shape
        for key in ('recon_python','checkpoint'):
            if not Path(self.cfg[key]).is_file():raise FileNotFoundError(f'{key}: {self.cfg[key]}')
        if not self.cfg['prompt'].strip():raise ValueError('Enter an analysis target')
        fingerprint={**self.cfg}
        for key in ('input','checkpoint'):
            stat=Path(self.cfg[key]).stat();fingerprint[key+'_fingerprint']=[stat.st_size,stat.st_mtime_ns]
        saved=self.output/'pipeline_config.json'
        if saved.exists() and json.loads(saved.read_text())!=fingerprint:raise ValueError('Cannot resume: input, checkpoint or settings changed')
        if (self.output/'analysis_summary.json').exists():raise FileExistsError('This analysis is already complete')
        atomic_json(saved,fingerprint)
        existing=self.search/'summary.json'
        if existing.exists():
            summary=json.loads(existing.read_text());self.emit(best_center=summary['recommended_center'],best_image=str(self.search/'best_input.png'))
        else:summary=self.find_center()
        self.check_cancel()
        files=self.reconstruct(summary['recommended_center'])
        if self.cfg['segment_all']:self.load_model()
        segmented=self.segment(files) if self.cfg['segment_all'] else 0
        self.check_cancel()
        result=dict(status='completed',center_search=summary,reconstructed_slices=len(files),segmented_slices=segmented,
                    elapsed_seconds=time.perf_counter()-self.started,configuration=self.cfg)
        atomic_json(self.output/'analysis_summary.json',result)
        self.emit(stage='Completed',completed=1,total=1,indeterminate=False)
        print('COMPLETE '+json.dumps({'center':summary['recommended_center'],'reconstructed':len(files),'segmented':segmented}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);a=parser.parse_args()
    config=json.loads(a.config.read_text(encoding='utf-8'))
    os.environ['CUDA_VISIBLE_DEVICES']=str(config['gpu']);os.environ['PYTHONNOUSERSITE']='1'
    sys.path.insert(0,config['repo'])
    pipeline=Pipeline(config)
    try:pipeline.run();return 0
    except Cancelled:pipeline.emit(stage='Stopped',indeterminate=False);print('Stopped safely.',flush=True);return 2
    except Exception as error:
        traceback.print_exc();pipeline.emit(stage='Failed',error=str(error),indeterminate=False);return 1

if __name__=='__main__':raise SystemExit(main())
