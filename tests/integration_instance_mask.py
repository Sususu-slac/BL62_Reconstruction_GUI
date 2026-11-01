"""Real GPU check: erasing inside an unchanged box changes visual tokens.

No GUI/worker mocks and no source-image writes. It validates mask conditioning,
not arbitrary-feature transfer or the physical accuracy of the chosen center.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                  OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONNOUSERSITE='1')
import gc
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),'D:/SAM3p1/sam3']
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from sam3_rotation_center.exemplar import InstanceExample,read_reference,mask_box,save_example
from sam3_rotation_center.inference import Sam3InstanceSegmenter
from sam3_rotation_center.data import normalize_reconstruction
from sam3_rotation_center.scoring import score_prediction
from sam3_rotation_center.visualization import save_overlay

BASE=Path('D:/BL62_Assistant_Chat/outputs/jobs/20261027_171855_3298bd4f_analyze/results')
OLD=Path('D:/BL62_Assistant_Chat/test_outputs/cross_image_instance_1793314713932328900')
OUT=ROOT/'test_outputs'/('instance_mask_gpu_'+str(time.time_ns()));OUT.mkdir(parents=True)
print('OUTPUT '+str(OUT),flush=True)
reference=BASE/'reconstruction/recon_01026.tiff'
image=read_reference(reference)
mask=np.load(OLD/'reference_mask.npy').astype(bool)
distance=distance_transform_edt(mask)
cy,cx=np.unravel_index(np.argmax(distance),distance.shape)
radius=max(1,int(distance[cy,cx]*.45))
yy,xx=np.mgrid[:mask.shape[0],:mask.shape[1]]
erased=mask.copy();erased[(xx-cx)**2+(yy-cy)**2<=radius**2]=False
assert mask_box(mask)==mask_box(erased) and erased.any()
assert int(erased.sum())<int(mask.sum())
import csv
rows=list(csv.DictReader((BASE/'center_search/scores.csv').open()))
target_path=Path(next(r['file'] for r in rows if float(r['center'])==1017))
target,_=normalize_reconstruction(target_path)
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.30)
records=[];tokens=[]
for name,selected in [('original',mask),('erased',erased)]:
    started=time.perf_counter()
    example=InstanceExample(image,selected,str(reference))
    save_example(example,OUT/name)
    model=Sam3InstanceSegmenter(Path('D:/Chcekpoints/checkpoint_demo_05142026.pt'),example=example,
        confidence_threshold=.2,device='cuda')
    tokens.append(model.reference_tokens.float().cpu().numpy().copy())
    prediction=model.predict(target)
    metrics,keep=score_prediction(prediction,nms_iou=.5)
    save_overlay(target,prediction.masks,keep,OUT/(name+'_target_overlay.png'))
    records.append(dict(name=name,mask_pixels=int(selected.sum()),count=len(keep),
        confidence_reward=metrics['confidence_reward'],seconds=time.perf_counter()-started,
        metadata=model.instance_metadata))
    print(json.dumps(records[-1]),flush=True)
    del prediction,model
    gc.collect();torch.cuda.empty_cache()
delta=float(np.max(np.abs(tokens[0]-tokens[1])))
report=dict(passed=bool(np.isfinite(delta) and delta>1e-5),same_box=True,
    erased_pixels=int(mask.sum()-erased.sum()),token_max_absolute_difference=delta,
    reference=str(reference),target=str(target_path),records=records,
    scope='An interior erasure with unchanged outer box changes real cached SAM3 visual features. Does not establish native mask encoding or generic feature specificity.')
(OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
np.savez(OUT/'tokens.npz',original=tokens[0],erased=tokens[1])
print(json.dumps(report,indent=2),flush=True)
raise SystemExit(not report['passed'])
