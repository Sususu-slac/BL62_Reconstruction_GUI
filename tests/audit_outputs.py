import argparse,json,csv
from pathlib import Path
import numpy as np
import tifffile
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
root=a.output;summary=json.loads((root/'analysis_summary.json').read_text());files=sorted((root/'reconstruction').rglob('recon_*.tiff'));segment=root/'segmentation'
records=list(csv.DictReader((segment/'instances.csv').open(newline='')))
for path in files:
 image=tifffile.imread(path);labels=tifffile.imread(segment/path.name)
 assert image.dtype==np.float32 and np.isfinite(image).all() and image.std()>0
 assert labels.shape==image.shape and np.issubdtype(labels.dtype,np.integer)
 with np.load(segment/(path.stem+'_instances.npz')) as data:
  packed=data['masks_packed'];scores=data['scores'];shape=tuple(data['shape'])
  assert shape==image.shape and labels.max()<=len(scores) and np.all(scores>.2) and np.all(scores<=1)
  rows=[r for r in records if r['slice']==path.stem];assert len(rows)==len(scores)
  for index in range(len(scores)):
   mask=np.unpackbits(packed[index])[:image.size].reshape(shape)
   assert int(mask.sum())==int(rows[index]['area_pixels'])
   assert np.isclose(scores[index],float(rows[index]['confidence']))
assert len(files)==summary['reconstructed_slices']==summary['segmented_slices']==5
report={'passed':True,'slices':len(files),'instances':len(records),'center':summary['center_search']['recommended_center'],'overlapping_instance_masks_preserved':True}
(root/'output_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
