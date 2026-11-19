"""Atomic, cancellable image/HDF5 conversions. Source files are read-only."""
from pathlib import Path
import json, os, re, shutil, uuid
import h5py
import numpy as np
import tifffile
from PIL import Image

class Cancelled(Exception):pass

def check(cancel):
    if cancel():raise Cancelled('Cancelled; incomplete output removed')

def image_files(folder):
    def natural(p):return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)',p.name)]
    return sorted((p for p in Path(folder).iterdir() if p.suffix.lower() in ('.tif','.tiff','.png')),key=natural)

def read_image(path):
    a=tifffile.imread(path) if Path(path).suffix.lower() in ('.tif','.tiff') else np.asarray(Image.open(path))
    if a.ndim!=2 or a.dtype.kind not in 'uif':raise ValueError('Use single-channel 2D numerical images')
    return a

def images_to_h5(files,destination,kind='projections',state='attenuation',theta=None,units='degrees',flat=None,dark=None,cancel=lambda:False,progress=lambda *a:None):
    from .input_state import STATES
    files=[Path(p) for p in files];destination=Path(destination)
    if not files:raise ValueError('No images selected')
    if destination.exists():raise ValueError('Output already exists; choose a new name')
    if kind not in ('projections','reconstruction'):raise ValueError('Invalid image kind')
    if state not in STATES:raise ValueError('Choose data preparation')
    if kind=='projections':
        theta=np.asarray(theta,dtype=np.float64)
        if theta.ndim!=1 or len(theta)!=len(files) or not np.isfinite(theta).all():raise ValueError('Provide one finite theta per image')
        if units not in ('degrees','radians'):raise ValueError('Choose theta units')
        if state=='raw' and (not flat or not dark):raise ValueError('Raw projections require Flat and Dark image folders')
    first=read_image(files[0]);destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_name(destination.name+'.partial-'+uuid.uuid4().hex)
    try:
        with h5py.File(temp,'w') as h:
            key='exchange/data' if kind=='projections' else 'reconstruction/data'
            d=h.create_dataset(key,shape=(len(files),*first.shape),dtype=first.dtype,chunks=(1,*first.shape))
            h.attrs['image_kind']=kind;h.attrs['source_folder']=str(files[0].parent)
            if kind=='projections':
                d.attrs['data_preparation']=state;h.attrs['data_preparation']=state
                h.create_dataset('exchange/theta',data=theta).attrs['units']=units
            for k,path in enumerate(files):
                check(cancel);a=first if k==0 else read_image(path)
                if a.shape!=first.shape or a.dtype!=first.dtype:raise ValueError('All images must have the same shape and dtype: '+str(path))
                d[k]=a;progress(k+1,len(files),'Writing H5')
            if kind=='projections' and state=='raw':
                for name,folder in [('data_white',flat),('data_dark',dark)]:
                    fields=image_files(folder)
                    if not fields:raise ValueError('Calibration folder is empty')
                    a=read_image(fields[0]);ds=h.create_dataset('exchange/'+name,shape=(len(fields),*first.shape),dtype=a.dtype,chunks=(1,*first.shape))
                    for k,path in enumerate(fields):
                        check(cancel);a=read_image(path)
                        if a.shape!=first.shape or a.dtype!=ds.dtype:raise ValueError('Calibration shape/dtype mismatch')
                        ds[k]=a;progress(k+1,len(fields),'Writing '+name)
        check(cancel)
        # No replacement: fail if another process created the destination.
        os.link(temp,destination);temp.unlink()
    finally:
        if temp.exists():temp.unlink()
    return destination

def datasets(path):
    result=[]
    with h5py.File(path,'r') as h:
        def visit(name,obj):
            if isinstance(obj,h5py.Dataset) and obj.ndim in (2,3) and obj.dtype.kind in 'uif':result.append((name,tuple(obj.shape),str(obj.dtype)))
        h.visititems(visit)
    return result

def h5_to_images(source,key,destination,start=0,end=None,stride=1,format='tiff',bits=16,cancel=lambda:False,progress=lambda *a:None):
    destination=Path(destination)
    if destination.exists():raise ValueError('Choose a new output folder; existing folders are not overwritten')
    if format not in ('tiff','png') or bits not in (8,16):raise ValueError('Invalid output format')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_name(destination.name+'.partial-'+uuid.uuid4().hex);temp.mkdir()
    try:
        with h5py.File(source,'r') as h:
            d=h[key];count=d.shape[0] if d.ndim==3 else 1;end=count if end is None else end
            if d.ndim not in (2,3) or not 0<=start<end<=count or stride<1:raise ValueError('Invalid dataset or range [start,end)')
            indices=list(range(start,end,stride));lo=hi=None
            if format=='png':
                lo,hi=np.inf,-np.inf
                for k,i in enumerate(indices):
                    check(cancel);a=d[i] if d.ndim==3 else d[:];a=a[np.isfinite(a)]
                    if a.size:lo=min(lo,float(a.min()));hi=max(hi,float(a.max()))
                    progress(k+1,len(indices)*2,'Measuring shared PNG range')
                if not np.isfinite(lo):raise ValueError('No finite pixels')
            for k,i in enumerate(indices):
                check(cancel);a=d[i] if d.ndim==3 else d[:]
                if format=='tiff':tifffile.imwrite(temp/f'image_{i:06d}.tiff',a,photometric='minisblack')
                else:
                    a=np.clip(np.nan_to_num((a.astype(np.float64)-lo)/max(hi-lo,1e-30),nan=0,posinf=1,neginf=0),0,1)
                    a=np.rint(a*(2**bits-1)).astype(np.uint8 if bits==8 else np.uint16);Image.fromarray(a).save(temp/f'image_{i:06d}.png')
                progress(k+1+(len(indices) if format=='png' else 0),len(indices)*(2 if format=='png' else 1),'Writing images')
            if key.strip('/')=='exchange/data' and 'exchange/theta' in h:
                theta=h['exchange/theta'];
                if len(theta)!=count:raise ValueError('Theta count mismatch')
                units=theta.attrs.get('units','unknown');units=units.decode() if isinstance(units,bytes) else str(units)
                np.savetxt(temp/'theta.csv',theta[:][indices],delimiter=',',header='theta_'+units,comments='# ')
            (temp/'conversion.json').write_text(json.dumps(dict(source=str(source),dataset=key,indices=indices,format=format,png_bits=bits if format=='png' else None,png_range=[lo,hi] if format=='png' else None),indent=2))
        check(cancel);temp.rename(destination)
    finally:
        if temp.exists():shutil.rmtree(temp)
    return destination
