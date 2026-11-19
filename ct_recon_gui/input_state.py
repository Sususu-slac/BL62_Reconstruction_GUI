"""Explicit projection preparation state; never infer it from gray values."""
from pathlib import Path
import h5py
import numpy as np

STATES=('raw','normalized','attenuation')
LABELS={'raw':'Raw projections','normalized':'Flat/Dark corrected','attenuation':'−Log applied'}
# Populated from the recorded local GPU validation matrix before release.
PROCESSED_METHODS={'fourierrec','linerec','lprec','fbp_cuda','sirt_cuda','cgls_cuda','lamino_linerec','lamino_fourierrec','lamino_tofu'}

def stored_state(handle):
    value=handle['exchange/data'].attrs.get('data_preparation',handle.attrs.get('data_preparation',''))
    if isinstance(value,bytes):value=value.decode()
    return str(value) if str(value) in STATES else None

def resolve(path,requested='auto'):
    if requested not in (*STATES,'auto'):raise ValueError('Unknown data preparation state')
    with h5py.File(path,'r') as h:
        recorded=stored_state(h)
        state=requested if requested!='auto' else recorded
        if state is None:
            if all(x in h for x in ('exchange/data_white','exchange/data_dark')):state='raw'
            else:raise ValueError('Flat/Dark are missing and preparation is unknown. Choose Flat/Dark corrected or −Log applied explicitly.')
        if state=='raw' and not all(x in h for x in ('exchange/data_white','exchange/data_dark')):raise ValueError('Raw projections require both Flat and Dark fields')
        return state

def supported(method,state):
    return state=='raw' or str(getattr(method,'value',method)) in PROCESSED_METHODS

def validate_options(options):
    state=options.get('data_state','raw')
    if state not in STATES:raise ValueError('Select a data preparation state')
    if not supported(options.get('method','fourierrec'),state):raise ValueError('This algorithm has not passed validation for processed input')
    if state!='raw':
        stripe=options.get('remove_stripe_method',options.get('stripe_method','none'))
        if stripe!='none' or options.get('dezinger',0) or options.get('flat_linear',False) or options.get('bright_ratio',1.)!=1.:
            raise ValueError('Processed input currently supports no additional stripe/dezinger/flat correction. Select None/Off; use bright ratio 1.')

def effective_options(options):
    result=dict(options);state=result.get('data_state','raw')
    if state!='raw':result['minus_log']=state=='normalized'
    return result

def tomocupy_view(source,destination,state):
    """Reader-format adapter, not measured calibration. Never changes the source.

    The installed TomoCuPy reader allocates flat/dark buffers unconditionally.
    Compact neutral fields satisfy that format; the process-local entry bypasses
    dark/flat correction entirely, including its denominator stabilization.
    """
    source=Path(source).resolve()
    with h5py.File(source,'r') as h,h5py.File(destination,'w') as out:
        group=out.create_group('exchange')
        for name in h['exchange']:
            if name not in ('data_white','data_dark'):group[name]=h5py.ExternalLink(str(source),'/exchange/'+name)
        shape=(1,*h['exchange/data'].shape[1:])
        for name,value in [('data_white',1.),('data_dark',0.)]:
            d=group.create_dataset(name,shape=shape,dtype='float32',chunks=(1,1,shape[2]),fillvalue=value)
            d.attrs['role']='internal reader compatibility only; not measured calibration'
        out.attrs.update(data_preparation=state,original_input=str(source),adapter='processed input; calibration bypassed in process-local entry')
    return Path(destination)
