"""Describe the current phase without pretending adaptive search has a fixed total."""

SEARCH_STAGES={'Finding Center','Reconstructing Candidates','Finalizing Center Search'}

def phase_for_stage(stage, previous='preparing'):
    if stage in SEARCH_STAGES:return 'center_search'
    return {'Preparing':'preparing','Reconstructing':'reconstruction',
            'Preparing Segmentation':'segmentation','Segmenting':'segmentation',
            'Finalizing':'finalizing','Completed':'completed','Stopped':'stopped',
            'Failed':'failed'}.get(stage,previous)

def format_duration(seconds):
    seconds=max(0,int(seconds));hours,remainder=divmod(seconds,3600);minutes,seconds=divmod(remainder,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes:02d}:{seconds:02d}'

def progress_view(status, segment_all=True):
    stage=status.get('stage','Preparing');phase=status.get('phase') or phase_for_stage(stage)
    segment_all=status.get('segment_all',segment_all);stages=3 if segment_all else 2
    total=max(0,int(status.get('total',0)));done=max(0,min(int(status.get('completed',0)),total))
    unknown=bool(status.get('indeterminate'))
    result=dict(title=stage,detail=status.get('detail',''),count='',total=total,completed=done,
                indeterminate=unknown,format='Current stage: %p%')
    if phase=='center_search':
        label={'Finding Center':'Finding rotation center',
               'Reconstructing Candidates':'Reconstructing center candidates',
               'Finalizing Center Search':'Saving rotation-center results',
               'Loading SAM3':'Loading SAM3 model'}.get(stage,stage)
        result['title']=f'Stage 1/{stages} · {label}'
        if status.get('round') is not None:result['count']=f'Round {status["round"]}'
        if not unknown and total:result['count']+=f' · {done} / {total} centers'
        result['format']='Current round: %p%'
    elif phase=='reconstruction':
        result['title']=f'Stage 2/{stages} · Reconstructing all data'
        if not unknown and total:result['count']=f'{done} / {total} images'
    elif phase=='segmentation':
        result['title']='Stage 3/3 · Reconstruction completed — segmenting all data'
        if not unknown and total:result['count']=f'{done} / {total} images'
    elif phase=='preparing':result['title']='Preparing analysis'
    elif phase=='finalizing':result['title']='Saving the final analysis report'
    elif phase=='completed':
        result.update(title='All tasks completed',indeterminate=False,format='%p%',detail='')
        if total:result['count']=f'{done} / {total} images'
    elif phase=='stopped':result.update(title='Stopped',indeterminate=False,detail='')
    elif phase=='failed':result.update(title='Failed — see Log',indeterminate=False,detail=status.get('error',''))
    return result
