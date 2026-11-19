"""Job-scoped Linux controller for the manual UFO GPU backend."""
from __future__ import annotations
from pathlib import Path
import json, os, signal, subprocess, sys, time, tempfile, shutil
import numpy as np
import tifffile
from ct_recon_gui.tofu_contract import block_parameters

def progress(stage,done=0,total=0,detail=''):
    print('PROGRESS '+json.dumps(dict(stage=stage,completed=done,total=total,detail=detail)),flush=True)

def environment(runtime,gpu,work=None,stop=None):
    prefix=Path(runtime['ufo_prefix']);cuda=Path(runtime['cuda_prefix']);plugin=Path(runtime['plugin_root'])
    env=dict(os.environ)
    env.update(PYTHONNOUSERSITE='1',PYTHONUTF8='1',PYTHONPATH=str(prefix/'usr/lib/python3/dist-packages'),
               GI_TYPELIB_PATH=str(prefix/'usr/lib/x86_64-linux-gnu/girepository-1.0'),
               LD_LIBRARY_PATH=':'.join(map(str,[cuda/'lib',prefix/'usr/lib/x86_64-linux-gnu',prefix/'usr/lib/llvm-16/lib',prefix/'usr/lib/x86_64-linux-gnu/blas',prefix/'usr/lib/x86_64-linux-gnu/lapack',Path('/usr/lib/wsl/lib')])),
               UFO_PLUGIN_PATH=str(plugin)+':'+str(prefix/'usr/lib/x86_64-linux-gnu/ufo'),
               UFO_KERNEL_PATH=str(plugin/'kernels')+':'+str(prefix/'usr/share/ufo'),
               OCL_ICD_VENDORS=str(cuda/'etc/OpenCL/vendors'),POCL_DEVICES='cuda',
               POCL_CUDA_TOOLKIT_PATH=str(cuda),POCL_CUDA_DISABLE_QUEUE_THREADS='0',
               CUDA_VISIBLE_DEVICES=str(gpu),UFO_DEVICE_TYPE='gpu',UFO_DEVICES='0',BL62_TOFU_REQUIRE_GPU='1')
    if work:env['BL62_TOFU_ANGLES_FILE']=str(work/'angles.bin')
    if stop:env['BL62_TOFU_STOP_FILE']=str(stop)
    return env

def main():
    request=Path(sys.argv[1]);info=json.loads(request.read_text());runtime=info['runtime']
    if '--cancel' in sys.argv:
        process=json.loads((request.parent/'process.json').read_text())
        if process['nonce']!=info['nonce']:raise RuntimeError('Job identity does not match')
        pid=int(process['pid']);cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        if str(request).encode() not in cmd or b'tofu_worker.py' not in cmd or os.getpgid(pid)!=pid:
            raise RuntimeError('Refusing to stop a different process group')
        os.killpg(pid,signal.SIGTERM);return 0
    if info.get('probe'):
        plugin=Path(runtime['plugin_root'])
        if not (plugin/'libufofilterlamino-backproject.so').is_file() or not (plugin/'kernels/z_kernel.cl').is_file():raise RuntimeError('Tofu GPU plugin is not installed')
        # Only import the GI runtime in its own process, never into Conda Python.
        env=environment(runtime,info.get('gpu',0))
        code='import gi; gi.require_version("Ufo","1.0"); from gi.repository import Ufo; s=Ufo.FixedScheduler(); assert s.get_resources().get_gpu_nodes(), "No OpenCL GPU"; print("Tofu GPU runtime ready",flush=True)'
        return subprocess.run([runtime.get('ufo_python','/usr/bin/python3'),'-c',code],env=env).returncode
    try:os.setsid()
    except PermissionError:
        if os.getpgrp()!=os.getpid():raise
    control=Path(info['work']);stop=Path(info['stop_file'])
    local=Path(tempfile.mkdtemp(prefix='bl62-tofu-'))
    (control/'process.json').write_text(json.dumps(dict(pid=os.getpid(),nonce=info['nonce'],local_work=str(local))))
    def terminated(_signum,_frame):raise SystemExit(130)
    signal.signal(signal.SIGTERM,terminated)
    try:
        staging_started=time.perf_counter()
        required=(control/'projections.tiff').stat().st_size
        if shutil.disk_usage(local).free < required+2*1024**3:raise RuntimeError('Not enough temporary space in the Tofu runtime')
        for name in ['projections.tiff','angles.bin']:
            source=control/name;size=source.stat().st_size;copied=0
            with source.open('rb') as src,(local/name).open('wb') as dst:
                while True:
                    if stop.exists():return 130
                    data=src.read(8*1024**2)
                    if not data:break
                    dst.write(data);copied+=len(data)
                    if name=='projections.tiff' and (copied%(64*1024**2)==0 or copied==size):
                        total_mib=(size+1024**2-1)//1024**2
                        progress('Staging GPU data',total_mib if copied==size else copied//1024**2,total_mib,'MiB')
            if copied!=size:raise RuntimeError('The projection cache was not completely copied')
        info['staging_seconds']=time.perf_counter()-staging_started
        return execute_job(info,runtime,control,local)
    finally:
        shutil.rmtree(local,ignore_errors=True)


def execute_job(info,runtime,control,work):
    output=Path(info['output']);stop=Path(info['stop_file']);job=info['job']
    env=environment(runtime,info['gpu'],work,stop);completed=0;total=len(job['values']);started=time.perf_counter();durations=[]
    entry=Path(__file__).with_name('tofu_gpu_entry.py')
    for start in range(0,total,job['block_slices']):
        if stop.exists():return 130
        end=min(total,start+job['block_slices']);block=block_parameters(job,start,end)
        block_path=work/'block.tiff';block_path.unlink(missing_ok=True)
        payload={**info,'work':str(work),'block':block};block_request=work/'block.json';block_request.write_text(json.dumps(payload))
        gpu_total=((total+job['block_slices']-1)//job['block_slices'])*job['nproj']
        gpu_base=(start//job['block_slices'])*job['nproj']
        progress('Reconstructing',gpu_base,gpu_total,f'Output block {start+1}–{end}')
        cmd=[runtime.get('ufo_python','/usr/bin/python3'),str(entry),str(block_request)]
        child=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        error=False;device_confirmed=False
        with (work/'block.log').open('w') as log:
            for line in child.stdout:
                log.write(line)
                if line.startswith('TOFU_PROJECTIONS '):
                    _,done,count=line.split();progress('Reconstructing',gpu_base+int(done),gpu_total,f'Output block {start+1}–{end}; projections {done}/{count}')
                else:print(line,end='',flush=True)
                if line.startswith('TOFU_DEVICE GPU '):device_confirmed=True
                if line.startswith('TOFU_EXECUTION_SECONDS '):durations.append(float(line.split()[-1]))
                if any(token in line for token in ('Traceback (most recent call last)','CUDA_ERROR_','PTHREAD ERROR','CL_BUILD_PROGRAM_FAILURE')):error=True
        code=child.wait()
        if stop.exists():return 130
        if code or error or not device_confirmed or not block_path.is_file():raise RuntimeError(f'Tofu GPU block failed (exit {code}); no CPU fallback was used')
        progress('Saving',completed,total)
        with tifffile.TiffFile(block_path) as image:
            if len(image.pages)!=block['count']:raise ValueError('Tofu block output count does not match the requested range')
            for offset,page in enumerate(image.pages):
                if stop.exists():return 130
                a=np.asarray(page.asarray(),np.float32)
                if a.shape!=(job['width'],job['width']) or not np.isfinite(a).all():raise ValueError('Invalid output image')
                a=np.ascontiguousarray(a[:,::-1]*(4/job['overall_angle']))
                index=start+offset;value=job['values'][index]
                if job['kind']=='full':name=f'recon_{int(value):05d}.tiff'
                else:name=f'recon_{int(job["try_z"]+job["height"]/2):05d}_{job["kind"]}{value:.6f}.tiff'
                target=output/name;temporary=target.with_suffix('.part')
                tifffile.imwrite(temporary,a,photometric='minisblack');temporary.replace(target)
                completed+=1;progress('Saving',completed,total);print('RESULT '+str(target),flush=True)
        block_path.unlink()
    result=dict(completed=completed,expected=total,worker_seconds=time.perf_counter()-started,gpu_block_seconds=durations,
                staging_seconds=info['staging_seconds'],gpu=info['gpu'],gpu_required=True,theta_exact=True,output_dtype='float32',output_flip_x=True)
    (control/'result.json').write_text(json.dumps(result,indent=2));progress('Completed',completed,total);return 0

if __name__=='__main__':raise SystemExit(main())
