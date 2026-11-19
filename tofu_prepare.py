"""Bounded-memory native preprocessing and the Windows-to-WSL job bridge."""
from __future__ import annotations
import json, math, os, shutil, subprocess, sys, threading, time, uuid
from pathlib import Path
import h5py
import numpy as np
from ct_recon_gui.tofu_contract import plan,projection_angles
from ct_recon_gui.tofu_runtime import configuration,configuration_path
from ct_recon_gui.paths import windows_to_wsl


class Cancelled(RuntimeError):
    pass


def atomic_json(path, value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def progress(stage, done=0, total=0, detail=''):
    print('PROGRESS '+json.dumps(dict(stage=stage,completed=int(done),total=int(total),detail=detail)),flush=True)


def check_stop(output):
    if (Path(output)/'.stop_requested').exists():
        raise Cancelled('Stopped safely; completed output slices have been retained')


def linux_path(path):
    return windows_to_wsl(str(Path(path).resolve())) if os.name=='nt' else str(Path(path).resolve())


def runtime_command(runtime, worker, request, cancel=False):
    command=[runtime['python'],linux_path(worker),linux_path(request)]
    if cancel:command.append('--cancel')
    if os.name=='nt':
        command=['wsl.exe','--distribution',runtime['distribution'],'--exec',*command]
    return command


def run_child(command, output, cancel_command, *, stop_grace=10, cancel_timeout=15):
    # Forward output without blocking cancellation on an idle stdout pipe.
    with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                          encoding='utf-8',errors='replace',cwd=str(output)) as child:
        def reader():
            for line in child.stdout:print(line,end='',flush=True)
        thread=threading.Thread(target=reader,daemon=True);thread.start();requested=None
        while child.poll() is None:
            if (output/'.stop_requested').exists():
                requested=requested or time.monotonic()
                if time.monotonic()-requested>stop_grace:
                    helper_ok=False
                    try:helper_ok=subprocess.run(cancel_command,timeout=cancel_timeout,check=False,cwd=str(output)).returncode==0
                    except (subprocess.TimeoutExpired,OSError) as error:print('Job cancellation helper did not respond: '+str(error),flush=True)
                    if not helper_ok:child.terminate()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill()
                    break
            time.sleep(.2)
        code=child.wait();thread.join(timeout=5)
    check_stop(output)
    if code:raise subprocess.CalledProcessError(code,command)


def _prepare(args, work, job):
    import cupy as cp
    import tifffile
    from backend_runner import preprocess_astra
    from tomocupy.reconstruction.fbp_filter import FBPFilter
    n,h,w=job['nproj'],job['height'],job['width'];factor=job['factor']
    timing=job['preparation_timing']={};clock=time.perf_counter()
    progress('Allocating temporary storage')
    # Transposed backing storage makes full-angle sinogram bands contiguous.
    # Only one input frame and one row band are materialized in host memory.
    raw=np.lib.format.open_memmap(work/'raw.npy',mode='w+',dtype=np.float32,shape=(h,n,w))
    timing['raw_cache_allocation_seconds']=time.perf_counter()-clock;clock=time.perf_counter()
    try:
        with h5py.File(args.input,'r') as handle:
            dataset=handle['exchange/data'];theta_dataset=handle['exchange/theta']
            theta_rad,angle_info=projection_angles(theta_dataset[:],theta_dataset.attrs.get('units',''))
            if len(theta_rad)!=n:raise ValueError('Theta/projection count mismatch')
            # Explicit measured angles, never silently replace them with linspace.
            np.asarray(theta_rad+np.pi,dtype='<f4').tofile(work/'angles.bin')
            job.update(angle_info)
            progress('Reading and binning',0,n)
            for k in range(n):
                check_stop(args.output)
                frame=np.asarray(dataset[k,:h*factor,:w*factor],np.float32)
                if factor!=1:frame=frame.reshape(h,factor,w,factor).mean(axis=(1,3))
                raw[:,k,:]=frame
                if k%16==0 or k+1==n:progress('Reading and binning',k+1,n)
            timing['read_binning_seconds']=time.perf_counter()-clock;clock=time.perf_counter()
            def fields(key):
                data=handle[key];count=data.shape[0] if data.ndim==3 else 1
                result=np.empty((count,h,w),np.float32)
                for k in range(count):
                    check_stop(args.output)
                    progress('Loading flat fields' if key.endswith('data_white') else 'Loading dark fields',k,count)
                    a=np.asarray(data[k,:h*factor,:w*factor] if data.ndim==3 else data[:h*factor,:w*factor],np.float32)
                    result[k]=a.reshape(h,factor,w,factor).mean(axis=(1,3)) if factor!=1 else a
                return result
            flat,dark=(fields('exchange/data_white'),fields('exchange/data_dark')) if args.data_state=='raw' else (None,None)
            timing['flat_dark_seconds']=time.perf_counter()-clock
        ne=4*w;band=max(1,min(int(args.chunk_size),32,h));filtered=None
        with cp.cuda.Device(args.gpu):
            free,total=cp.cuda.runtime.memGetInfo()
            job['gpu_name']=cp.cuda.runtime.getDeviceProperties(args.gpu)['name'].decode()
            job['block_slices']=int(getattr(args,'tofu_block_slices',0)) or max(1,min(1024,int(min(free*.15,2*1024**3)//(w*w*4))))
            progress('Preparing GPU filter')
            clock=time.perf_counter()
            seed=FBPFilter(ne,n,band,'float32');response=seed.calc_filter(args.filter)
            frequency=cp.fft.rfftfreq(ne).astype(cp.float32)
            weight=response*cp.exp(-2*cp.pi*1j*frequency*(-args.center/factor+w/2))
            solvers={band:seed}
            timing['filter_setup_seconds']=time.perf_counter()-clock;clock=time.perf_counter()
            progress('Allocating projection cache')
            filtered=tifffile.memmap(work/'projections.tiff',shape=(n,h,w),dtype=np.float32,
                                     photometric='minisblack',bigtiff=True,metadata=None)
            timing['projection_cache_allocation_seconds']=time.perf_counter()-clock
            timing.update(preprocess_seconds=0.,filter_seconds=0.,cache_write_seconds=0.)
            for start in range(0,h,band):
                check_stop(args.output);stop=min(start+band,h);rows=stop-start
                progress('Preprocessing',start,h)
                clock=time.perf_counter()
                data=np.ascontiguousarray(raw[start:stop].transpose(1,0,2))
                data=preprocess_astra(data,None if flat is None else flat[:,start:stop],None if dark is None else dark[:,start:stop],args)
                timing['preprocess_seconds']+=time.perf_counter()-clock
                check_stop(args.output);progress('Filtering',start,h)
                clock=time.perf_counter()
                solver=solvers.setdefault(rows,seed) if rows==band else FBPFilter(ne,n,rows,'float32')
                padded=cp.pad(cp.asarray(data).swapaxes(0,1),((0,0),(0,0),(ne//2-w//2,ne//2-w//2)),mode='edge')
                padded=cp.ascontiguousarray(padded)
                weights=cp.broadcast_to(weight,(rows,1,len(weight))).copy().astype(cp.complex64)
                solver.filter(padded,weights,cp.cuda.Stream.null)
                ready=cp.asnumpy(padded[:,:,ne//2-w//2:ne//2+w//2].swapaxes(0,1))
                timing['filter_seconds']+=time.perf_counter()-clock;clock=time.perf_counter()
                if not np.isfinite(ready).all():raise ValueError('Preprocessed projections contain NaN or Inf')
                filtered[:,start:stop]=ready
                timing['cache_write_seconds']+=time.perf_counter()-clock
                del padded,weights,ready,data
                progress('Preprocessing and filtering',stop,h)
            clock=time.perf_counter();progress('Flushing projection cache')
            filtered.flush();filtered._mmap.close();filtered=None
            timing['cache_flush_seconds']=time.perf_counter()-clock
    finally:
        if 'filtered' in locals() and filtered is not None:filtered._mmap.close()
        raw._mmap.close()
        (work/'raw.npy').unlink(missing_ok=True)


def run(args):
    try:runtime=configuration()
    except RuntimeError:
        if os.name!='nt' or configuration_path().is_file():raise
        script=Path(__file__).with_name('setup_tofu_runtime.ps1')
        if not script.is_file():raise
        progress('Setting up Tofu GPU runtime')
        subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(script)],check=True,timeout=900)
        runtime=configuration()
    output=Path(args.output);work=output/'.tofu_work'
    output.mkdir(parents=True,exist_ok=True);work.mkdir(exist_ok=False)
    started=time.perf_counter();source=Path(args.input).resolve();stat=source.stat()
    audit=dict(method='lamino_tofu',input=str(source),output=str(output.resolve()),
               input_size=stat.st_size,input_mtime_ns=stat.st_mtime_ns,status='running')
    command=[]
    try:
        check_stop(output)
        with h5py.File(source,'r') as h:job=plan(vars(args),h['exchange/data'].shape)
        audit['job']=job
        needed=job['nproj']*job['height']*job['width']*8+len(job['values'])*job['width']**2*4
        if shutil.disk_usage(work).free<needed+1024**3:raise RuntimeError(f'Tofu needs approximately {needed/1024**3:.1f} GiB free in the output drive')
        request=work/'request.json'
        # Verify the Linux environment before spending time on preprocessing.
        root=Path(__file__).resolve().parent;worker=root/'tofu_worker.py'
        probe=dict(runtime=runtime,probe=True,gpu=args.gpu,nonce=uuid.uuid4().hex)
        atomic_json(request,probe);command=runtime_command(runtime,worker,request)
        progress('Checking Tofu GPU environment')
        subprocess.run(command,check=True,timeout=60,cwd=str(output))
        check_stop(output)
        _prepare(args,work,job)
        check_stop(output)
        job.update(center_step=args.center_search_step,angle_step=args.lamino_search_step)
        request_value=dict(runtime=runtime,probe=False,nonce=uuid.uuid4().hex,job=job,
                           work=linux_path(work),output=linux_path(output),stop_file=linux_path(output/'.stop_requested'),gpu=args.gpu)
        atomic_json(request,request_value)
        progress('Starting Tofu GPU reconstruction',0,len(job['values']))
        run_child(command,output,runtime_command(runtime,worker,request,cancel=True))
        audit.update(json.loads((work/'result.json').read_text()),status='completed',job=job)
        return command,dict(environment='native preprocessing + WSL UFO GPU',**audit)
    except Cancelled:
        audit['status']='stopped';raise
    except BaseException as error:
        audit.update(status='failed',error=str(error));raise
    finally:
        audit['elapsed_seconds']=time.perf_counter()-started
        atomic_json(output/'tofu_run_metadata.json',audit)
        # Only this job's transient projections are removed; completed TIFFs stay.
        shutil.rmtree(work,ignore_errors=True)
