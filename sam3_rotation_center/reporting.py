"""Search history keeps measured and proposed intervals separate."""
import numpy as np
from .adaptive import grid

def round_record(number, step, intervals, decision, samples, new, recon_seconds, seconds):
    return {**decision, 'round':number, 'step':step,
            'intervals':[list(x) for x in intervals],
            'next_intervals':decision['intervals'],
            'samples':samples, 'new_evaluations':new,
            'reconstruction_seconds':recon_seconds, 'seconds':seconds}

def recommendation(peaks,score_scale=1.0):
    near=[p for p in peaks if p['score']>=peaks[0]['score']-0.03*score_scale]
    ambiguous=len(near)>1
    return {'ambiguity':ambiguous,
            'recommendation_status':'ambiguous' if ambiguous else 'single_stable_peak',
            'competitive_centers':[p['center'] for p in near],
            'competitive_center_range':[min(p['center'] for p in near),max(p['center'] for p in near)]}

def plot_history(history, rows, center, upper, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cache={r['center']:r for r in rows}
    score_key=history[-1].get('score_key','multithreshold_score')
    ylabel='Confidence-weighted total score' if score_key=='confidence_reward' else 'SAM3 multithreshold score'
    fig,axes=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
    global_mode='filter_curve' in history[-1]
    if global_mode:
        axes[0].plot([r['center'] for r in rows],[r[score_key] for r in rows],
                     '.',ms=2,color='0.65',alpha=.6,label='All measured raw scores')
    for h in history:
        if 'filter_curve' in h:
            curve=h['filter_curve']
            axis=curve['start']+np.arange(len(curve['scores']))*curve['step']
            axes[0].plot(axis,curve['scores'],lw=1,alpha=.8,
                         label=f'R{h["round"]}: {h["cumulative_points"]} points, sigma {h["gaussian_sigma_pixels"]:g}px')
            continue
        cx=np.array(grid(h['intervals'],h['step'],upper))
        groups=np.split(np.arange(len(cx)),np.flatnonzero(np.diff(cx)>1.51*h['step'])+1)
        for part,ids in enumerate(groups):
            axes[0].plot(cx[ids],[cache[c][score_key] for c in cx[ids]],'.-',
                         ms=3,alpha=.6,label=f'R{h["round"]}: {h["step"]:g}px' if part==0 else None)
    axes[0].set(xlabel='Rotation center (original pixels)',ylabel=ylabel,xlim=(0,upper))
    axes[0].legend(ncol=3,fontsize=8);axes[0].grid(alpha=.2)
    h=history[-1]
    cx=np.array(h['cumulative_centers'] if global_mode else grid(h['intervals'],h['step'],upper))
    smooth=h['smoothed_scores']
    axes[1].plot(cx,[cache[c][score_key] for c in cx],'.',label='Measured score')
    if global_mode:
        curve=h['filter_curve'];axis=curve['start']+np.arange(len(curve['scores']))*curve['step']
        axes[1].plot(axis,curve['scores'],'-',color='C1')
        left=min(p['center'] for p in h['peaks'])-10
        right=max(p['center'] for p in h['peaks'])+10
        axes[1].set_xlim(left,right)
        raw=np.array([cache[c][score_key] for c in cx])
        local=np.r_[raw[(cx>=left)&(cx<=right)],np.asarray(curve['scores'])[(axis>=left)&(axis<=right)]]
        pad=max(float(np.ptp(local))*0.1,0.002)
        axes[1].set_ylim(float(local.min())-pad,float(local.max())+pad)
    else:
        for ids in np.split(np.arange(len(cx)),np.flatnonzero(np.diff(cx)>1.51*h['step'])+1):
            axes[1].plot(cx[ids],[smooth[c] if c in smooth else smooth[str(c)] for c in cx[ids]],'-',color='C1')
    for i,p in enumerate(h['peaks']):
        axes[1].axvline(p['center'],color=f'C{i+2}',linestyle='-' if i==0 else '--',
                       label=f'Peak {p["center"]:g}: {p["score"]:.4f}')
    axes[1].set(xlabel='Rotation center (original pixels)',ylabel=ylabel)
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.savefig(output,dpi=150);plt.close(fig)
