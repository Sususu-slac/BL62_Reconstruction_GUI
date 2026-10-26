"""Plan global-to-subpixel searches without using a known reference center."""
from dataclasses import asdict, dataclass
import math
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths


@dataclass
class Peak:
    center: float
    score: float
    low: float
    high: float
    edge: bool = False
    tentative: bool = False


def grid(intervals, step, upper):
    values = set()
    for lo, hi in intervals:
        lo, hi = max(0.0, lo), min(upper, hi)
        lo = math.floor(lo/step)*step
        hi = min(upper, math.ceil(hi/step)*step)
        values.update(round(float(x), 6) for x in np.arange(lo, hi+step*0.001, step))
        values.add(round(hi, 6))
    return sorted(x for x in values if 0 <= x <= upper)


def merge_intervals(intervals, upper):
    merged = []
    for lo, hi in sorted(intervals):
        lo, hi = max(0.0, lo), min(upper, hi)
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([float(lo), float(hi)])
    return merged


def global_filter(rows, sigma_pixels, upper, resolution=0.25, score_key='multithreshold_score'):
    """Filter a physical center axis, independent of sampling density.

    Only the saved raw scores are interpolated; no model evaluations are
    invented and no previously filtered values are filtered again.
    """
    rows = sorted(rows, key=lambda r: r['center'])
    x = np.array([float(r['center']) for r in rows])
    y = np.array([float(r[score_key]) for r in rows])
    if len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('At least two finite measured scores are required')
    if np.any(np.diff(x) <= 0):
        raise ValueError('Measured centers must be unique')
    axis = np.arange(0.0, upper + resolution*0.5, resolution)
    interpolated = np.interp(axis, x, y)
    filtered = gaussian_filter1d(interpolated, sigma_pixels/resolution,
                                 mode='nearest')
    return x, y, axis, filtered


def coverage_missing(known_centers, peaks, step, upper):
    """A globally rediscovered coarse peak must be measured before stopping."""
    radius = max(2*step, 3.0)
    required = grid([(p['center']-radius, p['center']+radius) for p in peaks], step, upper)
    known = set(known_centers)
    return [c for c in required if c not in known]


def plan(rows, step, upper, target=0.25, max_peaks=3, score_key='multithreshold_score'):
    """Rebuild one global filtered curve from all cumulative RAW scores."""
    if not rows or max(r['mean_threshold_count'] for r in rows) < 1:
        raise ValueError('No particle evidence: cannot choose a rotation center')
    sigma = max(step, 0.75)
    x, y, axis, smooth = global_filter(rows, sigma, upper, target, score_key)
    scale=max(float(y.max()),1.0) if score_key=='confidence_reward' else 1.0
    if np.ptp(y) < 1e-5:
        raise ValueError('Score curve is flat: center is unresolved')
    smooth_at_measured = np.interp(x, axis, smooth)
    local = list(find_peaks(smooth, prominence=0.005*scale)[0])
    if smooth[0] >= smooth[1]: local.append(0)
    if smooth[-1] >= smooth[-2]: local.append(len(axis)-1)
    if not local: local = [int(np.argmax(smooth))]
    candidates = []
    for i in local:
        edge = bool(i == 0 or i == len(axis)-1)
        measured = int(np.argmin(np.abs(x-axis[i])))
        center = float(x[measured])
        if edge:
            lo, hi = center-3*step, center+3*step
        else:
            width = peak_widths(smooth, [i], rel_height=0.5)
            lo = float(width[2][0]*target)-2*step
            hi = float(width[3][0]*target)+2*step
        radius = max(2*step, 3.0)
        # Global half-prominence widths can remain hundreds of pixels wide even
        # at the last round. Refine only a bounded neighborhood of each peak;
        # the complete cumulative curve still decides which peaks to revisit.
        radius_cap = max(6*step, 3.0)
        candidates.append(Peak(center, float(smooth_at_measured[measured]),
                               max(min(lo, center-radius), center-radius_cap),
                               min(max(hi, center+radius), center+radius_cap), edge))
    raw_i = int(np.argmax(y))
    if y[raw_i] > smooth.max()+0.08*scale and step>target:
        center=float(x[raw_i])
        candidates.append(Peak(center,float(y[raw_i]),center-2*step,center+2*step,False,True))
    candidates.sort(key=lambda p:p.score, reverse=True)
    chosen=[]
    best=candidates[0].score
    for peak in candidates:
        if peak.score < best-max(0.10*scale, 0.25*np.ptp(y)):
            continue
        if any(abs(peak.center-other.center)<max(3*step,3.0) for other in chosen):
            continue
        chosen.append(peak)
        if len(chosen)>=max_peaks:break
    intervals=merge_intervals([(p.low,p.high) for p in chosen],upper)
    feature_step=min(hi-lo for lo,hi in intervals)/12
    next_step=min(step/2,feature_step)
    if next_step>=1:next_step=float(max(1,math.floor(next_step)))
    else:next_step=0.5 if next_step>=0.5 else target
    return {'peaks':[asdict(p) for p in chosen], 'intervals':intervals,
            'next_step':max(target,next_step),
            'smoothed_scores':{float(c):float(v) for c,v in zip(x,smooth_at_measured)},
            'cumulative_centers':x.tolist(), 'cumulative_points':len(x),
            'filter_bounds':[0.0,float(upper)], 'gaussian_sigma_pixels':sigma,
            'score_key':score_key,'score_scale':scale,
            'refinement_radius_cap':max(6*step,3.0),
            'filter_curve':{'start':0.0,'step':target,'scores':smooth.tolist()}}
