"""Choose display bounds only; never alter cumulative scores or filtering."""
import numpy as np

def curve_view_bounds(graph, follow_search=True):
    curve=graph['curve'];values=np.asarray(curve['scores'],dtype=float)
    axis=curve['start']+np.arange(len(values))*curve['step']
    global_low,global_high=graph.get('filter_bounds',[float(axis[0]),float(axis[-1])])
    intervals=graph.get('view_intervals') if follow_search else None
    if intervals:
        low=min(pair[0] for pair in intervals);high=max(pair[1] for pair in intervals)
        margin=max((high-low)*.04,float(curve['step'])*2)
        low=max(global_low,low-margin);high=min(global_high,high+margin)
    else:low,high=global_low,global_high
    raw_x=np.asarray(graph['centers'],dtype=float);raw_y=np.asarray(graph['scores'],dtype=float)
    local=np.concatenate([values[(axis>=low)&(axis<=high)],raw_y[(raw_x>=low)&(raw_x<=high)]])
    local=local[np.isfinite(local)]
    if local.size:
        ylow=float(local.min());yhigh=float(local.max());pad=max((yhigh-ylow)*.08,abs(yhigh)*.002,1e-3)
        ylow=max(0.,ylow-pad);yhigh+=pad
    else:ylow,yhigh=0.,1.
    return float(low),float(high),ylow,yhigh
