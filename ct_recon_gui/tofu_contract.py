"""Coordinates and job planning for the manual UFO GPU backend."""
from __future__ import annotations
from decimal import Decimal, ROUND_FLOOR
import math


def projection_angles(theta, unit=''):
    import numpy as np
    values=np.asarray(theta,dtype=np.float64).reshape(-1)
    if len(values)<2 or not np.isfinite(values).all():raise ValueError('Invalid projection angles')
    unit=str(unit).lower()
    radians='rad' in unit or ('deg' not in unit and np.max(np.abs(values))<=2*np.pi+.1)
    angles=values if radians else np.deg2rad(values)
    span=float(np.ptp(np.unwrap(angles)))
    if span<1e-8:raise ValueError('At least two distinct projection directions are required')
    return angles,dict(theta_unit='radians' if radians else 'degrees',theta_min=float(values.min()),
                       theta_max=float(values.max()),theta_exact=True,
                       overall_angle=span*len(values)/(len(values)-1))


def candidates(center: float, radius: float, step: float) -> list[float]:
    if not all(math.isfinite(x) for x in (center, radius, step)) or radius <= 0 or step <= 0:
        raise ValueError('Search radius and step must be finite and positive')
    c, r, s = map(lambda x: Decimal(str(x)), (center, radius, step))
    count = int((2*r/s).to_integral_value(rounding=ROUND_FLOOR)) + 1
    if count > 100001:
        raise ValueError('Too many search candidates; increase the step')
    return [float(c-r+i*s) for i in range(count)]


def plan(options: dict, shape: tuple[int, int, int]) -> dict:
    n, original_h, original_w = map(int, shape)
    factor = 2**int(options['binning'])
    h, w = original_h//factor, original_w//factor
    if h < 16 or w < 32 or w % 2:
        raise ValueError('Tofu GPU currently requires an even effective width and height ≥16 and width ≥32 pixels')
    center, angle = float(options['center']), float(options['lamino_angle'])
    if not math.isfinite(angle) or abs(angle) >= 90:
        raise ValueError('Lamino tilt must be finite and below 90 degrees')
    kind = 'full' if options['reconstruction_type'] == 'full' else options['search_kind']
    if kind == 'full':
        first = int(options['start_row'])//factor
        count = (int(options['end_row'])-int(options['start_row']))//factor
        indices = list(range(first, first+count))
        values = indices
    elif kind == 'center':
        values = candidates(center, options['center_search_width'], options['center_search_step'])
        if min(values) <= 0:
            raise ValueError('All candidate centers must be greater than zero')
        indices = []
    elif kind == 'angle':
        values = candidates(angle, options['lamino_search_width'], options['lamino_search_step'])
        if max(abs(x) for x in values) >= 90:
            raise ValueError('All candidate tilts must be below 90 degrees')
        indices = []
    else:
        raise ValueError('Unknown manual reconstruction type')
    if not values:
        raise ValueError('The selected range contains no output slices')
    return dict(kind=kind, values=values, output_indices=indices, nproj=n, height=h, width=w,
                original_shape=list(shape), factor=factor, center_original=center,
                tilt_original=angle, filtered_center=w/2, ufo_center=w/2+.5,
                ufo_y_center=h/2+.5, try_z=int(options.get('try_row') or 0)//factor-h/2,
                x_region=[-w//2+1, w//2+1, 1], y_region=[-w//2+1, w//2+1, 1],
                output_flip_x=True, output_dtype='float32')


def block_parameters(job: dict, start: int, stop: int) -> dict:
    values = job['values'][start:stop]
    if not values:
        raise ValueError('Empty output block')
    kind = job['kind']
    if kind == 'full':
        parameter, low, step = 'z', values[0]-job['height']/2, 1.
    elif kind == 'center':
        parameter = 'x-center'
        low = job['ufo_center']+(values[0]-job['center_original'])/job['factor']
        step = (values[1]-values[0])/job['factor'] if len(values)>1 else job['center_step']/job['factor']
    else:
        parameter = 'lamino-angle'
        low = math.radians(90+values[0])
        step = math.radians(values[1]-values[0]) if len(values)>1 else math.radians(job['angle_step'])
    # UFO converts (stop-start)/step with ceil: keep the half-open upper bound
    # away from integer roundoff so it produces exactly the requested count.
    high = low+(len(values)-.25)*step
    return dict(parameter=parameter, region=[low, high, step], count=len(values),
                center=[job['ufo_center'], job['ufo_y_center']], z=job['try_z'],
                lamino_angle=math.radians(90+job['tilt_original']))
