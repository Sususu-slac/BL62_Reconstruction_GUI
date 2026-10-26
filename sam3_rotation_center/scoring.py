"""Original six-term score, with geometry shared across thresholds."""
import numpy as np
from .metrics import MaskGeometry, summarize_prepared

DEFAULT_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6)


def score_prediction(prediction, thresholds=DEFAULT_THRESHOLDS, nms_iou=0.5,
                     high_score_threshold=0.7, min_area=0, max_area=None):
    masks, scores = prediction.masks, np.asarray(prediction.scores)
    geometry = MaskGeometry(masks)
    kept = geometry.nms(scores, nms_iou, min_area, max_area)
    metrics = summarize_prepared(geometry, scores, kept, high_score_threshold)
    counts, median_areas, selections = [], [], []
    for threshold in thresholds:
        selected = np.flatnonzero(scores >= threshold)
        indices = geometry.nms(scores, nms_iou, min_area, max_area, selected)
        selections.append(indices)
        counts.append(len(indices))
        median_areas.append(float(np.median(geometry.areas[indices])) if indices else 0.0)
    base = max(counts[0], 1)
    retained_scores = scores[selections[0]] if selections[0] else np.empty(0)
    rewards = np.clip((retained_scores-thresholds[0])/max(0.8-thresholds[0], 1e-12), 0, 1)
    positive_areas = np.array([x for x in median_areas if x > 0])
    return {
        **metrics.to_dict(),
        **{f'count_at_{t:.2f}': n for t, n in zip(thresholds, counts)},
        'mean_threshold_count': float(np.mean(counts)),
        'confidence_reward': float(rewards.sum()),
        'mean_particle_reward': float(rewards.mean()) if rewards.size else 0.0,
        'persistence': float(np.mean(counts)/base),
        'matched_fraction': float(counts[-1]/base),
        # Retained for compatibility: historically this is a presence flag, not IoU.
        'mean_match_iou': 1.0 if counts[-1] else 0.0,
        'area_stability': float(1/(1+positive_areas.std()/positive_areas.mean())) if positive_areas.size else 0.0,
        'valid_area_fraction': float(metrics.nms_count/max(metrics.raw_count, 1)),
    }, kept


def assign_scores(rows):
    """Use one denominator across all evaluated centers, including all branches."""
    maximum = max(1.0, max((r['mean_threshold_count'] for r in rows), default=0.0))
    for row in rows:
        row['multithreshold_score'] = float(
            0.40*row['mean_threshold_count']/maximum + 0.20*row['persistence']
            + 0.15*row['matched_fraction'] + 0.15*row['mean_match_iou']
            + 0.05*row['area_stability'] + 0.05*row['valid_area_fraction'])
    return rows
