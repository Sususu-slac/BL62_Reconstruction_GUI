import unittest

import numpy as np

from sam3_rotation_center.metrics import (
    centered_rolling_mean,
    mask_nms,
    summarize_masks,
)


class MetricsTest(unittest.TestCase):
    def test_mask_nms_removes_overlapping_lower_score_mask(self) -> None:
        masks = np.zeros((3, 10, 10), dtype=bool)
        masks[0, 1:5, 1:5] = True
        masks[1, 1:5, 1:5] = True
        masks[2, 6:9, 6:9] = True
        scores = np.array([0.9, 0.4, 0.8], dtype=np.float32)

        self.assertEqual(mask_nms(masks, scores, iou_threshold=0.5), [0, 2])

    def test_summarize_masks_applies_area_filter(self) -> None:
        masks = np.zeros((2, 10, 10), dtype=bool)
        masks[0, 0, 0] = True
        masks[1, 2:6, 2:6] = True
        scores = np.array([0.9, 0.8], dtype=np.float32)

        metrics, kept = summarize_masks(masks, scores, min_area=4)

        self.assertEqual(kept, [1])
        self.assertEqual(metrics.raw_count, 2)
        self.assertEqual(metrics.nms_count, 1)
        self.assertEqual(metrics.min_area, 16)

    def test_centered_rolling_mean_handles_edges(self) -> None:
        result = centered_rolling_mean(np.array([1, 2, 9, 2, 1]), radius=1)

        np.testing.assert_allclose(result, [1.5, 4.0, 13 / 3, 4.0, 1.5])

    def test_empty_masks_are_supported(self) -> None:
        masks = np.zeros((0, 10, 10), dtype=bool)
        scores = np.zeros(0, dtype=np.float32)

        metrics, kept = summarize_masks(masks, scores)

        self.assertEqual(kept, [])
        self.assertEqual(metrics.raw_count, 0)
        self.assertEqual(metrics.nms_count, 0)


if __name__ == "__main__":
    unittest.main()
