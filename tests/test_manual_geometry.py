import unittest
import numpy as np
from ct_recon_gui.geometry import output_row_limit,output_slice_count
from tomocupy_manual_entry import inclusive_shifts

class ManualGeometryTests(unittest.TestCase):
    def test_full_laminography_extent_exceeds_detector_height(self):
        self.assertEqual(output_row_limit(2048,'laminography',30,0),2366)
        self.assertEqual(output_row_limit(2048,'laminography',-30,0),2366)
    def test_binning_follows_native_even_output_height(self):
        self.assertEqual(output_row_limit(2048,'laminography',30,1),2368)
        self.assertEqual(output_slice_count(2048,'laminography',30,1),1184)
    def test_tomography_extent_is_unchanged(self):
        self.assertEqual(output_row_limit(2048,'tomography',30,0),2048)
    def test_quarter_pixel_search_has_both_endpoints(self):
        centers=sorted((1070.5-inclusive_shifts(1,.25,2)*2).tolist())
        self.assertEqual(centers,[1069.5+.25*i for i in range(9)])
    def test_unaligned_endpoint_does_not_expand_the_search(self):
        shifts=inclusive_shifts(1.1,.25,1)
        self.assertTrue((shifts>=-1.10001).all() and (shifts<=1.10001).all())
        np.testing.assert_allclose(np.diff(shifts),.25,atol=1e-6)

if __name__=='__main__':unittest.main()
