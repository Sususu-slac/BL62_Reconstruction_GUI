"""Contracts that prevent unit, backend-selection and row-range regressions."""
import math
import unittest
from pathlib import Path
from ct_recon_gui.models import GeometryMode,ReconJob,ReconMethod,TOMO_METHODS
from ct_recon_gui.commands import build_wsl_argv
from ct_recon_gui.tofu_contract import plan,candidates,block_parameters,projection_angles

class TofuTests(unittest.TestCase):
    def options(self,**kw):
        d=dict(binning=3,center=1070.5,lamino_angle=30,reconstruction_type='full',search_kind='center',start_row=80,end_row=144,try_row=1024,center_search_width=1,center_search_step=.25,lamino_search_width=.5,lamino_search_step=.25)
        d.update(kw);return d

    def test_original_pixel_search_values_are_inclusive(self):
        self.assertEqual(candidates(1070.5,1,.25),[1069.5,1069.75,1070,1070.25,1070.5,1070.75,1071,1071.25,1071.5])
        self.assertEqual(len(candidates(0,.1,.01)),21)

    def test_output_rows_are_not_detector_rows(self):
        j=plan(self.options(),(1441,2048,2048))
        self.assertEqual(j['values'],list(range(10,18)))
        self.assertEqual(j['try_z'],0)
        b=block_parameters(j,0,3)
        self.assertEqual(b['region'][0],-118)
        self.assertEqual(math.ceil((b['region'][1]-b['region'][0])/b['region'][2]),3)

    def test_search_center_offsets_are_scaled_once(self):
        j=plan(self.options(reconstruction_type='try'),(1441,2048,2048));j.update(center_step=.25,angle_step=.25)
        b=block_parameters(j,0,1)
        self.assertAlmostEqual(b['region'][0],128.5-1/8)
        self.assertAlmostEqual(b['region'][2],.25/8)
        self.assertEqual(b['count'],1)

    def test_signed_tilt_candidates_use_ufo_angle_convention(self):
        j=plan(self.options(lamino_angle=-20,reconstruction_type='try',search_kind='angle'),(1441,2048,2048));j.update(center_step=.25,angle_step=.25)
        b=block_parameters(j,0,3)
        self.assertAlmostEqual(b['region'][0],math.radians(69.5))
        self.assertAlmostEqual(b['lamino_angle'],math.radians(70))

    def test_tofu_is_lamino_only_and_search_keeps_its_backend(self):
        self.assertNotIn(ReconMethod.LAMINO_TOFU,TOMO_METHODS)
        j=ReconJob(Path('a.h5'),Path('out'),mode=GeometryMode.LAMINOGRAPHY,method=ReconMethod.LAMINO_TOFU,center=1070.5,start_row=0,end_row=2048,try_row=1024,binning=3,gpu=1,tofu_block_slices=2)
        command=build_wsl_argv(j)
        self.assertEqual(command[2],'lamino_tofu')
        self.assertEqual(command[command.index('--gpu')+1],'1')
        self.assertEqual(command[command.index('--binning')+1],'3')
        self.assertEqual(command[command.index('--tofu-block-slices')+1],'2')
        self.assertEqual(command[command.index('--center')+1],'1070.5')

    def test_invalid_angle_search_rejected_before_preprocessing(self):
        with self.assertRaisesRegex(ValueError,'below 90'):
            plan(self.options(lamino_angle=89.8,reconstruction_type='try',search_kind='angle'),(1441,2048,2048))

    def test_wrapped_theta_is_not_replaced_or_rejected_as_zero_span(self):
        import numpy as np
        angles,info=projection_angles([350,355,0,5],'degrees')
        np.testing.assert_allclose(angles,np.deg2rad([350,355,0,5]))
        self.assertAlmostEqual(info['overall_angle'],math.radians(20))
        self.assertTrue(info['theta_exact'])

    def test_repeated_identical_directions_fail_before_reading_images(self):
        with self.assertRaises(ValueError):projection_angles([0,360,720],'degrees')

if __name__=='__main__':unittest.main()
