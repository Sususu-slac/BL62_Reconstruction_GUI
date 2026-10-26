import unittest
import json
import numpy as np
from sam3_rotation_center.adaptive import grid,plan,global_filter,coverage_missing
from sam3_rotation_center.metrics import MaskGeometry,mask_nms
from sam3_rotation_center.reporting import round_record,recommendation

class AdaptiveTest(unittest.TestCase):
    def test_global_coverage_and_quarter_grid(self):
        x=grid([[0,2047]],20,2047)
        self.assertEqual((x[0],x[-1],len(x)),(0,2047,104))
        x=grid([[200.12,208.73]],.25,2047)
        self.assertTrue(all(v*4==round(v*4) for v in x))

    def test_multimodal_peaks_survive(self):
        x=np.arange(0,1001,10)
        y=.8*np.exp(-((x-300)/40)**2)+.76*np.exp(-((x-700)/40)**2)
        rows=[{'center':float(a),'multithreshold_score':float(b),'mean_threshold_count':100*b} for a,b in zip(x,y)]
        result=plan(rows,10,1000)
        json.dumps(result)  # scipy/numpy peak indices must not leak scalar types.
        self.assertTrue(any(abs(p['center']-300)<20 for p in result['peaks']))
        self.assertTrue(any(abs(p['center']-700)<20 for p in result['peaks']))
        self.assertLess(result['next_step'],10)

    def test_no_detections_do_not_produce_fake_center(self):
        with self.assertRaises(ValueError):plan([{'center':0,'mean_threshold_count':0,'multithreshold_score':0}],1,10)

    def test_nms_threshold_subsets_and_ties(self):
        rng=np.random.default_rng(6)
        for _ in range(20):
            masks=rng.random((25,16,16))>.7
            masks[2]=masks[1];masks[4]=masks[3]
            scores=rng.choice([.2,.3,.4,.6,.8],size=25)
            geometry=MaskGeometry(masks)
            for threshold in [.2,.3,.4,.5,.6]:
                ids=np.flatnonzero(scores>=threshold)
                expected=[int(ids[i]) for i in mask_nms(masks[ids],scores[ids])]
                self.assertEqual(geometry.nms(scores,indices=ids),expected)

    def test_history_retains_measured_global_range(self):
        d={'intervals':[[900,1100]],'peaks':[]}
        r=round_record(1,20,[[0,2047]],d,104,104,5,200)
        self.assertEqual(r['intervals'],[[0,2047]])
        self.assertEqual(r['next_intervals'],[[900,1100]])

    def test_nearly_tied_peaks_are_reported(self):
        r=recommendation([{'center':1019.5,'score':.9065},{'center':1029,'score':.9016}])
        self.assertTrue(r['ambiguity'])
        self.assertEqual(r['recommendation_status'],'ambiguous')

    def test_uneven_density_does_not_change_same_underlying_curve(self):
        x=np.array([0.,20.,40.,60.,80.,100.])
        y=np.array([0.,.2,.9,.3,.1,0.])
        sparse=[{'center':float(a),'multithreshold_score':float(b)} for a,b in zip(x,y)]
        dense_x=np.unique(np.r_[x,np.arange(40,60,.25)])
        dense=[{'center':float(a),'multithreshold_score':float(b)} for a,b in zip(dense_x,np.interp(dense_x,x,y))]
        np.testing.assert_allclose(global_filter(sparse,2.5,100)[3],global_filter(dense,2.5,100)[3],atol=1e-12)

    def test_previous_global_peak_is_not_dropped_when_new_points_are_local(self):
        x=np.unique(np.r_[np.arange(0,1001,20),np.arange(650,751)])
        y=.9*np.exp(-((x-200)/30)**2)+.8*np.exp(-((x-700)/30)**2)
        rows=[{'center':float(a),'multithreshold_score':float(b),'mean_threshold_count':100*b} for a,b in zip(x,y)]
        d=plan(rows,1,1000)
        self.assertAlmostEqual(d['peaks'][0]['center'],200,delta=1)
        self.assertEqual(d['filter_bounds'],[0.,1000.])
        self.assertEqual(d['cumulative_points'],len(rows))

    def test_coarse_global_peak_needs_final_resolution_support(self):
        known=grid([[197,203]],.25,1000)+[700]
        missing=coverage_missing(known,[{'center':200},{'center':700}],.25,1000)
        self.assertTrue(missing)
        self.assertTrue(all(697<=c<=703 for c in missing))

    def test_confidence_reward_drives_selection_in_its_original_units(self):
        x=np.arange(0,1001,10)
        reward=100*np.exp(-((x-700)/40)**2)
        alternate=np.exp(-((x-200)/40)**2)
        rows=[{'center':float(c),'confidence_reward':float(r),'multithreshold_score':float(m),'mean_threshold_count':r}
              for c,r,m in zip(x,reward,alternate)]
        d=plan(rows,10,1000,score_key='confidence_reward')
        self.assertAlmostEqual(d['peaks'][0]['center'],700,delta=10)
        self.assertGreater(d['peaks'][0]['score'],50)
        self.assertEqual(d['score_key'],'confidence_reward')
        self.assertEqual(d['score_scale'],100)

    def test_broad_global_peak_does_not_force_wide_final_search(self):
        x=np.arange(0,1001,1)
        y=100*np.exp(-((x-500)/200)**2)
        rows=[{'center':float(c),'confidence_reward':float(r),'mean_threshold_count':r} for c,r in zip(x,y)]
        d=plan(rows,.25,1000,score_key='confidence_reward')
        self.assertEqual(d['filter_bounds'],[0.,1000.])
        for peak in d['peaks']:
            self.assertLessEqual(peak['high']-peak['low'],6.000001)

if __name__=='__main__':unittest.main()
