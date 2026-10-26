import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile
from analysis_worker import Pipeline
from ct_recon_gui.analysis_progress import progress_view,format_duration
from ct_recon_gui.curve_view import curve_view_bounds
from ct_recon_gui.runtime import defaults

class ProgressTests(unittest.TestCase):
    def test_round_counter_is_only_used_during_center_search(self):
        common=dict(round=7,total=2048,completed=1024)
        search=progress_view(dict(common,stage='Finding Center'))
        self.assertIn('Round 7',search['count']);self.assertIn('Current round',search['format'])
        for stage in ('Reconstructing','Segmenting'):
            result=progress_view(dict(common,stage=stage))
            self.assertNotIn('Round',result['count']);self.assertEqual(result['count'],'1024 / 2048 images')
        self.assertIn('Reconstruction completed — segmenting all data',result['title'])

    def test_unknown_model_loading_progress_is_indeterminate(self):
        result=progress_view(dict(stage='Loading SAM3',phase='center_search',indeterminate=True,round=1,total=104,completed=104))
        self.assertTrue(result['indeterminate']);self.assertEqual(result['count'],'Round 1')

    def test_optional_segmentation_changes_stage_count(self):
        result=progress_view(dict(stage='Reconstructing',segment_all=False,total=10,completed=2))
        self.assertIn('Stage 2/2',result['title'])

    def test_completion_and_long_duration_are_explicit(self):
        result=progress_view(dict(stage='Completed',round=7,total=2048,completed=2048))
        self.assertEqual(result['title'],'All tasks completed');self.assertEqual(result['count'],'2048 / 2048 images')
        self.assertEqual(format_duration(6552.19),'01:49:12');self.assertEqual(format_duration(61),'01:01')

    def test_auto_zoom_changes_only_display_bounds(self):
        graph={'centers':[0,10,20,30,40,50,60,70,80,90,100],
               'scores':[1000,1,1,1,10,12,10,1,1,1,1000],
               'curve':{'start':0,'step':10,'scores':[1000,1,1,1,10,12,10,1,1,1,1000]},
               'view_intervals':[[40,60]],'filter_bounds':[0,100]}
        original=copy.deepcopy(graph);local=curve_view_bounds(graph);global_view=curve_view_bounds(graph,False)
        self.assertEqual(graph,original)
        self.assertLess(local[1]-local[0],100);self.assertLess(local[3],1000)
        self.assertEqual(global_view[:2],(0.,100.));self.assertGreater(global_view[3],1000)

    def test_multiple_current_intervals_are_all_visible(self):
        graph={'centers':[0,50,100],'scores':[1,3,1],
               'curve':{'start':0,'step':50,'scores':[1,3,1]},
               'view_intervals':[[20,30],[70,80]],'filter_bounds':[0,100]}
        bounds=curve_view_bounds(graph);self.assertLessEqual(bounds[0],20);self.assertGreaterEqual(bounds[1],80)

    def test_segmentation_keeps_outputs_and_records_substep_times(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'recon_00000.tiff';image=np.arange(120,dtype=np.float32).reshape(10,12)
            tifffile.imwrite(source,image)
            masks=np.zeros((2,10,12),bool);masks[0,2:7,2:7]=True;masks[1,5:9,5:10]=True
            prediction=SimpleNamespace(masks=masks,scores=np.array([.9,.8],np.float32))
            pipe=Pipeline({**defaults(),'output':str(root/'out'),'segment_all':True})
            pipe.model=SimpleNamespace(predict=lambda _image:prediction)
            self.assertEqual(pipe.segment([source]),1)
            expected=np.zeros((10,12),np.uint16);expected[masks[0]]=1;expected[masks[1]&(expected==0)]=2
            np.testing.assert_array_equal(tifffile.imread(root/'out/segmentation/recon_00000.tiff'),expected)
            with np.load(root/'out/segmentation/recon_00000_instances.npz') as packed:
                restored=np.unpackbits(packed['masks_packed'],axis=1)[:,:120].reshape(2,10,12)
                np.testing.assert_array_equal(restored,masks)
            rows=list(csv.DictReader((root/'out/segmentation/timings.csv').open()))
            self.assertEqual(len(rows),1);self.assertGreater(float(rows[0]['total_seconds']),0)
            self.assertIn('instance_npz_write_seconds',rows[0]);self.assertEqual(pipe.status['phase'],'segmentation')
            self.assertEqual(pipe.status['completed'],1)

    def test_preview_file_has_current_bounds_and_global_curve(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);image=root/'candidate.tiff';tifffile.imwrite(image,np.arange(100,dtype=np.float32).reshape(10,10))
            pipe=Pipeline({**defaults(),'output':str(root/'out')});pipe.low=0;pipe.high=100
            pipe.status.update(round=2,search_intervals=[[40,60]])
            pipe.decision=lambda _rows,_step:{'peaks':[{'center':50}],
                'filter_curve':{'start':0,'step':25,'scores':[1,2,3,2,1]},'filter_bounds':[0,100]}
            row=dict(center=50,file=str(image),confidence_reward=3,mean_threshold_count=2,
                     persistence=1,matched_fraction=1,mean_match_iou=1,area_stability=1,valid_area_fraction=1)
            pipe.preview({50:row},1,True)
            graph=json.loads((root/'out/center_search/plot.json').read_text())
            self.assertEqual(graph['view_intervals'],[[40,60]])
            self.assertEqual(graph['filter_bounds'],[0,100]);self.assertEqual(len(graph['curve']['scores']),5)

if __name__=='__main__':unittest.main()
