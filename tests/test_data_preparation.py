import tempfile,unittest
from pathlib import Path
import h5py,numpy as np,tifffile
from ct_recon_gui.conversion import images_to_h5,h5_to_images,Cancelled
from ct_recon_gui.input_state import resolve,validate_options,tomocupy_view
from ct_recon_gui.hdf5_inspector import inspect_hdf5,read_input_frame,frame_count

class PreparationTests(unittest.TestCase):
    def test_roundtrip_and_theta_alignment(self):
        for dtype in (np.uint8,np.uint16,np.float32):
            with self.subTest(dtype=dtype),tempfile.TemporaryDirectory() as tmp:
                r=Path(tmp);source=r/'input';source.mkdir();arrays=[(np.arange(48).reshape(6,8)+k*5).astype(dtype) for k in range(5)]
                files=[]
                for i,a in enumerate(arrays):p=source/f'{i}.tiff';tifffile.imwrite(p,a);files.append(p)
                path=images_to_h5(files,r/'data.h5',state='attenuation',theta=np.arange(5)*.01,units='radians')
                self.assertEqual(resolve(path),'attenuation');info=inspect_hdf5(path);self.assertEqual(info.theta_unit,'radians');self.assertEqual(frame_count(info,'flat'),0)
                out=h5_to_images(path,'exchange/data',r/'out',start=1,end=5,stride=2)
                for i in (1,3):np.testing.assert_array_equal(tifffile.imread(out/f'image_{i:06d}.tiff'),arrays[i])
                np.testing.assert_allclose(np.loadtxt(out/'theta.csv'),[.01,.03])
                image,_=read_input_frame(path,'attenuation',2);np.testing.assert_array_equal(image,arrays[2])
    def test_reconstruction_has_no_theta(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);p=r/'a.tiff';tifffile.imwrite(p,np.zeros((8,8),np.uint16));path=images_to_h5([p],r/'v.h5',kind='reconstruction')
            with h5py.File(path) as h:self.assertIn('reconstruction/data',h);self.assertNotIn('exchange/theta',h)
    def test_invalid_and_cancel_are_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);p=r/'a.tiff';tifffile.imwrite(p,np.ones((8,8),np.float32))
            with self.assertRaises(ValueError):images_to_h5([p],r/'bad.h5',theta=[0,1])
            with self.assertRaises(Cancelled):images_to_h5([p],r/'cancel.h5',theta=[0],cancel=lambda:True)
            self.assertFalse((r/'cancel.h5').exists());self.assertFalse(list(r.glob('*.partial-*')))
            path=images_to_h5([p],r/'ok.h5',theta=[0])
            with self.assertRaises(Cancelled):h5_to_images(path,'exchange/data',r/'cancel',cancel=lambda:True)
            self.assertFalse((r/'cancel').exists());self.assertFalse(list(r.glob('*.partial-*')))
            with self.assertRaises(ValueError):images_to_h5([p],path,theta=[0])
    def test_explicit_state_and_backend_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'unknown.h5'
            with h5py.File(path,'w') as h:h.create_dataset('exchange/data',data=np.ones((2,4,8)));h.create_dataset('exchange/theta',data=[0,90])
            with self.assertRaises(ValueError):resolve(path)
            self.assertEqual(resolve(path,'normalized'),'normalized')
            with self.assertRaises(ValueError):resolve(path,'raw')
            with self.assertRaises(ValueError):validate_options(dict(data_state='attenuation',method='fourierrec',remove_stripe_method='vo-all'))
            with self.assertRaises(ValueError):validate_options(dict(data_state='normalized',method='unvalidated',remove_stripe_method='none'))
            view=tomocupy_view(path,Path(tmp)/'adapter.h5','normalized')
            with h5py.File(view) as h:self.assertIn('not measured calibration',h['exchange/data_white'].attrs['role'])
            with h5py.File(path) as h:self.assertNotIn('exchange/data_white',h)
    def test_shared_png_range(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);p=r/'s.h5'
            with h5py.File(p,'w') as h:h.create_dataset('volume',data=np.array([[[0,10]],[[10,20]]],np.float32))
            out=h5_to_images(p,'volume',r/'png',format='png',bits=16)
            a=np.asarray(Image.open(out/'image_000000.png'));b=np.asarray(Image.open(out/'image_000001.png'))
            self.assertEqual(a[0,1],b[0,0]);self.assertEqual(int(b[0,1]),65535)
