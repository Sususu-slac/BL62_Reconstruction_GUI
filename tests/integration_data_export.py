"""Read-only real input previews and asynchronous GUI export validation."""
import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import tifffile
from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from ct_recon_gui.main_window import MainWindow
from ct_recon_gui.hdf5_inspector import inspect_hdf5

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);app=QApplication([]);w=MainWindow();w.input_edit.setText(str(a.input));w.info=inspect_hdf5(a.input)
    report={'previews':[],'binning':[],'exports':[]};cases=[('projection','tiff','preserve'),('corrected','tiff','float32'),('projection','png','display16')];active={}
    def finish():
        (a.output/'data_export_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);w.close();app.quit()
    def export_next():
        if not cases:finish();return
        source,fmt,mode=cases.pop(0);active.update(source=source,format=fmt,mode=mode,output=a.output/f'{source}_{fmt}_{mode}')
        w.export_source_combo.setCurrentIndex(w.export_source_combo.findData(source));w.export_scope_combo.setCurrentIndex(w.export_scope_combo.findData('current'))
        w.export_format_combo.setCurrentIndex(w.export_format_combo.findData(fmt));w.export_mode_combo.setCurrentIndex(w.export_mode_combo.findData(mode))
        w.export_output_edit.setText(str(active['output']));w._start_export()
    def poll():
        if not active or w.export_thread is not None:return
        root=active['output'];metadata=json.loads((root/'export_metadata.json').read_text());files=list(root.glob('*.tiff'))+list(root.glob('*.png'));assert len(files)==1
        image=tifffile.imread(files[0]) if files[0].suffix=='.tiff' else np.array(Image.open(files[0]));assert image.shape==(2048,2048) and np.isfinite(image).all()
        expected={'preserve':np.uint16,'float32':np.float32,'display16':np.uint16}[active['mode']];assert image.dtype==expected,image.dtype
        report['exports'].append(dict(source=active['source'],format=active['format'],mode=active['mode'],dtype=str(image.dtype),passed=True));active.clear();QTimer.singleShot(0,export_next)
    for source in ('projection','flat','dark','corrected','attenuation'):
        w.input_source_combo.setCurrentIndex(w.input_source_combo.findData(source));w._configure_input_navigation(0);w._load_input_frame();assert w.input_view._pixmap is not None
        report['previews'].append(source)
    w.input_source_combo.setCurrentIndex(0)
    for exponent in range(4):
        w.binning_combo.setCurrentIndex(exponent);w._load_input_frame();assert w.input_view._pixmap.width()==2048//(2**exponent)
        report['binning'].append(2**exponent)
    w.binning_combo.setCurrentIndex(0);timer=QTimer();timer.setInterval(100);timer.timeout.connect(poll);timer.start();QTimer.singleShot(0,export_next)
    QTimer.singleShot(60000,app.quit);app.exec();return 0 if len(report['exports'])==3 else 1

if __name__=='__main__':raise SystemExit(main())
