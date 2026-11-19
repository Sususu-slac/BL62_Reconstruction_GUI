from __future__ import annotations

import sys
import json
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> int:
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs,True)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("BL62 Reconstruction")
    from .runtime import VERSION
    app.setApplicationVersion(VERSION)
    if '--theme-preview' in sys.argv:
        from .runtime import load_settings,settings_path,save_settings
        original=load_settings();preview=settings_path().with_name('theme_preview.json')
        os.environ['BL62_RECON_SETTINGS']=str(preview)
        if not preview.exists():save_settings({**original,'theme':'light'})
    window = MainWindow()
    if '--theme-preview' in sys.argv:window.setWindowTitle('BL62 Reconstruction — Theme Preview')
    window.show()
    if '--self-test' in sys.argv:
        from .runtime import APP_ROOT,WORKER_ROOT,reconstruction_python
        target=Path(sys.argv[sys.argv.index('--self-test')+1]);app.processEvents()
        report={'version':VERSION,'tabs':window.main_tabs.count(),'manual_methods':window.method_combo.count(),
            'log_collapsed':not window.log_toggle.isChecked(),
            'optimization_tab':window.center_tabs.tabText(2),'apply_parameter':window.apply_center_btn.text(),
            'advanced_settings_width':window.preprocessing_btn.width(),'segment_all':window.segment_all.isChecked(),
            'reconstruction_python':reconstruction_python(),
            'backend_present':(WORKER_ROOT/'backend_runner.py').is_file(),
            'analysis_worker_present':(WORKER_ROOT/'analysis_worker.py').is_file(),
            'bundled_sam3':(APP_ROOT/'sam3_repo/sam3/model_builder.py').is_file()}
        report['preparation_choices']=window.data_state_combo.count()
        try:
            import tempfile
            import numpy as np
            import tifffile
            from .conversion import images_to_h5,h5_to_images
            from .conversion_dialog import ConversionDialog
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);source=root/'source.tiff';pixels=np.arange(512,dtype=np.uint16).reshape(16,32);tifffile.imwrite(source,pixels)
                h5=images_to_h5([source],root/'test.h5',state='attenuation',theta=[0.])
                output=h5_to_images(h5,'exchange/data',root/'tiff')
                h5_to_images(h5,'exchange/data',root/'png',format='png',bits=16)
                report['conversion_roundtrip']=bool(np.array_equal(tifffile.imread(output/'image_000000.tiff'),pixels))
                for direction in ('import','export'):
                    dialog=ConversionDialog(direction,window,str(h5) if direction=='export' else '')
                    report['conversion_dialog_'+direction]=True;dialog.close()
        except Exception as error:report['conversion_error']=str(error);report['conversion_roundtrip']=False
        target.write_text(json.dumps(report,indent=2));window.close();return 0 if report['conversion_roundtrip'] else 1
    if '--validation-config' in sys.argv:
        from .release_validation import validate_release
        report=validate_release(app,window,sys.argv[sys.argv.index('--validation-config')+1])
        app.exec();return 0 if report['manual'] and (report.get('manual_only') or report['automatic']) else 1
    return app.exec()
