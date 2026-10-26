from pathlib import Path
from PySide6.QtWidgets import (QCheckBox,QComboBox,QDialog,QDialogButtonBox,QDoubleSpinBox,QFileDialog,
    QFormLayout,QHBoxLayout,QLineEdit,QPushButton,QSpinBox,QTabWidget,QVBoxLayout,QWidget,QScrollArea)
from .gpu_monitor import GpuSelector

class AnalysisSettings(QDialog):
    def __init__(self,settings,gpus,parent=None):
        super().__init__(parent);self.values=dict(settings);self.setWindowTitle('Analysis Settings');self.resize(700,610)
        layout=QVBoxLayout(self);tabs=QTabWidget();layout.addWidget(tabs)
        def page(name):
            panel=QWidget();form=QFormLayout(panel);form.setVerticalSpacing(12)
            scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(panel);tabs.addTab(scroll,name);return form
        def combo(items,value):
            w=QComboBox()
            for label,data in items:w.addItem(label,data)
            w.setCurrentIndex(max(0,w.findData(value)));return w
        def integer(value,maximum=100000):
            w=QSpinBox();w.setRange(0,maximum);w.setValue(value or 0);return w
        def number(value,minimum=0,maximum=100000,step=.25):
            w=QDoubleSpinBox();w.setRange(minimum,maximum);w.setDecimals(3);w.setSingleStep(step);w.setValue(value or 0);return w
        def path_row(key,directory=False):
            edit=QLineEdit(str(settings[key]));button=QPushButton('Browse');row=QWidget();r=QHBoxLayout(row);r.setContentsMargins(0,0,0,0);r.addWidget(edit,1);r.addWidget(button)
            def browse():
                value=QFileDialog.getExistingDirectory(self,'Select Folder',edit.text()) if directory else QFileDialog.getOpenFileName(self,'Select File',edit.text())[0]
                if value:edit.setText(value)
            button.clicked.connect(browse);return edit,row
        form=page('Reconstruction')
        self.output_mode=combo([('Input Folder','input'),('Custom Folder','custom')],'custom' if settings['output_root'] else 'input')
        form.addRow('Output Location',self.output_mode)
        self.output_root,output_row=path_row('output_root',True);form.addRow('Output Folder',output_row)
        output_row.setEnabled(bool(settings['output_root']));self.output_mode.currentIndexChanged.connect(lambda:output_row.setEnabled(self.output_mode.currentData()=='custom'))
        template=QLineEdit('{input_name}_analysis_{timestamp}');template.setReadOnly(True);form.addRow('Results Folder Name',template)
        methods=[('FourierRec','fourierrec'),('LpRec','lprec'),('LineRec','linerec'),('FBP_CUDA','fbp_cuda'),('SIRT_CUDA','sirt_cuda'),('CGLS_CUDA','cgls_cuda')]
        self.method=combo(methods,settings['method']);form.addRow('Reconstruction Method',self.method)
        self.gpu=GpuSelector();self.gpu.update_gpus(gpus);self.gpu.setValue(settings['gpu']);form.addRow('GPU',self.gpu)
        self.binning=combo([('1 × 1 · Original',0),('2 × 2',1),('4 × 4',2),('8 × 8',3)],settings['binning']);form.addRow('Binning',self.binning)
        self.all_rows=QCheckBox('All Slices');self.all_rows.setChecked(settings['start_row'] is None);form.addRow('Reconstruction Range',self.all_rows)
        self.start=integer(settings['start_row']);self.end=integer(settings['end_row'] or 2048)
        form.addRow('Start Row',self.start);form.addRow('End Row (exclusive)',self.end)
        for w in (self.start,self.end):w.setEnabled(not self.all_rows.isChecked());self.all_rows.toggled.connect(lambda checked,w=w:w.setEnabled(not checked))
        self.chunk=integer(settings['chunk_size'],256);self.chunk.setMinimum(4);form.addRow('Chunk Size',self.chunk)
        self.iterations=integer(settings['iterations'],10000);self.iterations.setMinimum(1);form.addRow('Iterations',self.iterations)
        self.method.currentIndexChanged.connect(lambda:self.iterations.setEnabled(self.method.currentData() in ('sirt_cuda','cgls_cuda')))
        self.iterations.setEnabled(settings['method'] in ('sirt_cuda','cgls_cuda'))
        form=page('Center Search')
        self.middle=QCheckBox('Middle Slice');self.middle.setChecked(settings['search_row'] is None);form.addRow('Center-search Slice',self.middle)
        self.row=integer(settings['search_row'] or 1023);self.row.setEnabled(not self.middle.isChecked());self.middle.toggled.connect(lambda checked:self.row.setEnabled(not checked));form.addRow('Detector Row',self.row)
        self.global_range=QCheckBox('Global');self.global_range.setChecked(settings['range_low'] is None);form.addRow('Initial Search Range',self.global_range)
        self.low=number(settings['range_low']);self.high=number(settings['range_high'] or 2047)
        for label,w in [('Range Start',self.low),('Range End',self.high)]:
            form.addRow(label,w);w.setEnabled(not self.global_range.isChecked());self.global_range.toggled.connect(lambda checked,w=w:w.setEnabled(not checked))
        self.auto_step=QCheckBox('Auto');self.auto_step.setChecked(settings['initial_step'] is None);form.addRow('Initial Step',self.auto_step)
        self.step=number(settings['initial_step'] or 20,.25);self.step.setEnabled(not self.auto_step.isChecked());self.auto_step.toggled.connect(lambda checked:self.step.setEnabled(not checked));form.addRow('Initial Step (px)',self.step)
        self.target=combo([('0.25 px',.25),('0.5 px',.5),('1 px',1.)],settings['target_step']);form.addRow('Target Step',self.target)
        self.search_method=combo([('FourierRec','fourierrec'),('LineRec','linerec')],settings['search_method']);form.addRow('Search Reconstruction Method',self.search_method)
        score=QLineEdit('Confidence Weighted');score.setReadOnly(True);form.addRow('Scoring Method',score)
        form=page('SAM3')
        self.checkpoint,row=path_row('checkpoint');form.addRow('Checkpoint',row)
        self.confidence=number(settings['confidence'],.2,.79,.05);form.addRow('Minimum Confidence',self.confidence)
        self.nms=number(settings['nms_iou'],0,1,.05);form.addRow('NMS IoU',self.nms)
        self.repo,row=path_row('repo',True);form.addRow('SAM3 Repository',row)
        self.python,row=path_row('sam3_python');form.addRow('SAM3 Python',row)
        self.recon_python,row=path_row('recon_python');form.addRow('Reconstruction Python',row)
        self.refresh=integer(settings['refresh_seconds'],60);self.refresh.setMinimum(1);form.addRow('Preview Refresh (s)',self.refresh)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)

    def result_settings(self):
        result=dict(self.values)
        result.update(output_root=self.output_root.text().strip() if self.output_mode.currentData()=='custom' else '',
            method=self.method.currentData(),gpu=self.gpu.value(),binning=self.binning.currentData(),chunk_size=self.chunk.value(),iterations=self.iterations.value(),
            start_row=None if self.all_rows.isChecked() else self.start.value(),end_row=None if self.all_rows.isChecked() else self.end.value(),
            search_row=None if self.middle.isChecked() else self.row.value(),range_low=None if self.global_range.isChecked() else self.low.value(),
            range_high=None if self.global_range.isChecked() else self.high.value(),initial_step=None if self.auto_step.isChecked() else self.step.value(),
            target_step=self.target.currentData(),search_method=self.search_method.currentData(),checkpoint=self.checkpoint.text().strip(),
            confidence=self.confidence.value(),nms_iou=self.nms.value(),repo=self.repo.text().strip(),sam3_python=self.python.text().strip(),
            recon_python=self.recon_python.text().strip(),refresh_seconds=self.refresh.value())
        return result
