from PySide6.QtWidgets import QWidget,QFormLayout,QComboBox,QSpinBox,QDoubleSpinBox,QCheckBox,QDialog,QVBoxLayout,QDialogButtonBox
from .preprocessing import values
class PreprocessingWidget(QWidget):
    def __init__(self,settings,parent=None):
        super().__init__(parent);v=values(settings);form=QFormLayout(self);form.setVerticalSpacing(12)
        self.stripe=QComboBox()
        for label,key in [('Vo-all','vo-all'),('None','none'),('Fourier-wavelet','fw'),('Titarenko','ti')]:self.stripe.addItem(label,key)
        self.stripe.setCurrentIndex(max(0,self.stripe.findData(v['remove_stripe_method'])));form.addRow('Remove stripes',self.stripe)
        self.dezinger=QSpinBox();self.dezinger.setRange(0,99);self.dezinger.setSpecialValueText('Off');self.dezinger.setValue(v['dezinger']);form.addRow('Dezinger size',self.dezinger)
        self.threshold=QSpinBox();self.threshold.setRange(0,1000000000);self.threshold.setValue(int(v['dezinger_threshold']));form.addRow('Dezinger threshold',self.threshold)
        self.dezinger.valueChanged.connect(lambda n:self.threshold.setEnabled(n>0));self.threshold.setEnabled(self.dezinger.value()>0)
        self.filter=QComboBox()
        for key in ['parzen','ramp','shepp','hann','hamming','cosine','cosine2','none']:self.filter.addItem(key,key)
        self.filter.setCurrentIndex(max(0,self.filter.findData(v['filter_name'])));form.addRow('FBP filter',self.filter)
        self.ratio=QDoubleSpinBox();self.ratio.setRange(.000001,1e6);self.ratio.setDecimals(6);self.ratio.setValue(v['bright_ratio']);form.addRow('Bright ratio',self.ratio)
        self.log=QCheckBox();self.log.setChecked(v['minus_log']);form.addRow('Minus-log',self.log)
        self.linear=QCheckBox();self.linear.setChecked(v['flat_linear']);form.addRow('Flat-linear',self.linear)
    def settings(self):return dict(remove_stripe_method=self.stripe.currentData(),dezinger=self.dezinger.value(),dezinger_threshold=self.threshold.value(),filter_name=self.filter.currentData(),bright_ratio=self.ratio.value(),minus_log=self.log.isChecked(),flat_linear=self.linear.isChecked())
class PreprocessingDialog(QDialog):
    def __init__(self,settings,parent=None):
        super().__init__(parent);self.setWindowTitle('Advanced Settings');self.resize(460,370);layout=QVBoxLayout(self);self.editor=PreprocessingWidget(settings,self);layout.addWidget(self.editor)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
