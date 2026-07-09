#!/usr/bin/env python
from __future__ import division, print_function, absolute_import
from . import functions, residuals, plot_fit, bootstrap
from .compat import user_text, CSV_READ_FLAGS
from .compat import unicode_csv_reader as csv_reader

try:
    from PySide6 import QtCore, QtUiTools, QtWidgets
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QDoubleValidator
    QT_BINDING = "PySide6"
except ImportError:
    try:
        from PySide2 import QtCore, QtUiTools, QtWidgets
        from PySide2.QtCore import Qt
        from PySide2.QtGui import QDoubleValidator
        QT_BINDING = "PySide2"
    except ImportError:
        raise ImportError("PySide6 is required for Python 3; PySide2 is "
                          "supported as the legacy Qt 5 binding")


class _QtGui(object):
    QApplication = QtWidgets.QApplication
    QDialog = QtWidgets.QDialog
    QDialogButtonBox = QtWidgets.QDialogButtonBox
    QFileDialog = QtWidgets.QFileDialog
    QDoubleValidator = QDoubleValidator


QtGui = _QtGui
QMessageBox = QtWidgets.QMessageBox


def pyqtSignature(signature):
    """Translate the legacy string signatures into Qt for Python slots."""
    slot_types = {
        "bool": bool,
        "const QString&": str,
        "QAbstractButton*": QtWidgets.QAbstractButton,
    }
    if signature:
        return QtCore.Slot(slot_types.get(signature, object))
    return QtCore.Slot()


def _qt_enum(scope, name):
    """Read an enum from both Qt 5's and Qt 6's enum layouts."""
    value = getattr(Qt, name, None)
    if value is not None:
        return value
    return getattr(getattr(Qt, scope), name)


DISPLAY_ROLE = _qt_enum("ItemDataRole", "DisplayRole")
EDIT_ROLE = _qt_enum("ItemDataRole", "EditRole")
CHECK_STATE_ROLE = _qt_enum("ItemDataRole", "CheckStateRole")
HORIZONTAL = _qt_enum("Orientation", "Horizontal")
ITEM_IS_ENABLED = _qt_enum("ItemFlag", "ItemIsEnabled")
ITEM_IS_EDITABLE = _qt_enum("ItemFlag", "ItemIsEditable")
ITEM_IS_USER_CHECKABLE = _qt_enum("ItemFlag", "ItemIsUserCheckable")
NO_ITEM_FLAGS = _qt_enum("ItemFlag", "NoItemFlags")


def _dialog_button_role(name):
    role = getattr(QtGui.QDialogButtonBox, name, None)
    if role is not None:
        return role
    return getattr(QtGui.QDialogButtonBox.ButtonRole, name)


RESET_ROLE = _dialog_button_role("ResetRole")
APPLY_ROLE = _dialog_button_role("ApplyRole")


def _load_ui(filename, target):
    """Load a Designer form into an existing dialog using Qt for Python."""
    loader = QtUiTools.QUiLoader()
    form = loader.load(filename, target)
    if form is None:
        raise RuntimeError("Unable to load Qt Designer form: {}".format(filename))

    target.setObjectName(form.objectName())
    target.setGeometry(form.geometry())
    target.setWindowTitle(form.windowTitle())

    for child in form.findChildren(QtCore.QObject):
        name = child.objectName()
        if name:
            setattr(target, str(name), child)

    layout = form.layout()
    if layout is not None:
        form.setLayout(None)
        target.setLayout(layout)

    form.deleteLater()
import matplotlib
from numpy import nan, array, ma, arange
from .curve_fitting import CurveFitting
import os
import sys
from pylab import close as close_figure
import traceback
import re

CIsplitting = re.compile(r'[;, :-]')


def _selected_filename(value):
    """Return a filename from either the Qt4 or Qt5 dialog API."""
    if isinstance(value, tuple):
        return value[0]
    return value


def get_args(*a, **k):
    return a, k


def find(array):
    return arange(len(array))[array]


class ParametersModel(QtCore.QAbstractTableModel):
    def __init__(self, data, function, res, idxX, idxY, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        values = data[:, [idxX, idxY]]
        values = values.data[values.mask.sum(axis=1) == 0]
        self.valuesX = values[:, 0]
        self.valuesY = values[:, 1]
        self.fct = function
        self.res = res
        self.parm_names = function.args
        self.parm_values = list(function.init_args(self.valuesX, self.valuesY))
        self.fixed = [False] * len(function.args)

    def rowCount(self, idx=QtCore.QModelIndex()):
        return len(self.parm_names)

    def columnCount(self, idx=QtCore.QModelIndex()):
        return 3

    def headerData(self, section, orientation, role=DISPLAY_ROLE):
        if role != DISPLAY_ROLE:
            return
        if orientation == HORIZONTAL:
            if section == 0:
                return "Parameter"
            elif section == 1:
                return "Value"
            elif section == 2:
                return "Fixed"

    def flags(self, index):
        if index.column() == 0:
            return ITEM_IS_ENABLED
        elif index.column() == 1:
            return ITEM_IS_ENABLED | ITEM_IS_EDITABLE
        elif index.column() == 2:
            return ITEM_IS_ENABLED | ITEM_IS_USER_CHECKABLE
        return NO_ITEM_FLAGS

    def data(self, index, role=DISPLAY_ROLE):
        r = index.row()
        c = index.column()
        if 0 <= r < len(self.parm_names) and 0 <= c < 3:
            if c == 0:
                if role == DISPLAY_ROLE:
                    return self.parm_names[r]
            elif c == 1:
                if role == DISPLAY_ROLE:
                    return "%g" % (self.parm_values[r],)
                elif role == EDIT_ROLE:
                    return "%g" % (self.parm_values[r],)
            elif c == 2:
                if role == CHECK_STATE_ROLE:
                    return self.fixed[r]

    def setData(self, index, value, role=DISPLAY_ROLE):
        r = index.row()
        c = index.column()
        if 0 <= r < len(self.parm_names) and 0 < c < 3:
            if c == 1 and role == EDIT_ROLE:
                try:
                    f = float(value)
                    self.parm_values[r] = f
                    self.dataChanged.emit(index, index)
                    return True
                except ValueError:
                    print("Error, cannot convert value to double")
            elif c == 2 and role == CHECK_STATE_ROLE:
                self.fixed[r] = value
                self.dataChanged.emit(index, index)
                return True
        return False


class QtFitDlg(QtGui.QDialog):
    def __init__(self, *args, **kwords):
        QtGui.QDialog.__init__(self, *args, **kwords)
        p = os.path.abspath(os.path.join(os.path.dirname(__file__), 'qt_fit.ui'))
        _load_ui(p, self)
        self.function.currentTextChanged.connect(self.on_function_currentIndexChanged)
        self.residuals.currentTextChanged.connect(self.on_residuals_currentIndexChanged)
        self.selectInputFile.clicked.connect(self.on_selectInputFile_clicked)
        self.selectOutputFile.clicked.connect(self.on_selectOutputFile_clicked)
        self.fieldXbox.currentTextChanged.connect(self.on_fieldXbox_currentIndexChanged)
        self.fieldYbox.currentTextChanged.connect(self.on_fieldYbox_currentIndexChanged)
        self.inputFile.textChanged.connect(self.on_inputFile_textChanged)
        self.outputFile.textChanged.connect(self.on_outputFile_textChanged)
        self.writeOutput.toggled.connect(self.on_writeOutput_toggled)
        self.computeCI.toggled.connect(self.on_computeCI_toggled)
        self.CIvalues.textEdited.connect(self.on_CIvalues_textEdited)
        self.CIvalues.editingFinished.connect(self.on_CIvalues_editingFinished)
        self.CImethod.currentTextChanged.connect(self.on_CImethod_currentIndexChanged)
        self.buttonBox.clicked.connect(self.on_buttonBox_clicked)
        self.buttonBox.rejected.connect(self.on_buttonBox_rejected)
        if sys.platform != "darwin":
            self.selectInputFile.setMaximumWidth(32)
            self.selectOutputFile.setMaximumWidth(32)
        self.validator = QtGui.QDoubleValidator()
        self.xMin.setValidator(self.validator)
        self.xMax.setValidator(self.validator)
        self.buttonBox.addButton("Plot", APPLY_ROLE)
        self.buttonBox.addButton("Close Plots", RESET_ROLE)
        self.init()

    def init(self):
        self._fct = None
        self._parameters = None
        self._res = None
        self._data = None
        self._fieldX = None
        self._fieldY = None
        self._input = None
        self._output = None
        self._CI = None
        self._scale = None
        self._header = None
        self._CIchanged = False
        self._write = False
        self.setData(None, None)
        residuals.load()
        functions.load()
        list_fcts = sorted(functions.names())
        list_res = sorted(residuals.names())
        self.function.clear()
        self.function.addItems(list_fcts)
        self.residuals.clear()
        self.residuals.addItems(list_res)
        self.residuals.setCurrentIndex(list_res.index("Standard"))
        self.on_computeCI_toggled(self.computeCI.isChecked())

    @pyqtSignature("const QString&")
    def on_function_currentIndexChanged(self, txt):
        print("New function: {}".format(txt))
        self.fct = functions.get(str(txt))

    @pyqtSignature("const QString&")
    def on_residuals_currentIndexChanged(self, txt):
        print("New residual: {}".format(txt))
        self.res = residuals.get(str(txt))

    @pyqtSignature("")
    def on_selectInputFile_clicked(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, "Open CSV file",
                                                     filter="CSV file (*.csv);;All Files (*.*)")
        filename = _selected_filename(filename)
        if filename:
            self.input = filename

    @pyqtSignature("")
    def on_selectOutputFile_clicked(self):
        filename = QtGui.QFileDialog.getSaveFileName(self, "Save CSV file",
                                                     filter="CSV file (*.csv);;All Files (*.*)")
        filename = _selected_filename(filename)
        if filename:
            self.output = filename

    @pyqtSignature("const QString&")
    def on_fieldXbox_currentIndexChanged(self, txt):
        self.fieldX = txt

    @pyqtSignature("const QString&")
    def on_fieldYbox_currentIndexChanged(self, txt):
        self.fieldY = txt

    def _getFct(self):
        return self._fct

    def _setFct(self, f):
        if f != self._fct:
            self._fct = f
            if self.function.currentText() != f.name:
                self.function.setCurrentIndex(self.function.findText(f.name))
            if self.input:
                self.updateParameters()

    fct = property(_getFct, _setFct)

    def _getRes(self):
        return self._res

    def _setRes(self, res):
        if res != self._res:
            self._res = res
            if self.residuals.currentText() != res.name:
                self.residuals.setCurrentIndex(self.residuals.findText(res.name))
    res = property(_getRes, _setRes)

    @pyqtSignature("const QString&")
    def on_inputFile_textChanged(self, txt):
        self.input = txt

    def _getInput(self):
        return self._input

    def _setInput(self, txt):
        txt = user_text(txt)
        if txt != self._input and os.path.isfile(txt):
            try:
                data = None
                header = None
                with open(txt, CSV_READ_FLAGS) as f:
                    try:
                        r = csv_reader(f)
                        header = next(r)
                        if len(header) < 2:
                            QMessageBox.critical(self, "Error reading CSV file",
                                                 "Error, the file doesn't have at least 2 columns")
                            return
                        data = []
                        for line in r:
                            if not line:
                                break
                            data.append([float(field) if field else nan for field in line])
                        max_length = max(len(l) for l in data)
                        data = array([line + [nan] * (max_length - len(line)) for line in data],
                                     dtype=float)
                        data = ma.masked_invalid(data)
                    except Exception as ex:
                        QMessageBox.critical(self, "Error reading CSV file", str(ex))
                        data = None
                        header = None
                if data is not None:
                    self._input = txt
                    print("input: {}".format(self._input))
                    if self._input != self.inputFile.text():
                        self.inputFile.setText(self._input)
                    self.setData(header, data)
            except IOError:
                pass
    input = property(_getInput, _setInput)

    def setData(self, header, data):
        if header is None or data is None:
            self._header = None
            self._data = None
            self.parameters.setModel(None)
            self.param_model = None
        else:
            self._header = header
            self._data = data
            self.fieldXbox.clear()
            self.fieldXbox.addItems(self._header)
            self.fieldYbox.clear()
            self.fieldYbox.addItems(self._header)
            self.fieldX = self._header[0]
            self.fieldY = self._header[1]

    def _getOutput(self):
        return self._output

    def _setOutput(self, txt):
        txt = user_text(txt)
        if self._output != txt:
            if txt and not txt.endswith(".csv"):
                txt += ".csv"
            self._output = txt
            if self._output != self.outputFile.text():
                self.outputFile.setText(self._output)
    output = property(_getOutput, _setOutput)

    @pyqtSignature("const QString&")
    def on_outputFile_textChanged(self, txt):
        self.output = txt

    def _getWriteResult(self):
        return self._write

    def _setWriteResult(self, on):
        on = bool(on)
        if on != self._write:
            self._write = on
            self.writeOutput.setChecked(on)
    writeResult = property(_getWriteResult, _setWriteResult)

    @pyqtSignature("bool")
    def on_writeOutput_toggled(self, on):
        self.writeResult = on

    def _getHeader(self):
        return self._header
    header = property(_getHeader)

    def _getFieldX(self):
        return self._fieldX

    def _setFieldX(self, txt):
        if txt != self._fieldX and txt in self.header:
            self._fieldX = txt
            if txt != self.fieldXbox.currentText():
                self.fieldXbox.setCurrentIndex(self.fieldXbox.findText(txt))
            self.updateParameters()
    fieldX = property(_getFieldX, _setFieldX)

    def _getFieldY(self):
        return self._fieldY

    def _setFieldY(self, txt):
        if txt != self._fieldY and txt in self.header:
            self._fieldY = txt
            if txt != self.fieldYbox.currentText():
                self.fieldYbox.setCurrentIndex(self.fieldYbox.findText(txt))
            self.updateParameters()
    fieldY = property(_getFieldY, _setFieldY)

    def updateParameters(self):
        if self._data is not None and \
                self.fct is not None and \
                self.res is not None and \
                self.fieldX is not None \
                and self.fieldY is not None:
            idxX = self.header.index(user_text(self.fieldX))
            idxY = self.header.index(user_text(self.fieldY))
            self.param_model = ParametersModel(self._data, self.fct, self.res, idxX, idxY)
            self.parameters.setModel(self.param_model)
            minx = self._data[:, idxX].min()
            maxx = self._data[:, idxX].max()
            self.xMin.setText(str(minx))
            self.xMax.setText(str(maxx))
        #elif self._data is None:
            #print "Missing data"
        #elif self.function is None:
            #print "Missing function"
        #elif self.res is None:
            #print "Missing res"
        #elif self.fieldX is None:
            #print "Missing fieldX"
        #elif self.fieldY is None:
            #print "Missing fieldY"

    @pyqtSignature("bool")
    def on_computeCI_toggled(self, on):
        if on:
            meth = self.CImethod.currentText()
            ints = [float(f) for f in CIsplitting.split(user_text(self.CIvalues.text())) if f]
            self.CI = [meth, ints]
        else:
            self.CI = None

    @pyqtSignature("const QString&")
    def on_CIvalues_textEdited(self, txt):
        self._CIchanged = True

    @pyqtSignature("")
    def on_CIvalues_editingFinished(self):
        if self.CI:
            try:
                ints = [float(f) for f in CIsplitting.split(user_text(self.CIvalues.text())) if f]
                self.setIntervals(ints)
            except:
                pass
            if self.CI[1]:
                self.CIvalues.setText(";".join("{:g}".format(f) for f in self.CI[1]))
            else:
                self.CIvalues.setText("")
            self._CIchanged = False

    @pyqtSignature("const QString&")
    def on_CImethod_currentIndexChanged(self, txt):
        if self.CI:
            meth = user_text(txt)
            self.setCIMethod(meth)

    def _getCI(self):
        return self._CI

    def _setCI(self, val):
        if val is not None:
            val = (user_text(val[0]), [float(f) for f in val[1]])
        if val != self._CI:
            self._CI = val
            if val is not None:
                meth, ints = val
                if meth != self.CImethod.currentText():
                    self.CImethod.setCurrentIndex(self.CImethod.findText(meth))
                self.CIvalues.setText(";".join("{:g}".format(f) for f in ints))
    CI = property(_getCI, _setCI)

    def setCIMethod(self, meth):
        if meth != self._CI[0]:
            self._CI = (meth, self._CI[1])
            if meth != self.CImethod.currentText():
                self.CImethod.setCurrentIndex(self.CImethod.findText(meth))

    def setIntervals(self, ints):
        if ints != self._CI[1]:
            self._CI = (self._CI[0], ints)
            self.CIvalues.setText(";".join("{:g}".format(f) for f in ints))

    @pyqtSignature("QAbstractButton*")
    def on_buttonBox_clicked(self, button):
        role = self.buttonBox.buttonRole(button)
        if role == RESET_ROLE:
            close_figure('all')
        elif role == APPLY_ROLE:
            self.plot()

    @pyqtSignature("")
    def on_buttonBox_rejected(self):
        close_figure('all')

    def plot(self):
        if self.param_model is None:
            QMessageBox.critical(self, "Error plotting", "Error, you don't have any data loaded")
        else:
            if self._CIchanged:
                self.on_CIvalues_editingFinished()
            fct = self.fct
            res = self.res
            model = self.param_model
            xdata = model.valuesX
            ydata = model.valuesY
            p0 = model.parm_values
            parm_names = model.parm_names
            eval_points = None
            fixed = tuple(find(array(model.fixed) > 0))
            if self.interpolate.isChecked():
                if self.autoScale.isChecked():
                    xmin = xdata.min()
                    xmax = xdata.max()
                else:
                    xmin = float(self.xMin.text())
                    xmax = float(self.xMax.text())
                eval_points = arange(xmin, xmax, (xmax - xmin) / 1024)
            CImethod = None
            CImethodName = user_text(self.CImethod.currentText())
            if CImethodName == u"Bootstrapping":
                CImethod = bootstrap.bootstrap_regression
            elif CImethodName == u"Residual resampling":
                CImethod = bootstrap.bootstrap_residuals
            outfile = self.output
            CI = ()
            result = None
            loc = str(self.legendLocation.currentText())
            fct_desc = "$%s$" % (fct.description,)
            try:
                cf_kwrds = dict(residuals=res.__call__,
                                p0=p0,
                                function=fct,
                                maxfev=10000,
                                fix_params=fixed,
                                Dfun=fct.Dfun,
                                Dres=res.Dfun,
                                col_deriv=True)
                if self.CI is not None:
                    CI = self.CI[1]
                    bs = bootstrap.bootstrap(CurveFitting, xdata, ydata, CI,
                                             shuffle_method=CImethod,
                                             shuffle_kwrds={"add_residual": res.invert,
                                                            "fit": CurveFitting},
                                             extra_attrs=('popt',), eval_points=eval_points,
                                             fit_kwrds=cf_kwrds)
                    result = plot_fit.fit_evaluation(bs.y_fit, xdata, ydata,
                                                     eval_points=eval_points, xname=self.fieldX,
                                                     yname=self.fieldY, fct_desc=fct_desc,
                                                     param_names=parm_names, res_name=res.name,
                                                     CI=CI, CIresults=bs)
                else:
                    fit = CurveFitting(xdata, ydata, **cf_kwrds)
                    fit.fit()
                    result = plot_fit.fit_evaluation(fit, xdata, ydata, eval_points=eval_points,
                                                     xname=self.fieldX, yname=self.fieldY,
                                                     fct_desc=fct_desc, param_names=parm_names,
                                                     res_name=res.name)
            except Exception as ex:
                traceback.print_exc()
                QMessageBox.critical(self, "Error during Parameters Estimation",
                                     "{0} exception: {1}".format(type(ex).__name__, str(ex)))
                return
            plot_fit.plot1d(result, loc=loc)
            if self.writeResult and outfile:
                #print("output to file '%s'" % (outfile,))
                plot_fit.write1d(outfile, result, res.description, CImethodName)
            #else:
                #print("self.writeResult = %s\noutfile='%s'" % (self.writeResult, outfile))


def main():
    wnd = QtFitDlg()
    wnd.show()
    wnd.raise_()
    return wnd


def run():
    """Launch the graphical 1D fitting application."""
    app = QtGui.QApplication(sys.argv)
    matplotlib.interactive(True)
    main()
    if hasattr(app, "exec"):
        return app.exec()
    return app.exec_()

if __name__ == "__main__":
    run()
