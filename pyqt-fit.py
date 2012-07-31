#!/usr/bin/env python
from PyQt4 import QtGui, QtCore, uic
from PyQt4.QtCore import QObject, pyqtSignature, Qt
import matplotlib
matplotlib.use('Qt4Agg')
from numpy import nan, array, ma, isnan, arange, c_
import functions
import residuals
from path import path
from curve_fit import curve_fit
from plot_fit import plot_fit as _plot_fit
from plot_fit import write_fit as _write_fit
import bootstrap
from csv import reader as csv_reader
from csv import writer as csv_writer
import sys
from pylab import close as close_figure
from itertools import izip, chain

def get_args(*a, **k):
    return a,k

def find(array):
    return arange(len(array))[array]

def plot_fit(fct_name, xdata, ydata, p0, fit = curve_fit, eval_points=None,
             CI=(), residuals_name = 'Standard',
             xname="X", yname="Y", 
             args=(), loc=None, **kwrds):
    """
    The parameters are the same as for plot_fit.plot_fit but functions and residuals are given by name
    """
    fct_desc, param_names, fct = functions.get(fct_name)
    if fct is None:
        raise ValueError("Unknown function: '%s'" % fct_name)
    res_desc, invert, res = residuals.get(residuals_name)
    if res is None:
        raise ValueError("Unknown residuals: '%s'" % residuals_name)
    return _plot_fit(fct, xdata, ydata, p0, fit, eval_points, CI, bootstrap,
                     xname, yname, fct_desc, param_names, res_name, args, loc)

class ParametersModel(QtCore.QAbstractTableModel):
    def __init__(self, data, function, res, idxX, idxY, parent = None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        values = data[:,[idxX, idxY]]
        values = values.data[values.mask.sum(axis=1)==0]
        self.valuesX = values[:,0]
        self.valuesY = values[:,1]
        self.fct = function
        self.res = res
        self.parm_names = function.args
        self.parm_values = list(function.parms(self.valuesX, self.valuesY))
        self.fixed = [False]*len(function.args)

    def rowCount(self, idx = QtCore.QModelIndex()):
        return len(self.parm_names)

    def columnCount(self, idx = QtCore.QModelIndex()):
        return 3

    def headerData(self, section, orientation, role = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return
        if orientation == Qt.Horizontal:
            if section == 0:
                return "Parameter"
            elif section == 1:
                return "Value"
            elif section == 2:
                return "Fixed"

    def flags(self, index):
        if index.column() == 0:
            return Qt.ItemIsEnabled
        elif index.column() == 1:
            return Qt.ItemIsEnabled | Qt.ItemIsEditable
        elif index.column() == 2:
            return Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
        return Qt.NoItemFlags

    def data(self, index, role = Qt.DisplayRole):
        r = index.row()
        c = index.column()
        if 0 <= r < len(self.parm_names) and 0 <= c < 3:
            if c == 0:
                if role == Qt.DisplayRole:
                    return QtCore.QVariant(self.parm_names[r])
            elif c == 1:
                if role == Qt.DisplayRole:
                    return QtCore.QVariant("%g" % (self.parm_values[r],))
                elif role == Qt.EditRole:
                    return QtCore.QVariant("%g" % (self.parm_values[r],))
            elif c == 2:
                if role == Qt.CheckStateRole:
                    return QtCore.QVariant(self.fixed[r])

    def setData(self, index, value, role = Qt.DisplayRole):
        r = index.row()
        c = index.column()
        if 0 <= r < len(self.parm_names) and 0 < c < 3:
            if c == 1 and role == Qt.EditRole:
                f,ok = value.toDouble()
                if ok:
                    self.parm_values[r] = f
                    self.dataChanged.emit(index, index)
                    return True
                else:
                    print "Error, cannot convert value to double"
            elif c == 2 and role == Qt.CheckStateRole:
                self.fixed[r] = value.toPyObject()
                self.dataChanged.emit(index, index)
                return True
        return False


class QtFitDlg(QtGui.QDialog):
    def __init__(self, *args, **kwords):
        QtGui.QDialog.__init__(self, *args, **kwords)
        p = (path(__file__).dirname() / 'qt_fit.ui').abspath()
        uic.loadUi(p, self)
        if sys.platform != "darwin":
            self.selectInputFile.setMaximumWidth(32)
            self.selectOutputFile.setMaximumWidth(32)
        self.validator = QtGui.QDoubleValidator()
        self.xMin.setValidator(self.validator)
        self.xMax.setValidator(self.validator)
        self.buttonBox.addButton("Plot", QtGui.QDialogButtonBox.ApplyRole)
        self.buttonBox.addButton("Close Plots", QtGui.QDialogButtonBox.ResetRole)
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
        print "New function: %s" % txt
        self.fct = functions.get(str(txt))

    @pyqtSignature("const QString&")
    def on_residuals_currentIndexChanged(self, txt):
        print "New residual: %s" % txt
        self.res = residuals.get(str(txt))

    @pyqtSignature("")
    def on_selectInputFile_clicked(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, "Open CSV file", filter="CSV file (*.csv);;All Files (*.*)")
        if filename:
            self.input = filename

    @pyqtSignature("")
    def on_selectOutputFile_clicked(self):
        filename = QtGui.QFileDialog.getSaveFileName(self, "Save CSV file", filter="CSV file (*.csv);;All Files (*.*)")
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

    def _getInput(self):
        return self._input
    def _setInput(self, txt):
        txt = path(txt)
        if txt != self._input:
            try:
                data = None
                header = None
                with file(txt, "rb") as f:
                    r = csv_reader(f)
                    header = r.next()
                    if len(header) < 2:
                        QtGui.QMessageBox.critical(self, "Error reading CSV file", "Error, the file doesn't have at least 2 columns")
                        return
                    data = [[float(f) if f else nan for f in line] for line in r if r]
                    max_length = max(len(l) for l in data)
                    data = array([l + [nan]*(max_length-len(l)) for l in data], dtype=float)
                    data = ma.masked_invalid(data)
                    header = header
                self._input = txt
                if self._input != self.inputFile.text():
                    self.inputFile.setText(self._input)
                self.setData(header, data)
            except Exception, ex:
                QtGui.QMessageBox.critical(self, "Error reading CSV file", str(ex))
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
        txt = path(txt)
        if self._output != txt:
            if not txt.endswith(".csv"):
                txt += ".csv"
            self._output = txt
            if self._output != self.outputFile.text():
                self.outputFile.setText(self._output)
    output = property(_getOutput, _setOutput)

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
        if self._data is not None and self.fct is not None and self.res is not None and self.fieldX is not None and self.fieldY is not None:
            idxX = self.header.index(self.fieldX)
            idxY = self.header.index(self.fieldY)
            self.param_model = ParametersModel(self._data, self.fct, self.res, idxX, idxY)
            self.parameters.setModel(self.param_model)
            minx = self._data[:,idxX].min()
            maxx = self._data[:,idxX].max()
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
            ints = [float(f) for f in unicode(self.CIvalues.text()).split(";")]
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
                ints = [float(f) for f in unicode(self.CIvalues.text()).split(";")]
                self.setIntervals(ints)
            except:
                pass
            if self.CI[1]:
                self.CIvalues.setText(";".join("%g"%f for f in self.CI[1]))
            else:
                self.CIvalues.setText("")
            self._CIchanged = False

    @pyqtSignature("const QString&")
    def on_CImethod_currentIndexChanged(self, txt):
        if self.CI:
            meth = unicode(txt)
            self.setCIMethod(meth)

    def _getCI(self):
        return self._CI
    def _setCI(self, val):
        if val is not None:
            val = (unicode(val[0]), [float(f) for f in val[1]])
        if val != self._CI:
            self._CI = val
            if val is not None:
                meth, ints = val
                if meth != self.CImethod.currentText():
                    self.CImethod.setCurrentIndex(self.CImethod.findText(meth))
                self.CIvalues.setText(";".join("%g"%f for f in ints))
    CI = property(_getCI, _setCI)

    def setCIMethod(self, meth):
        if meth != self._CI[0]:
            self._CI = (meth, self._CI[1])
            if meth != self.CImethod.currentText():
                self.CImethod.setCurrentIndex(self.CImethod.findText(meth))

    def setIntervals(self, ints):
        if ints != self._CI[1]:
            self._CI = (self._CI[0], ints)
            self.CIvalues.setText(";".join("%g"%f for f in ints))

    @pyqtSignature("QAbstractButton*")
    def on_buttonBox_clicked(self, button):
        role = self.buttonBox.buttonRole(button)
        if role == QtGui.QDialogButtonBox.ResetRole:
            close_figure('all')
        elif role == QtGui.QDialogButtonBox.ApplyRole:
            self.plot()

    @pyqtSignature("")
    def on_buttonBox_rejected(self):
        close_figure('all')

    def plot(self):
        if self.param_model is None:
            QtGui.QMessageBox.critical(self, "Error plotting", "Error, you don't have any data loaded")
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
            repeats = self.repeats.value()
            fixed = tuple(find(array(model.fixed) > 0))
            if self.interpolate.isChecked():
                if self.autoScale.isChecked():
                    xmin = xdata.min()
                    xmax = xdata.max()
                else:
                    xmin = float(self.xMin.text())
                    xmax = float(self.xMax.text())
                eval_points = arange(xmin, xmax, (xmax-xmin)/1024)
            CImethod = None
            if str(self.CImethod.currentText()) == "Bootstrapping":
                CImethod = bootstrap.bootstrap_regression
            elif str(self.CImethod.currentText()) == "Residual resampling":
                CImethod = bootstrap.bootstrap_residuals
            outfile = self.outputFile.text()
            CI = ()
            result = None
            loc = str(self.legendLocation.currentText())
            if self.CI is not None:
                method = self.CI[0]
                CI = self.CI[1]
                result = _plot_fit(fct, xdata, ydata, p0,
                        eval_points=eval_points, CI = CI,
                        xname = self.fieldX, yname = self.fieldY, fct_desc = "$%s$" % (fct.description,),
                        param_names = parm_names, res_name = res.name, repeats=repeats, residuals = res, loc=loc,
                        fit_args={"maxfev": 10000, "fix_params": fixed},
                        shuffle_method=CImethod, shuffle_args={"add_residual": res.invert, "fit":curve_fit})
            else:
                result = _plot_fit(fct, xdata, ydata, p0, fit=curve_fit, fix_params=fixed,
                        eval_points=eval_points,
                        xname = self.fieldX, yname = self.fieldY, fct_desc = fct.description,
                        param_names = parm_names, res_name = res.name, loc=loc,
                        residuals = res, maxfev=10000)
            if outfile:
                _write_fit(outfile, result, res.description, parm_names, self.CI[0] if self.CI is not None else None)

def main():
    wnd = QtFitDlg()
    wnd.show()
    wnd.raise_()
    return wnd

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    matplotlib.interactive(True)
    wnd = main()
    app.exec_()

