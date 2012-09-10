from pyqt_fit import pyqt_fit1d
from PyQt4 import QtGui
import matplotlib

import sys
app = QtGui.QApplication(sys.argv)
matplotlib.interactive(True)
wnd = pyqt_fit1d.main()
sys.exit(app.exec_())

