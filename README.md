PyQt-Fit
========

PyQt-Fit is a regression toolbox with plotting and graphical tools for checking
results. It supports parametric regression with user-defined functions and
residuals, as well as non-parametric local-constant and local-polynomial
regression.

The core regression modules do not require Qt. The optional GUI currently
provides an interface for one-dimensional parametric regression.

Running the GUI
---------------

After installing the GUI extra, launch the 1D fitting application with::

    pyqt_fit1d

You can also run it as a module::

    python -m pyqt_fit.pyqt_fit1d
    

Using the library
-----------------

PyQt-Fit can be used directly from Python. Here is a typical session::

    import matplotlib.pyplot as plt
    import numpy as np
    import pyqt_fit
    from pyqt_fit import plot_fit

    x = np.arange(0, 3, 0.01)
    y = 2 * x + 4 * x**2 + np.random.randn(*x.shape)

    def fct(params, x):
        a0, a1, a2 = params
        return a0 + a1 * x + a2 * x * x

    fit = pyqt_fit.CurveFitting(x, y, (0, 1, 0), fct)
    result = plot_fit.fit_evaluation(fit, x, y)
    print(fit(x))
    plot_fit.plot1d(result)
    plt.show()

Features
--------

Parametric regression accepts vectorized user-defined functions. A regular
function wrapped with NumPy's ``vectorize`` function is also supported. The
package includes bootstrap methods for estimating confidence intervals for
parameters and fitted functions.

Non-parametric regression supports local-constant regression in n dimensions
and local-polynomial regression in one dimension. Bootstrapping also works
with the non-parametric methods.

Regression diagnostics include residuals versus the X axis, normalized
residuals versus the Y axis, a QQ plot, and a residual histogram. Results can
also be written to CSV for further analysis.

Installation
------------

PyQt-Fit targets Python 3 and retains a Python 2.7-compatible source path for
legacy applications. Install the core package with::

    python -m pip install .

Install the optional GUI or development dependencies with::

    python -m pip install ".[gui]"       # PySide6, or PySide2 on legacy Python
    python -m pip install ".[gui-qt6]"   # PySide6 explicitly
    python -m pip install ".[gui-qt5]"   # PySide2 legacy Qt 5 path
    python -m pip install ".[test]"      # pytest

Choose one Qt binding per environment. PySide6 is the primary binding for
modern Python 3 installations; PySide2 is the legacy Qt 5 path. The old
``path.py`` dependency is no longer required.

Testing
-------

Install the test dependencies and run pytest with::

    python -m pip install ".[test]"
    python -m pytest

The full numerical suite includes deliberately expensive KDE and bootstrap
tests. The ``Makefile`` also provides ``make test`` and ``make coverage``
targets.

License
-------

PyQt-Fit is distributed under the GNU General Public License version 3 or
later. See ``LICENSE.txt`` for the complete license.

Compatibility note
------------------

Version 1.3.0 introduced interface changes that are not fully compatible with
earlier releases. Existing applications may require small code changes.
