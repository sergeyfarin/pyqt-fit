========
PyQt-Fit
========

PyQt-Fit is a least-square curve fitting in Python with simple GUI and
graphical tools to check your results.

The GUI for 1D data analysis is invoked with:

    $ pyqt_fit1d.py

PyQt-Fit can also be used from the python interpreter. Here is a typical session:

    >>> import pyqt_fit
    >>> import numpy as np
    >>> from matplotlib import pylab
    >>> x = np.arange(0,3,0.01)
    >>> y = 2*x + 4*x**2 + np.random.randn(*x.shape)
    >>> def fct((a0, a1, a2), x):
    ...     return a0 + a1*x + a2*x*x
    >>> result = pyqt_fit.fit(fct, x, y, p0=(0,1,0))
    >>> print result[0] # Display the estimated values
    >>> pyqt_fit.plot1d(result)
    >>> pylab.show()

PyQt-Fit is a package that allows you to define a function defined in a
vector manner, and find the parameters that best fit some data. It also
implement bootstrapping methods (either on the samples or on the residuals) to
estimate confidence intervals on the parameter values and/or the fitted
functions.

The package also provides with four evaluation functions: the plot of residuals
vs. the X axis, the plot of normalized residuals vs. the Y axis, the QQ-plot of
the residuals and the histogram of the residuals. All this can be output to a
CSV file, which should be properly labeled for further analysis in your
favorite software (including most spreadsheet programs).

