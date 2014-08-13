========
PyQt-Fit
========

PyQt-Fit is a regression toolbox in Python with simple GUI and graphical tools
to check your results. It currently handles regression based on user-defined
functions with user-defined residuals (i.e. parametric regression) or
non-parametric regression, either local-constant or local-polynomial, with the
option to provide your own. There is also a full-GUI access, that currently
provides an interface only to parametric regression.

The GUI for 1D data analysis is invoked with:

    $ pyqt_fit1d.py

PyQt-Fit can also be used from the python interpreter. Here is a typical session:

    >>> import pyqt_fit
    >>> from pyqt_fit import plot_fit
    >>> import numpy as np
    >>> from matplotlib import pylab
    >>> x = np.arange(0,3,0.01)
    >>> y = 2*x + 4*x**2 + np.random.randn(*x.shape)
    >>> def fct(params, x):
    ...     (a0, a1, a2) = params
    ...     return a0 + a1*x + a2*x*x
    >>> fit = pyqt_fit.CurveFitting(x, y, (0,1,0), fct)
    >>> result = plot_fit.fit_evaluation(fit, x, y)
    >>> print(fit(x)) # Display the estimated values
    >>> plot_fit.plot1d(result)
    >>> pylab.show()

PyQt-Fit is a package for regression in Python. There are two set of tools: for
parametric, or non-parametric regression.

For the parametric regression, the user can define its own vectorized function
(note that a normal function wrappred into numpy's "vectorize" function is
perfectly fine here), and find the parameters that best fit some data. It also
provides bootstrapping methods (either on the samples or on the residuals) to
estimate confidence intervals on the parameter values and/or the fitted
functions.

The non-parametric regression can currently be either local constant (i.e.
spatial averaging) in nD or local-polynomial in 1D only. The bootstrapping
function will also work with the non-parametric regression methods.

The package also provides with four evaluation of the regression: the plot of residuals
vs. the X axis, the plot of normalized residuals vs. the Y axis, the QQ-plot of
the residuals and the histogram of the residuals. All this can be output to a
CSV file for further analysis in your favorite software (including most
spreadsheet programs).


Note
----

 Version 1.3.0 is not fully compatible with previous versions. Although
the interfaces offer better flexibility, it will require some code change.
