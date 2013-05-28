Module ``pyqt_fit.plot_fit``
============================

.. automodule:: pyqt_fit.plot_fit

Analyses of the residuals
-------------------------

.. autofunction:: fit_evaluation

.. autofunction:: residual_measures

Plotting the residuals
----------------------

.. autofunction:: plot_dist_residuals

.. autofunction:: plot_residuals

.. autofunction:: scaled_location_plot

.. autofunction:: qqplot

.. autofunction:: plot_residual_tests

General plotting
----------------

.. autofunction:: plot1d

Output to a file
----------------

.. autofunction:: write1d

Return types
------------

Most function return a tuple. For easier access, there are named tuple, i.e.
tuples that can be accessed by name.

.. class:: ResultStruct(...)

  .. note::

    This is a class created with :py:func:`pyqt_fit.utils.namedtuple`.

  .. py:attribute:: fct

    Fitted function (i.e. result of the fitted function)

  .. py:attribute:: fct_desc

    Description of the function being fitted

  .. py:attribute:: param_names

    Name of the parameters fitted

  .. py:attribute:: xdata

    Explaining variables used for fitting

  .. py:attribute:: ydata

    Dependent variables observed during experiment

  .. py:attribute:: xname

    Name of the explaining variables

  .. py:attribute:: yname

    Name of the dependent variabled

  .. py:attribute:: res_name

    Name of the residuals

  .. py:attribute:: residuals

    Function used to compute the residuals

  .. py:attribute:: popt

    Optimal parameters

  .. py:attribute:: res

    Residuals computed with the parameters ``popt``

  .. py:attribute:: yopts

    Evaluation of the optimized function on the observed points

  .. py:attribute:: eval_points

    Points on which the function has been interpolated (may be equal to xdata)

  .. py:attribute:: interpolation

    Interpolated function on ``eval_points`` (may be equal to ``yopt``)

  .. py:attribute:: sorted_yopts

    Evaluated function for each data points, sorted in increasing residual order

  .. py:attribute:: scaled_res

    Scaled residuals, ordered by increasing residuals

  .. py:attribute:: normq

    Expected values for the residuals, based on their quantile

  .. py:attribute:: CI

    List of confidence intervals evaluated (in percent)

  .. py:attribute:: CIs

    List of arrays giving the confidence intervals for the dependent variables and for the parameters.

  .. py:attribute:: CIresults

    Object returned by the confidence interval method

.. class:: ResidualMeasures(scaled_res, res_IX, prob, normq)

  .. note::

    This is a class created with :py:func:`pyqt_fit.utils.namedtuple`.

  .. py:attribute:: scaled_res

    Scaled residuals, sorted

  .. py:attribute:: res_IX

    Sorting indices for the residuals

  .. py:attribute:: prob

    Quantiles of the scaled residuals

  .. py:attribute:: normq

    Expected values of the quantiles for a normal distribution

.. class:: ResTestResult(res_figure, residuals, scaled_residuals, qqplot, dist_residuals)

  .. note::

    This is a class created with :py:func:`pyqt_fit.utils.namedtuple`.

  .. py:attribute:: res_figure

    Handle to the figure

  .. py:attribute:: residuals

    Handles created by :py:func:`plot_residuals`

  .. py:attribute:: scaled_residuals

    Handles created by :py:func:`scaled_location_plot`

  .. py:attribute:: qqplot

    Handles created by :py:func:`qqplot`

  .. py:attribute:: dist_residuals

    Handles created by :py:func:`plot_dist_residuals`


.. class:: Plot1dResult(figure, estimate, data, CIs, \*ResTestResult)

  .. note::

    This is a class create with :py:func:`pyqt_fit.utils.namedtuple`. Also, it
    contains all the first of :py:class:`ResTestResult` at the end of the
    tuple.

  .. py:attribute:: figure

    Handle to the figure with the data and fitted curve

  .. py:attribute:: estimate

    Handle to the fitted curve

  .. py:attribute:: data

    Handle to the data

  .. py:attribute:: CIs

    Handles to the confidence interval curves


