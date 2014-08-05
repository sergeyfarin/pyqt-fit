Module :py:mod:`pyqt_fit.kde`
=============================

.. automodule:: pyqt_fit.kde

Kernel Density Estimation Methods
---------------------------------

.. autoclass:: KDE1D
    :members: bandwidth, closed, copy, covariance, evaluate, grid, kernel,
              cdf, cdf_grid, sf, hazard, cumhazard, icdf, icdf_grid,
              lambdas, lower, method, total_weights, fit, upper,
              weights, xdata, __call__

Bandwidth Estimation Methods
----------------------------

.. autofunction:: variance_bandwidth

.. autofunction:: silverman_covariance

.. autofunction:: scotts_covariance

.. autofunction:: botev_bandwidth

