Module :py:mod:`pyqt_fit.kde`
=============================

.. automodule:: pyqt_fit.kde

Kernel Density Estimation Methods
---------------------------------

.. autoclass:: KDE1D
    :members: bandwidth, closed, copy, covariance, evaluate, grid, kernel, lambdas, lower, method, total_weights, update_bandwidth, upper, weights, xdata, __call__

Estimation in a Different Domain
````````````````````````````````

.. autoclass:: TransformKDE
  :members: copy, evaluate, grid, __call__

Bandwidth Estimation Methods
----------------------------

.. autofunction:: variance_bandwidth

.. autofunction:: silverman_bandwidth

.. autofunction:: scotts_bandwidth

.. autofunction:: botev_bandwidth

