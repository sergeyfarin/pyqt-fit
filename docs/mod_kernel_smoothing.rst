Module :py:mod:`pyqt_fit.kernel_smoothing`
==========================================

.. automodule:: pyqt_fit.kernel_smoothing

Kernel Estimation Methods
------------------------

.. autoclass:: SpatialAverage
  :members: evaluate, __call__, set_density_correction, correction, bandwidth, covariance

.. autoclass:: LocalLinearKernel1D
  :members: evaluate, __call__, bandwidth, covariance

.. autoclass:: LocalPolynomialKernel1D
  :members: evaluate, __call__, bandwidth, covariance

.. autoclass:: LocalPolynomialKernel
  :members: evaluate, __call__, bandwidth, covariance

Utility functions
-----------------

.. autofunction:: designMatrix

.. autofunction:: designMatrixSize

