Module :py:mod:`pyqt_fit.kernels`
=================================

.. automodule:: pyqt_fit.kernels


Helper class
------------

This class is provided with default implementations of everything in term of the PDF.

.. autoclass:: Kernel1D
      :members: pdf, cdf, dct, fft, pm1, pm2, convolution

Gaussian Kernels
----------------

.. autoclass:: normal_kernel
    :members: pdf

.. autoclass:: normal_kernel1d
    :members: pdf, cdf, dct, fft, pm1, pm2, convolution

Tricube Kernel
--------------

.. autoclass:: tricube
    :members: pdf, cdf, pm1, pm2

Epanechnikov Kernel
-------------------

.. autoclass:: Epanechnikov
    :members: pdf, cdf, pm1, pm2, convolution

Higher Order Kernels
--------------------

High order kernels are kernel that give up being valid probabilities. We will say a kernel :math:`K_{[n]}` is of order :math:`n` if:

.. math::


    \begin{array}{rcl}
    \int_\R K_{[n]}(x) dx & = & 1 \\
    \forall 1 \leq k < n  \int_\R x^k K_{[n]} dx & = & 0 \\
    \int_\R x^n K_{[n]} dx & \neq & 0
    \end{array}

PyQt-Fit implements two high order kernels.

.. autoclass:: Epanechnikov_order4

.. autoclass:: normal_order4

