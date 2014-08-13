Module :py:mod:`pyqt_fit.npr_methods`
=====================================

.. automodule:: pyqt_fit.npr_methods

Non-Parametric Regression Methods
---------------------------------

Methods must either inherit or follow the same definition as the
:py:class:`pyqt_fit.npr_methods.RegressionKernelMethod`.

.. autofunction:: compute_bandwidth

.. autoclass:: RegressionKernelMethod

   The following methods are interface methods that should be overriden with ones specific to the implemented method.

   .. automethod:: fit

   .. automethod:: evaluate

Provided methods
----------------

Only extra methods will be described:

.. autoclass:: SpatialAverage

   .. autoattribute:: q

   .. automethod:: correction

   .. automethod:: set_density_correction

.. autoclass:: LocalLinearKernel1D

   .. autoattribute:: q

This class uses the following function:

.. autofunction:: pyqt_fit.py_local_linear.local_linear_1d

.. autoclass:: LocalPolynomialKernel1D

   .. autoattribute:: q

.. autoclass:: LocalPolynomialKernel

   .. autoattribute:: q

.. py:data:: default_method

   Defaut non-parametric regression method.
   :Default: LocalPolynomialKernel(q=1)

Utility functions and classes
-----------------------------

.. autoclass:: PolynomialDesignMatrix1D
  :members: __call__

.. autoclass:: PolynomialDesignMatrix
  :members: __call__

