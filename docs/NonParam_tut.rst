.. Python-based non-parametric regrssion tutorial

Non-Parametric regression tutorial
==================================

Introduction
------------

In general, given a set of observations :math:`(x_i,y_i)`, with :math:`x_i =
(x_{i1}, \ldots, x_{ip})^T \in \R^p`. We assume there exists a function
:math:`f(x)` such that:

.. math::

  y_i = f(x_i) + \epsilon_i

with :math:`\epsilon_i \in\R` such that :math:`E(\epsilon) = 0`. This function,
however, is not accessible. So we will consider the function :math:`\hat{f}` such that:

.. math::

  \hat{f}(x) = \argmin_f \left( y_i - f(x_i) \right)^2

The various methods presented here consists in numerical approximations finding
the minimum in a part of the function space. The most general method offered by
this module is called the local-polynomial smoother. It uses the
Taylor-decomposition of the function f on each point, and a local weigthing of
the points, to find the values. The function is then defined as:

.. math::

  \hat{f}_n(x) = \argmin_{a_0} \sum_i K\left(\frac{x-x_i}{h}\right) \left(y_i - \mathcal{P}_n(x_i)\right)^2

Where :math:`\mathcal{P}_n` is a polynomial of order :math:`n` whose constant
term is :math:`a_0`, :math:`K` is a kernel used for weighing the values and
:math:`h` is the selected bandwidth. In particular, in 1D:

.. math::

  \hat{f}_n(x) = \argmin_{a_0} \sum_i K\left(\frac{x-x_i}{h}\right) \left(y_i - a_0 - a_1(x-x_i) - \ldots - a_n\frac{(x-x_i)^n}{n!}\right)^2

In general, higher polynomials will reduce the error term but will overfit the
data, in particular at the boundaries.

..  The usual theoretical criterion to
..  estimate how good the fit is is called Mean Integrated Square Error (MISE):
..
..  .. math
..
..    \text{MISE}(\hat{f}) = E\left(\int_{\R^p}\left[\hat{f}(x) - f(x)\right]^2 dx\right)
..
..  where :math:`\hat{f}` is the estimated function and :math:`f` the real function.

A simple example
----------------

For our example, lets first degine our target function::

  >>> import numpy as np
  >>> def f(x):
  ...     return 3*np.cos(x/2) + x**2/5 + 3

Then, we will generate our data::

  >>> xs = np.random.rand(200) * 10
  >>> ys = f(xs) * (1+0.2*np.random.randn(*xs.shape))

We can then visualize the data::

  >>> import matplotlib.pyplot as plt
  >>> grid = np.r_[0:10:512j]
  >>> plt.plot(grid, f(grid), 'r--', label='Reference')
  >>> plt.plot(xs, ys, '+', label='Data')
  >>> plt.legend(loc='best')

.. figure:: NonParam_tut_data.png
  :align: center

  Generated data with generative function.

At first, we will try to use a simple Nadaraya-Watson method, or spatial averaging, using a gaussian kernel::

  >>> import pyqt_fit.kernel_smoothing as smooth
  >>> k1 = smooth.SpatialAverage(xs, ys)
  >>> plt.plot(grid, k1(grid), label="Spatial Averaging")

Confidence Intervals
--------------------

Local-Constant Regression
-------------------------

Local-Linear Regression
-----------------------

