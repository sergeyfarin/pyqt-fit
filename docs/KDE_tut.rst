.. Kernel Density Estimation tutorial

Kernel Density Estimation tutorial
==================================

Introduction
------------

Kernel Density Estimation is a method to estimate the frequency of a given value given a random
sample.

Given a set of observations :math:`(x_i)_{1\leq i \leq n}`. We assume the observations are a random
sampling of a probability distribution :math:`f`. We first consider the kernel estimator:

.. math::

  \hat{f}(x) = \frac{1}{Wnh} \sum_{i=1}^n \frac{w_i}{\lambda_i} K\left(\frac{x_i - x}{h\lambda_i}\right)

Where:
  1. :math:`K: \R^p\rightarrow \R` is the kernel, a function centered on 0 and that integrates to 1;
  2. math:`h` is the bandwidth, a smoothing parameter that would typically tend to 0 when the number of samples
     tend to :math:`\infty`;
  3. :math:`(w_i)` are the weights of each of the points, and :math:`W` is the sum of the weigths;
  4. :math:`(\lambda_i)` are the adaptation factor of the kernel.
Also, it is desirable if the second moment of the kernel (i.e. the variance) is 1 for the bandwidth
to keep a uniform meaning across the kernels.

A simple example
----------------

First, let's assume we have a random variable following a normal law :math:`\mathcal{N}(0,1)`::

  >>> import numpy as np
  >>> from scipy.stats import norm
  >>> from matplotlib import pylab as plt
  >>> f = norm(loc=0, scale=1)
  >>> x = f.rvs(500)
  >>> xs = np.r_[-3:3:1024j]
  >>> ys = f.pdf(xs)
  >>> h = plt.hist(x, bins=30, normed=True, label='data')
  >>> plt.plot(xs, ys, 'r--', linewidth=2, label='$\mathcal{N}(0,1)$')
  >>> plt.xlim(-3,3)
  >>> plt.xlabel('X')

We can get estimate the density with the default options with::

  >>> from pyqt_fit import kde
  >>> est = kde.KDE1D(x)
  >>> plot(xs, est(xs), label='Estimate')
  >>> plt.legend(loc='best')

.. figure:: KDE_tut_normal.png
   :align: center

Boundary Conditions
-------------------

Confidence Intervals
--------------------

