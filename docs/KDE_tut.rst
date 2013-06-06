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

You may wonder why use KDE rather than a histogram. Let's test the variability of both method. To
that purpose, let first generate a set of a thousand datasets and the corresponding histograms and
KDE, making sure the width of the KDE and the histogram are the same::

  >>> import numpy as np
  >>> from scipy.stats import norm
  >>> from pyqt_fit import kde
  >>> f = norm(loc=0, scale=1)
  >>> xs = np.r_[-3:3:1024j]
  >>> nbins = 20
  >>> x = f.rvs(1000*1000).reshape(1000,1000)
  >>> hs = np.empty((1000, nbins), dtype=float)
  >>> kdes = np.empty((1000, 1024), dtype=float)
  >>> hs[0], edges = np.histogram(x[0], bins=nbins, range=(-3,3), density=True)
  >>> mod = kde.KDE1D(x[0])
  >>> mod.bandwidth = mod.bandwidth  # Prevent future recalculation
  >>> kdes[0] = mod(xs)
  >>> for i in xrange(1, 1000):
  >>>   hs[i] = np.histogram(x[i], bins=nbins, range=(-3,3), density=True)[0]
  >>>   mod.xdata = x[i]
  >>>   kdes[i] = mod(xs)

Now, let's find the mean and the 90% confidence interval::

  >>> h_mean = hs.mean(axis=0)
  >>> h_ci = np.array(np.percentile(hs, (5, 95), axis=0))
  >>> h_err = np.empty(h_ci.shape, dtype=float)
  >>> h_err[0] = h_mean - h_ci[0]
  >>> h_err[1] = h_ci[1] - h_mean
  >>> kde_mean = kdes.mean(axis=0)
  >>> kde_ci = np.array(np.percentile(kdes, (5, 95), axis=0))
  >>> width = edges[1:]-edges[:-1]
  >>> fig = plt.figure()
  >>> ax1 = fig.add_subplot(1,2,1)
  >>> ax1.bar(edges[:-1], h_mean, yerr=h_err, width = width, label='Histogram',
  ...         facecolor='g', edgecolor='k', ecolor='b')
  >>> ax1.plot(xs, f.pdf(xs), 'r--', lw=2, label='$\mathcal{N}(0,1)$')
  >>> ax1.set_xlabel('X')
  >>> ax1.set_xlim(-3,3)
  >>> ax1.legend(loc='best')
  >>> ax2 = fig.add_subplot(1,2,2)
  >>> ax2.fill_between(xs, kde_ci[0], kde_ci[1], color=(0,1,0,.5), edgecolor=(0,.4,0,1))
  >>> ax2.plot(xs, kde_mean, 'k', label='KDE (bw = {:.3g})'.format(mod.bandwidth))
  >>> ax2.plot(xs, f.pdf(xs), 'r--', lw=2, label='$\mathcal{N}(0,1)$')
  >>> ax2.set_xlabel('X')
  >>> ax2.legend(loc='best')
  >>> ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
  >>> ax2.set_ylim(0, ymax)
  >>> ax1.set_ylim(0, ymax)
  >>> ax1.set_title('Histogram, max variation = {:.3g}'.format((h_ci[1] - h_ci[0]).max()))
  >>> ax2.set_title('KDE, max variation = {:.3g}'.format((kde_ci[1] - kde_ci[0]).max()))
  >>> fig.set_title('Comparison Histogram vs. KDE')

.. figure:: KDE_tut_compare.png
   :align: center
   :scale: 50%
   :alt: Comparison Histogram / KDE

   Comparison Histogram / KDE -- KDE has less variability

Note that the KDE doesn't tend toward the true density. Instead, given a kernel :math:`K`,
the mean value will be the convolution of the true density with the kernel. But for that price, we
get a much narrower variation on the values.

Boundary Conditions
-------------------

Confidence Intervals
--------------------

Transformations
---------------

