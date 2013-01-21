"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

Module implementing kernel-based estimation of density of probability.
"""

import numpy as np
from scipy.special import erf
import cyth
from kernels import normal_kernel1d

def variance_bandwidth(factor, xdata):
    r"""
    Returns the covariance matrix:

    .. math::

        \mathcal{C} = \tau^2 cov(X)

    where :math:`\tau` is a correcting factor that depends on the method.
    """
    data_covariance = np.atleast_2d(np.cov(xdata, rowvar=1, bias=False))
    sq_bandwidth = data_covariance*factor*factor
    return sq_bandwidth

def silverman_bandwidth(xdata, ydata = None):
    r"""
    The Silverman bandwidth is defined as a variance bandwidth with factor:

    .. math::

        \tau = \left( n \frac{d+2}{4} \right)^\frac{-1}{d+4}
    """
    xdata = np.atleast_2d(xdata)
    d,n = xdata.shape
    return variance_bandwidth(np.power(n*(d+2.)/4., -1./(d+4.)), xdata)

def scotts_bandwidth(xdata, ydata = None):
    r"""
    The Scotts bandwidth is defined as a variance bandwidth with factor:

    .. math::

        \tau = n^\frac{-1}{d+4}
    """
    xdata = np.atleast_2d(xdata)
    d,n = xdata.shape
    return variance_bandwidth(np.power(n, -1./(d+4.)), xdata)


class BoundedKDE1D(object):
    r"""
    Perform a kernel based density estimation in 1D, but on a bounded domain
    :math:`[l,u]` using the linear combination correction published in [1]_.

    The estimation is done with:

    .. math::

        f(x) \triangleq \frac{1}{n} \sum_{i=1}^n \frac{1}{h} K_r\left(\frac{x-X_1}{h};l,u\right)

    where :math:`K_r` is a corrected kernel defined by:

    .. math::

        K_r(z;l,u) = \frac{a_2(l,u) - a_1(-u, -l) z}{a_2(l,u)a_0(l,u) - a_1(-u,-l)^2} K(z)

        a_0(l,u) = \int_l^u K(z) dz

        a_1(l,u) = \int_l^u zK(z) dz

        a_2(l,u) = \int_l^u z^2K(z) dz

    :param ndarray data: 1D array with the data points
    :param float lower: Lower bound of the domain
    :param float upper: Upper bound of the domain
    :param kernel: Kernel object. Should provide the following methods:

        ``kernel.pdf(xs)``
            Density of the kernel

        ``kernel.cdf(z)``
            Cumulative density of probability

        ``kernel.pm1(z)``
            A function whose derivative is :math:`zK(z)`. The name stands for
            'partial moment 1', even though it doesn't need to be the sum from
            :math:`-\infty`.

        ``kernel.pm2(z)``
            A function whose derivative is :math:`z^2K(z)`. The name stands for
            'partial moment 2', even though it doesn't need to be the sum from
            :math:`-\infty`.

    .. [1] Jones, M. C. 1993. Simple boundary correction for kernel density estimation. Statistics and Computing 3: 135--146.
    """

    def __init__(self, xdata, lower = -np.inf, upper = np.inf, kernel = None, cov = scotts_bandwidth):
        self.xdata = np.atleast_1d(xdata)
        self.n = self.xdata.shape[0]
        self.upper = float(upper)
        self.lower = float(lower)
        if kernel is None:
            kernel = normal_kernel1d()
        self.kernel = kernel

        self._bw = None
        self._covariance = None

        self.covariance = cov

    @property
    def bandwidth(self):
        """
        Bandwidth of the kernel.
        """
        return self._bw


    @property
    def covariance(self):
        """
        Covariance of the gaussian kernel.
        Can be set either as a fixed value or using a bandwith calculator, that is a function
        of signature ``w(xdata)`` that returns a single value.

        .. note::

            A ndarray with a single value will be converted to a floating point value.
        """
        return self._covariance

    @covariance.setter
    def covariance(self, cov):
        if callable(cov):
            _cov = float(cov(self.xdata))
        else:
            _cov = float(cov)
        self._covariance = _cov
        self._bw = np.sqrt(_cov)

    def evaluate(self, points, output=None):
        xdata = self.xdata
        points = np.atleast_1d(points)[:,np.newaxis]

        bw = self.bandwidth

        n = self.n
        l = (self.lower - points)/bw
        u = (self.upper - points)/bw

        kernel = self.kernel

        dX = (points - xdata) / bw

        a0 = kernel.cdf(u) - kernel.cdf(l)
        a1 = kernel.pm1(-l) - kernel.pm1(-u)
        a2 = kernel.pm2(u) - kernel.pm2(l)

        denom = a2*a0 - a1*a1
        upper = a2 - a1*dX

        upper /= denom
        upper *= kernel(dX)

        output = upper.sum(axis=1, out=output)
        output /= n*bw

        return output


    def __call__(self, *args, **kwords):
        """
        This method is an alias for :py:meth:`BoundedKDE1D.evaluate`
        """
        return self.evaluate(*args, **kwords)

