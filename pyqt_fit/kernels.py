#import cyth
#from _kernels import normal_kernel1d

import numpy as np
from scipy.special import erf
#import _kernels

S2PI = np.sqrt(2*np.pi)
S2 = np.sqrt(2)

class normal_kernel1d(object):
    """
    1D normal density kernel with extra integrals for 1D bounded kernel estimation.
    """

    def pdf(self, z, out = None):
        """
        Return the probability density of the function.

        :param ndarray xs: Array of any shape
        :returns: an array of shape identical to ``xs``
        """
        z = np.asarray(z)
        if out is None:
            out = np.empty(z.shape, dtype = z.dtype)
        np.multiply(z, z, out)
        out *= -0.5
        np.exp(out, out)
        out /= S2PI
        return out

    __call__ = pdf

    def cdf(self, z, out = None):
        """
        Cumulative density of probability. The formula used is:

        .. math::

            cdf(z) = \frac{1}{2}\text{erf}\left(\frac{z}{\sqrt{2}}\right) + \frac{1}{2}
        """
        z = np.asarray(z)
        if out is None:
            out = np.empty(z.shape, dtype = z.dtype)
        np.divide(z, S2, out)
        erf(out, out)
        out *= 0.5
        out += 0.5
        return out

    def pm1(self, z, out = None):
        """
        Partial moment of order 1. That is, a primitive of :math:`zK(z)`. The formula used is:

        .. math::

            pm1(z) = -\frac{1}{\sqrt{2\pi}}e^{-\frac{z^2}{2}}
        """
        z = np.asarray(z)
        if out is None:
            out = np.empty(z.shape, dtype=z.dtype)
        np.multiply(z, z, out)
        out *= -0.5
        np.exp(out, out)
        out /= -S2PI
        return out

    def pm2(self, z, out = None):
        """
        Partial moment of order 1. That is, a primitive of :math:`z^2K(z)`. The formula is:

        .. math::

            pm2(z) = \frac{1}{2}\text{erf}\left(\frac(z}{2}\right) - \frac{z}{\sqrt{2\pi}} e^{-\frac{z^2}{2}} + \frac{1}{2}
        """
        z = np.asarray(z)
        if out is None:
            out = np.empty(z.shape)
        np.divide(z, S2, out)
        erf(out, out)
        out /= 2
        if z.shape:
            zz = np.isfinite(z)
            sz = z[zz]
            out[zz] -= sz*np.exp(-0.5*sz*sz)/S2PI
        elif np.isfinite(z):
            out -= z*np.exp(-0.5*z*z)/S2PI
        out += 0.5
        return out

class normal_kernel(object):
    """
    Returns a function-object for the PDF of a Normal kernel of variance
    identity and average 0 in dimension ``dim``.
    """

    def __new__(klass, dim):
        """
        The __new__ method will automatically select :py:class:`normal_kernel1d` if dim is 1.
        """
        if dim == 1:
            return normal_kernel1d()
        return object.__new__(klass, dim)

    def __init__(self, dim):
        self.factor = 1/np.sqrt(2*np.pi)**dim

    def pdf(self, xs):
        """
        Return the probability density of the function.

        :param ndarray xs: Array of shape (D,N) where D is the dimension of the kernel and N the number of points.
        :returns: an array of shape (N,) with the density on each point of ``xs``
        """
        xs = np.atleast_2d(xs)
        return self.factor*np.exp(-0.5*np.sum(xs*xs, axis=0))

    __call__ = pdf


