#import cyth
#from _kernels import normal_kernel1d

import numpy as np
from scipy.special import erf
import _kernels
from scipy.weave import inline

S2PI = np.sqrt(2*np.pi)
S2 = np.sqrt(2)

class normal_kernel1d(object):
    """
    1D normal density kernel with extra integrals for 1D bounded kernel estimation.
    """

    def pdf(self, z, out = None):
        r"""
        Return the probability density of the function. The formula used is:

        .. math::

            \phi(z) = \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}

        :param ndarray xs: Array of any shape
        :returns: an array of shape identical to ``xs``
        """
        return _kernels.norm1d_pdf(np.asarray(z, dtype=np.float64), out)

    def _pdf(self, z, out = None):
        """
        Full-python implementation of :py:func:`normal_kernel1d.pdf`
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
        r"""
        Cumulative density of probability. The formula used is:

        .. math::

            \text{cdf}(z) \triangleq \int_{-\infty}^z \phi(z) dz = \frac{1}{2}\text{erf}\left(\frac{z}{\sqrt{2}}\right) + \frac{1}{2}
        """
        return _kernels.norm1d_cdf(np.asarray(z, dtype=np.float64), out)

    def _cdf(self, z, out=None):
        """
        Full-python implementation of :py:func:`normal_kernel1d.cdf`
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
        r"""
        Partial moment of order 1:

        .. math::

            \text{pm1}(z) \triangleq \int_{-\infty}^z z\phi(z) dz = -\frac{1}{\sqrt{2\pi}}e^{-\frac{z^2}{2}}
        """
        return _kernels.norm1d_pm1(np.asarray(z, dtype=np.float64), out)

    def _pm1(self, z, out=None):
        """
        Full-python implementation of :py:func:`normal_kernel1d.pm1`
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
        r"""
        Partial moment of order 2:

        .. math::

            \text{pm2}(z) \triangleq \int_{-\infty}^z z^2\phi(z) dz = \frac{1}{2}\text{erf}\left(\frac{z}{2}\right) - \frac{z}{\sqrt{2\pi}} e^{-\frac{z^2}{2}} + \frac{1}{2}
        """
        return _kernels.norm1d_pm2(np.asarray(z, dtype=np.float64), out)

    def _pm2(self, z, out=None):
        """
        Full-python implementation of :py:func:`normal_kernel1d.pm2`
        """
        z = np.asarray(z, dtype=float)
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

