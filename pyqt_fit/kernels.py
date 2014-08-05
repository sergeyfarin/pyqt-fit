r"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

Module providing a set of kernels for use with either the :py:mod:`pyqt_fit.kde` or the :py:mod:`kernel_smoothing` 
module.

Kernels should be created following this template:

"""
from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.special import erf
from scipy import fftpack, integrate
from .utils import make_ufunc
from . import _kernels_py

from .cyth import HAS_CYTHON

kernels_imp = None


def usePython():
    """
    Force the use of the Python implementation of the kernels
    """
    global kernels_imp
    from .import _kernels_py
    kernels_imp = _kernels_py


def useCython():
    """
    Force the use of the Cython implementation of the kernels, if available
    """
    global kernels_imp
    if HAS_CYTHON:
        from . import _kernels
        kernels_imp = _kernels


if HAS_CYTHON:
    useCython()
else:
    usePython()
    import sys
    print("Warning, cannot import Cython kernel functions, "
          "pure python functions will be used instead", file=sys.stderr)

S2PI = np.sqrt(2 * np.pi)


S2 = np.sqrt(2)

class Kernel1D(object):
    r"""
    A 1D kernel :math:`K(z)` is a function with the following properties:

    .. math::

        \begin{array}{rcl}
        \int_\mathbb{R} K(z) &=& 1 \\
        \int_\mathbb{R} zK(z)dz &=& 0 \\
        \int_\mathbb{R} z^2K(z) dz &<& \infty \quad (\approx 1)
        \end{array}

    Which translates into the function should have:

    - a sum of 1 (i.e. a valid density of probability);
    - an average of 0 (i.e. centered);
    - a finite variance. It is even recommanded that the variance is close to 1 to give a uniform meaning to the 
      bandwidth.

    .. py:attribute:: cut

        :type: float

        Cutting point after which there is a negligeable part of the probability. More formally, if :math:`c` is the 
        cutting point:

        .. math::

            \int_{-c}^c p(x) dx \approx 1

    .. py:attribute:: lower

        :type: float

        Lower bound of the support of the PDF. Formally, if :math:`l` is the lower bound:

        .. math::

            \int_{-\infty}^l p(x)dx = 0

    .. py:attribute:: upper

        :type: float

        Upper bound of the support of the PDF. Formally, if :math:`u` is the upper bound:

        .. math::

            \int_u^\infty p(x)dx = 0

    """
    cut = 3.
    lower = -np.inf
    upper = np.inf

    def pdf(self, z, out=None):
        r"""
        Returns the density of the kernel on the points `z`. This is the funtion :math:`K(z)` itself.

        :param ndarray z: Array of points to evaluate the function on. The method should accept any shape of array.
        :param ndarray out: If provided, it will be of the same shape as `z` and the result should be stored in it.
            Ideally, it should be used for as many intermediate computation as possible.
        """
        raise NotImplementedError()

    def __call__(self, z, out=None):
        """
        Alias for :py:meth:`Kernel1D.pdf`
        """
        return self.pdf(z, out=out)

    def cdf(self, z, out=None):
        r"""
        Returns the cumulative density function on the points `z`, i.e.:

        .. math::

            K_0(z) = \int_{-\infty}^z K(t) dt
        """
        z = np.asfarray(z)
        try:
            comp_pdf = self.__comp_pdf
        except AttributeError:
            def pdf(x):
                return self.pdf(np.atleast_1d(x))
            lower = self.lower
            upper = self.upper
            @make_ufunc()
            def comp_pdf(x):
                if x < lower:
                    return 0
                if x > upper:
                    x = upper
                return integrate.quad(pdf, lower, x)[0]
            self.__comp_cdf = comp_pdf
        if out is None:
            out = np.empty(z.shape, dtype=float)
        return comp_pdf(z, out=out)

    def pm1(self, z, out=None):
        r"""
        Returns the first moment of the density function, i.e.:

        .. math::

            K_1(z) = \int_{-\infty}^z z K(t) dt
        """
        z = np.asfarray(z)
        try:
            comp_pm1 = self.__comp_pm1
        except AttributeError:
            lower = self.lower
            upper = self.upper
            def pm1(x):
                return x * self.pdf(np.atleast_1d(x))
            @make_ufunc()
            def comp_pm1(x):
                if x <= lower:
                    return 0
                if x > upper:
                    x = upper
                return integrate.quad(pm1, lower, x)[0]
            self.__comp_pm1 = comp_pm1
        if out is None:
            out = np.empty(z.shape, dtype=float)
        return comp_pm1(z, out=out)


    def pm2(self, z, out=None):
        r"""
        Returns the second moment of the density function, i.e.:

        .. math::

            K_2(z) = \int_{-\infty}^z z^2 K(t) dt
        """
        z = np.asfarray(z)
        try:
            comp_pm2 = self.__comp_pm2
        except AttributeError:
            lower = self.lower
            upper = self.upper
            def pm2(x):
                return x * x * self.pdf(np.atleast_1d(x))
            @make_ufunc()
            def comp_pm2(x):
                if x <= lower:
                    return 0
                if x > upper:
                    x = upper
                return integrate.quad(pm2, lower, x)[0]
            self.__comp_pm2 = comp_pm2
        if out is None:
            out = np.empty(z.shape, dtype=float)
        return comp_pm2(z, out=out)

    def fft(self, z, out=None):
        """
        FFT of the kernel on the points of ``z``. The points will always be provided as a grid with :math:`2^n` points, 
        representing the whole frequency range to be explored. For convenience, the second half of the points will be 
        provided as negative values.
        """
        z = np.asfarray(z)
        t_star = 2*np.pi/(z[1]-z[0])**2 / len(z)
        dz = t_star * (z[1] - z[0])
        return fftpack.fft(self(z * t_star) * dz).real

    def dct(self, z, out=None):
        r"""
        DCT of the kernel on the points of ``z``. The points will always be provided as a grid with :math:`2^n` points, 
        representing the whole frequency range to be explored.
        """
        z = np.asfarray(z)
        a1 = z[1] - z[0]
        gp = (z / a1 + 0.5) * np.pi / (len(z) * a1)
        return fftpack.dct(self(gp) * (gp[1] - gp[0])).real

class normal_kernel1d(Kernel1D):
    """
    1D normal density kernel with extra integrals for 1D bounded kernel estimation.
    """

    def pdf(self, z, out=None):
        r"""
        Return the probability density of the function. The formula used is:

        .. math::

            \phi(z) = \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}

        :param ndarray xs: Array of any shape
        :returns: an array of shape identical to ``xs``
        """
        return kernels_imp.norm1d_pdf(z, out)

    def _pdf(self, z, out=None):
        """
        Full-python implementation of :py:func:`normal_kernel1d.pdf`
        """
        z = np.asarray(z)
        if out is None:
            out = np.empty(z.shape, dtype=z.dtype)
        np.multiply(z, z, out)
        out *= -0.5
        np.exp(out, out)
        out /= S2PI
        return out

    __call__ = pdf

    def fft(self, z, out=None):
        """
        Returns the FFT of the normal distribution
        """
        z = np.asfarray(z)
        out = np.multiply(z, z, out)
        out *= -0.5
        np.exp(out, out)
        return out

    def dct(self, z, out=None):
        """
        Returns the DCT of the normal distribution
        """
        z = np.asfarray(z)
        out = np.multiply(z, z, out)
        out *= -0.5
        np.exp(out, out)
        return out

    def cdf(self, z, out=None):
        r"""
        Cumulative density of probability. The formula used is:

        .. math::

            \text{cdf}(z) \triangleq \int_{-\infty}^z \phi(z)
                dz = \frac{1}{2}\text{erf}\left(\frac{z}{\sqrt{2}}\right) + \frac{1}{2}
        """
        return kernels_imp.norm1d_cdf(z, out)

    def _cdf(self, z, out=None):
        """
        Full-python implementation of :py:func:`normal_kernel1d.cdf`
        """
        z = np.asarray(z)
        if out is None:
            out = np.empty(z.shape, dtype=z.dtype)
        np.divide(z, S2, out)
        erf(out, out)
        out *= 0.5
        out += 0.5
        return out

    def pm1(self, z, out=None):
        r"""
        Partial moment of order 1:

        .. math::

            \text{pm1}(z) \triangleq \int_{-\infty}^z z\phi(z) dz
                = -\frac{1}{\sqrt{2\pi}}e^{-\frac{z^2}{2}}
        """
        return kernels_imp.norm1d_pm1(z, out)

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

    def pm2(self, z, out=None):
        r"""
        Partial moment of order 2:

        .. math::

            \text{pm2}(z) \triangleq \int_{-\infty}^z z^2\phi(z) dz
                = \frac{1}{2}\text{erf}\left(\frac{z}{2}\right) - \frac{z}{\sqrt{2\pi}}
                e^{-\frac{z^2}{2}} + \frac{1}{2}
        """
        return kernels_imp.norm1d_pm2(z, out)

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
            out[zz] -= sz * np.exp(-0.5 * sz * sz) / S2PI
        elif np.isfinite(z):
            out -= z * np.exp(-0.5 * z * z) / S2PI
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
        self.factor = 1 / np.sqrt(2 * np.pi) ** dim

    def pdf(self, xs):
        """
        Return the probability density of the function.

        :param ndarray xs: Array of shape (D,N) where D is the dimension of the kernel
            and N the number of points.
        :returns: an array of shape (N,) with the density on each point of ``xs``
        """
        xs = np.atleast_2d(xs)
        return self.factor * np.exp(-0.5 * np.sum(xs * xs, axis=0))

    __call__ = pdf


class tricube(Kernel1D):
    r"""
    Return the kernel corresponding to a tri-cube distribution, whose expression is.
    The tri-cube function is given by:

    .. math::

        f_r(x) = \left\{\begin{array}{ll}
                        \left(1-|x|^3\right)^3 & \text{, if } x \in [-1;1]\\
                                0 & \text{, otherwise}
                        \end{array}\right.

    As :math:`f_r` is not a probability and is not of variance 1, we use a normalized function:

    .. math::

        f(x) = a b f_r(ax)

        a = \sqrt{\frac{35}{243}}

        b = \frac{70}{81}

    """

    def pdf(self, z, out=None):
        return kernels_imp.tricube_pdf(z, out)

    __call__ = pdf

    upper = 1. / _kernels_py.tricube_width
    lower = -upper
    cut = upper

    def cdf(self, z, out=None):
        r"""
        CDF of the distribution:

        .. math::

            \text{cdf}(x) = \left\{\begin{array}{ll}
                \frac{1}{162} {\left(60 (ax)^{7} - 7 {\left(2 (ax)^{10} + 15 (ax)^{4}\right)}
                \mathrm{sgn}\left(ax\right) + 140 ax + 81\right)} & \text{, if}x\in[-1/a;1/a]\\
                0 & \text{, if} x < -1/a \\
                1 & \text{, if} x > 1/a
                \end{array}\right.
        """
        return kernels_imp.tricube_cdf(z, out)

    def pm1(self, z, out=None):
        r"""
        Partial moment of order 1:

        .. math::

            \text{pm1}(x) = \left\{\begin{array}{ll}
                \frac{7}{3564a} {\left(165 (ax)^{8} - 8 {\left(5 (ax)^{11} + 33 (ax)^{5}\right)}
                \mathrm{sgn}\left(ax\right) + 220 (ax)^{2} - 81\right)}
                & \text{, if} x\in [-1/a;1/a]\\
                0 & \text{, otherwise}
                \end{array}\right.
        """
        return kernels_imp.tricube_pm1(z, out)

    def pm2(self, z, out=None):
        r"""
        Partial moment of order 2:

        .. math::

            \text{pm2}(x) = \left\{\begin{array}{ll}
            \frac{35}{486a^2} {\left(4 (ax)^{9} + 4 (ax)^{3} - {\left((ax)^{12} + 6 (ax)^{6}\right)}
            \mathrm{sgn}\left(ax\right) + 1\right)} & \text{, if} x\in[-1/a;1/a] \\
            0 & \text{, if } x < -1/a \\
            1 & \text{, if } x > 1/a
            \end{array}\right.
        """
        return kernels_imp.tricube_pm2(z, out)


class Epanechnikov(Kernel1D):
    r"""
    1D Epanechnikov density kernel with extra integrals for 1D bounded kernel estimation.
    """
    def pdf(self, xs, out=None):
        r"""
        The PDF of the kernel is usually given by:

        .. math::

            f_r(x) = \left\{\begin{array}{ll}
                            \frac{3}{4} \left(1-x^2\right) & \text{, if} x \in [-1:1]\\
                                    0 & \text{, otherwise}
                            \end{array}\right.

        As :math:`f_r` is not of variance 1 (and therefore would need adjustments for
        the bandwidth selection), we use a normalized function:

        .. math::

            f(x) = \frac{1}{\sqrt{5}}f\left(\frac{x}{\sqrt{5}}\right)
        """
        return kernels_imp.epanechnikov_pdf(xs, out)
    __call__ = pdf

    upper = 1./_kernels_py.epanechnikov_width
    lower = -upper
    cut = upper

    def cdf(self, xs, out=None):
        r"""
        CDF of the distribution. The CDF is defined on the interval :math:`[-\sqrt{5}:\sqrt{5}]` as:

        .. math::

            \text{cdf}(x) = \left\{\begin{array}{ll}
                    \frac{1}{2} + \frac{3}{4\sqrt{5}} x - \frac{3}{20\sqrt{5}}x^3
                    & \text{, if } x\in[-\sqrt{5}:\sqrt{5}] \\
                    0 & \text{, if } x < -\sqrt{5} \\
                    1 & \text{, if } x > \sqrt{5}
                    \end{array}\right.
        """
        return kernels_imp.epanechnikov_cdf(xs, out)

    def pm1(self, xs, out=None):
        r"""
        First partial moment of the distribution:

        .. math::

            \text{pm1}(x) = \left\{\begin{array}{ll}
                    -\frac{3\sqrt{5}}{16}\left(1-\frac{2}{5}x^2+\frac{1}{25}x^4\right)
                    & \text{, if } x\in[-\sqrt{5}:\sqrt{5}] \\
                    0 & \text{, otherwise}
                    \end{array}\right.
        """
        return kernels_imp.epanechnikov_pm1(xs, out)

    def pm2(self, xs, out=None):
        r"""
        Second partial moment of the distribution:

        .. math::

            \text{pm2}(x) = \left\{\begin{array}{ll}
                    \frac{5}{20}\left(2 + \frac{1}{\sqrt{5}}x^3 - \frac{3}{5^{5/2}}x^5 \right)
                    & \text{, if } x\in[-\sqrt{5}:\sqrt{5}] \\
                    0 & \text{, if } x < -\sqrt{5} \\
                    1 & \text{, if } x > \sqrt{5}
                    \end{array}\right.
        """
        return kernels_imp.epanechnikov_pm2(xs, out)


class Epanechnikov_order4(Kernel1D):
    r"""
    Order 4 Epanechnikov kernel. That is:

    .. math::

        K_{[4]}(x) = \frac{3}{2} K(x) + \frac{1}{2} x K'(x) = -\frac{15}{8}x^2+\frac{9}{8}

    where :math:`K` is the non-normalized Epanechnikov kernel.
    """

    upper = 1
    lower = -upper
    cut = upper

    def pdf(self, xs, out=None):
        return kernels_imp.epanechnikov_o4_pdf(xs, out)
    __call__ = pdf

    def cdf(self, xs, out=None):
        return kernels_imp.epanechnikov_o4_cdf(xs, out)

    def pm1(self, xs, out=None):
        return kernels_imp.epanechnikov_o4_pm1(xs, out)

    def pm2(self, xs, out=None):
        return kernels_imp.epanechnikov_o4_pm2(xs, out)


class normal_order4(Kernel1D):
    r"""
    Order 4 Normal kernel. That is:

    .. math::

        \phi_{[4]}(x) = \frac{3}{2} \phi(x) + \frac{1}{2} x \phi'(x) = \frac{1}{2}(3-x^2)\phi(x)

    where :math:`\phi` is the normal kernel.

    """

    lower = -np.inf
    upper = np.inf
    cut = 3.

    def pdf(self, xs, out=None):
        return kernels_imp.normal_o4_pdf(xs, out)
    __call__ = pdf

    def cdf(self, xs, out=None):
        return kernels_imp.normal_o4_cdf(xs, out)

    def pm1(self, xs, out=None):
        return kernels_imp.normal_o4_pm1(xs, out)

    def pm2(self, xs, out=None):
        return kernels_imp.normal_o4_pm2(xs, out)

kernels1D = [normal_kernel1d, tricube, Epanechnikov, Epanechnikov_order4, normal_order4]
kernelsnD = [normal_kernel]

