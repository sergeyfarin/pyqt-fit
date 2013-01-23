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


class KDE1D(object):
    r"""
    Perform a kernel based density estimation in 1D, but on a bounded domain
    :math:`[l,u]`.

    The method rely on an estimator of kernel density given by:

    .. math::

        f(x) \triangleq \frac{1}{hW} \sum_{i=1}^n \frac{w_i}{\lambda_i} K\left(\frac{X-x}{h\lambda_i}\right)

        W = \sum_{i=1}^n w_i

    where :math:`h` is the bandwidth of the kernel, and :math:`K` is the kernel
    used for the density estimation, :math:`w_i` is the weight of a data point
    and :math:`\lambda_i` is adaptation of the kernel width. :math:`K` should
    be a function such that:

    .. math::

        \forall z, K(z) > 0

        \int_\mathbb{R} K(z) = 1

        \int_\mathbb{R} zK(z)dz = 0

        \int_\mathbb{R} z^2K(z) < \infty

    Which translates into, the function should be positive, and of sum 1 (i.e.
    a valid density of probability), of average 0 (i.e. centered) and of finite
    variance. It is even recommanded that the variance is close to 1 to give
    uniform meaning to the bandwidth.

    There is a choice of methods:
        - renormalization
        - reflexion
        - linear combination

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
    :param str method: See :py:func:`BoundedKDE1D.method`


    1. Renormalization

        This method consists in using the normal kernel method, but renormalize to
        only take into account the part of the kernel within the domain of the
        density [1]_.

        The renormalized estimator is then:

        .. math::

            f^n(x) \triangleq \frac{1}{hW} \sum_{i=1}^n \frac{w_i}{a_0(l,u) \lambda_i} K\feft((\frac{x-X_i}{h}\right)

            l = \frac{L-x}{h}

            u = \frac{U-x}{h}

            a_0(l,u) = \int_l^u K(z) dz

    2. Reflexion

        This method consist in simulating the reflection of the data left and right of the boundaries.
        If one of the boundary is infinite, then the data is not reflected in that direction. To this
        purpose, the kernel is replaced with:

        .. math::

            K^r(x; H, h, L, U) = K\left(\frac{x-X}{h}\right) + K\left(\frac{x+X-2L}{h}\right) + K\left(\frc{x+X-2U}{h}\right)

    3. Linear Combination

        This method uses the linear combination correction published in [1]_.

        The estimation is done with:

        .. math::

            f(x) \triangleq \frac{1}{n} \sum_{i=1}^n \frac{1}{h} K_r\left(\frac{x-X_1}{h};l,u\right)

        where :math:`K_r` is a corrected kernel defined by:

        .. math::

            K_r(z;l,u) = \frac{a_2(l,u) - a_1(-u, -l) z}{a_2(l,u)a_0(l,u) - a_1(-u,-l)^2} K(z)

            z = \frac{x-X}{h}

            l = \frac{L-x}{h}

            u = \frac{U-x}{h}

            a_0(l,u) = \int_l^u K(z) dz

            a_1(l,u) = \int_l^u zK(z) dz

            a_2(l,u) = \int_l^u z^2K(z) dz


    .. [1] Jones, M. C. 1993. Simple boundary correction for kernel density estimation. Statistics and Computing 3: 135--146.
    """

    def __init__(self, xdata, lower = -np.inf, upper = np.inf, kernel = None, cov = scotts_bandwidth, **kwords):
        self.xdata = np.atleast_1d(xdata)
        self.n = self.xdata.shape[0]
        self.upper = float(upper)
        self.lower = float(lower)
        if kernel is None:
            kernel = normal_kernel1d()
        self.kernel = kernel

        self._bw = None
        self._covariance = None
        self._method = None
        self._weights = None
        self._lambda = None

        attrs = dict(kwords)
        attrs.setdefault('covariance', scotts_bandwidth)
        attrs.setdefault('method', method)

        for n in attrs:
            setattr(self, n, attrs[n])

    @property
    def weights(self):
        """
        Weigths associated to each data point. Set to ``None`` to remove them (i.e. all weights are 1).
        """
        return self._weights

    @weights.setter
    def weights(self, ws):
        try:
            ws = float(ws)
            self._weights = 1.
            self._total_weights = float(self.xdata.shape[0])
        except TypeError:
            ws = np.array(ws, dtype=float)
            ws.shape = self.xdata.shape
            self._total_weights = sum(ws)
            self._weights = ws

    @weights.deleter
    def weights(self):
        self._weights = 1.
        self._total_weights = float(self.xdata.shape[0])

    @property
    def total_weights(self):
        return self._total_weights

    @property
    def lambdas(self):
        return self._lambda

    @lambdas.setter
    def lambdas(self, ls):
        """
        Scaling of the bandwidth, per data point. Set to ``None`` to remove them (i.e. all scales are 1).
        """
        try:
            self._lambda = float(ls)
        except TypeError:
            ls = np.array(ls, dtype=float)
            ls.shape = self.xdata.shape
            self._lambda = ls

    @lambdas.deleter
    def lambdas(self):
        self._lambda = 1.

    @property
    def bandwidth(self):
        """
        Bandwidth of the kernel.
        Can be set either as a fixed value or using a bandwidth calculator, that is a function
        of signature ``w(xdata)`` that returns a single value.

        .. note::

            A ndarray with a single value will be converted to a floating point value.
        """
        return self._bw

    @bandwidth.setter
    def bandwidth(self, bw):
        if callable(bw):
            _bw = float(bw(self.xdata))
        else:
            _bw = float(bw)
        self._covariance = _bw*_bw
        self._bw = _bw


    @property
    def covariance(self):
        """
        Covariance of the gaussian kernel.
        Can be set either as a fixed value or using a bandwidth calculator, that is a function
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

    def evaluate_unbounded(self, points, output=None):
        """
        Method to use if there is, effectively, no bounds
        """
        xdata = self.xdata
        points = np.atleast_1d(points)[:,np.newaxis]

        bw = self.bandwidth * self.lambdas

        n = self.n
        z = (points - xdata) / bw

        kernel = self.kernel

        terms = kernel(z)

        terms *= self.weights / bw

        output = terms.sum(axis=1, out=output)
        output /= self.total_weights

        return output

    def evaluate_renorm(self, points, output=None):
        xdata = self.xdata
        points = np.atleast_1d(points)[:,np.newaxis]

        bw = self.bandwidth * self.lambdas

        n = self.n
        l = (self.lower - points)/bw
        u = (self.upper - points)/bw
        z = (points - xdata) / bw

        kernel = self.kernel

        a1 = (kernel.cdf(u) - kernel.cdf(l))

        terms = kernel(z) * ((self.weights / bw) / a1)

        output = terms.sum(axis=1, out=output)
        output /= self.total_weights

        return output

    def evaluate_reflexion(self, points, output=None):
        xdata = self.xdata
        points = np.atleast_1d(points)[:,np.newaxis]

        bw = self.bandwidth * self.lambdas

        n = self.n
        z = (points - xdata) / bw
        z1 = (points + xdata) / bw
        L = self.lower
        U = self.upper

        kernel = self.kernel

        terms = kernel(z)

        if L > -np.inf:
            terms += kernel(z1 - (2*L/bw))

        if U < np.inf:
            terms += kernel(z1 - (2*U/bw))

        terms *= self.weights / bw
        output = terms.sum(axis=1, out=output)
        output /= self.total_weights

        return output

    def evaluate_linear(self, points, output=None):
        xdata = self.xdata
        points = np.atleast_1d(points)[:,np.newaxis]

        bw = self.bandwidth * self.lambdas

        n = self.n
        l = (self.lower - points)/bw
        u = (self.upper - points)/bw
        z = (points - xdata)/bw

        kernel = self.kernel

        a0 = kernel.cdf(u) - kernel.cdf(l)
        a1 = kernel.pm1(-l) - kernel.pm1(-u)
        a2 = kernel.pm2(u) - kernel.pm2(l)

        denom = a2*a0 - a1*a1
        upper = a2 - a1*z

        upper /= denom
        upper *= kernel(z)

        upper *= self.weights / self.lambdas

        output = upper.sum(axis=1, out=output)
        output /= n*bw

        return output

    def evaluate(self, points, output=None):
        """
        Evaluate the kernel on the set of points ``points``
        """
        if self.bounded:
            return self._evaluate(points, output=output)
        return self.evaluate_unbounded(points, output=output)

    def __call__(self, points, output=None):
        """
        This method is an alias for :py:meth:`BoundedKDE1D.evaluate`
        """
        return self.evaluate(points, output=output)

    @property
    def method(self):
        """
        Select the method to use. Must be one of:

            - ``renormalization``
            - ``reflexion``
            - ``linear_combination``
        """
        return self._method

    @method.setter
    def method(self, m):
        _known_methods = { 'renormalization': self.evaluate_renorm,
                           'reflexion': self.evaluate_reflexion,
                           'linear_combination': self.evaluate_linear }
        self._evaluate = _known_methods[m]
        self._method = m

    @property
    def bounded(self):
        """
        Returns true if the density domain is actually bounded
        """
        if self.lower >= self.upper:
            raise ValueError("Error, the lower bound must be strictly smaller than the upper bound")
        return self.lower > -np.inf or self.upper < np.inf

