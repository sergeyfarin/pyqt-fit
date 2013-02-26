"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

Module implementing kernel-based estimation of density of probability.
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.special import erf, gamma
from .kernels import normal_kernel1d
from scipy import fftpack, optimize

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

def _botev_fixed_point(t, M, I, a2):
    l=7
    I = np.float128(I)
    M = np.float128(M)
    a2 = np.float128(a2)
    f = 2*np.pi**(2*l)*np.sum(I**l*a2*np.exp(-I*np.pi**2*t))
    for s in range(l, 1, -1):
        K0 = np.prod(np.arange(1, 2*s, 2))/np.sqrt(2*np.pi)
        const = (1 + (1/2)**(s + 1/2))/3
        time=(2*const*K0/M/f)**(2/(3+2*s))
        f=2*np.pi**(2*s)*np.sum(I**s*a2*np.exp(-I*np.pi**2*time))
    return t-(2*M*np.sqrt(np.pi)*f)**(-2/5)

class botev_bandwidth(object):
    """
    Implementation of the KDE bandwidth selection method outline in:

    Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
    estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

    Based on the implementation of Daniel B. Smith, PhD.

    The object is a callable returning the bandwidth for a 1D kernel.
    """
    def __init__(self, lower=None, upper=None, N = None):
        self.lower = lower
        self.upper = upper
        self.N = N

    def __call__(self, data):
        """
        Returns the optimal bandwidth based on the data
        """
        N = 2**10 if self.N is None else int(2**np.ceil(np.log2(self.N)))
        lower = self.lower
        upper = self.upper
        if lower is None or upper is None:
            minimum = np.min(data)
            maximum = np.max(data)
            span = maximum - minimum
            lower = minimum - span / 10 if lower is None else lower
            upper = maximum + span / 10 if upper is None else upper
        # Range of the data
        span = upper - lower

        # Histogram of the data to get a crude approximation of the density
        M = len(data)
        DataHist, bins = np.histogram(data, bins=N, range=(lower, upper))
        DataHist = DataHist / M
        DCTData = fftpack.dct(DataHist, norm=None)

        I = np.arange(1,N,dtype=int)**2
        SqDCTData = (DCTData[1:]/2)**2
        guess = 0.1

        t_star = optimize.brentq(_botev_fixed_point, 0, guess, args=(M, I, SqDCTData))

        return np.sqrt(t_star)*span


class KDE1D(object):
    r"""
    Perform a kernel based density estimation in 1D, possibly on a bounded domain
    :math:`[L,U]`.

    :param ndarray data: 1D array with the data points

    Any other named argument will be equivalent to setting the property after the fact. For example::

        >>> k = KDE1D(xs, lower=0)

    will be equivalent to::

        >>> k = KDE1D(xs)
        >>> k.lower = 0

    The method rely on an estimator of kernel density given by:

    .. math::

        f(x) \triangleq \frac{1}{hW} \sum_{i=1}^n \frac{w_i}{\lambda_i} K\left(\frac{X-x}{h\lambda_i}\right)

        W = \sum_{i=1}^n w_i

    where :math:`h` is the bandwidth of the kernel (:py:attr:`bandwidth`), and :math:`K` is the kernel
    used for the density estimation (:py:attr:`kernel`), :math:`w_i` are the
    weights of the data points (:py:attr:`weights`) and :math:`\lambda_i` are
    the adaptation factor of the kernel width (:py:attr:`lambdas`). :math:`K`
    should be a function such that:

    .. math::

        \begin{array}{rcl}
        \int_\mathbb{R} K(z) &=& 1 \\
        \int_\mathbb{R} zK(z)dz &=& 0 \\
        \int_\mathbb{R} z^2K(z) dz &<& \infty \quad (\approx 1)
        \end{array}

    Which translates into, the function should be of sum 1 (i.e.
    a valid density of probability), of average 0 (i.e. centered) and of finite
    variance. It is even recommanded that the variance is close to 1 to give
    a uniform meaning to the bandwidth.

    If the domain of the density estimation is bounded to the interval
    :math:`[L,U]` (i.e. from :py:attr:`lower` to :py:attr:`upper`), the density
    is then estimated with:

    .. math::

        f(x) \triangleq \frac{1}{hW} \sum_{i=1}^n \frac{w_i}{\lambda_i} \hat{K}(x;X,\lambda_i h,L,U)

    Where :math:`\hat{K}` is a modified kernel that depends on the exact method used.

    To express the various methods, we will refer to the following functions:

    .. math::

        a_0(l,u) = \int_l^u K(z) dz

        a_1(l,u) = \int_l^u zK(z) dz

        a_2(l,u) = \int_l^u z^2K(z) dz


    There are currently five methods available:
        1. unbounded
        2. renormalization
        3. reflexion
        4. near combination
        5. cyclic

    1. Unbounded

        This is the usual method, as described above. The method doesn't need
        to be selected, but is automatically used as soon as the domain is
        unbounded. When computing a grid, this method uses the CDT with 3 times
        the bandwidth as padding to avoid border effects.

    2. Renormalization

        This method consists in using the normal kernel method, but renormalize to
        only take into account the part of the kernel within the domain of the
        density [1]_.

        The kernel is then replaced with:

        .. math::

            \hat{K}(x;X,h,L,U) \triangleq \frac{1}{a_0\left(\frac{L-x}{h},\frac{U-x}{h}\right)} K\left(\frac{x-X}{h}\right)

    3. Reflexion

        This method consist in simulating the reflection of the data left and right of the boundaries.
        If one of the boundary is infinite, then the data is not reflected in that direction. To this
        purpose, the kernel is replaced with:

        .. math::

            \hat{K}(x; X, h, L, U) = K\left(\frac{x-X}{h}\right) + K\left(\frac{x+X-2L}{h}\right) + K\left(\frac{x+X-2U}{h}\right)

        When computing grids, if the bandwidth is constant, the result is computing using CDT.

    4. Linear Combination

        This method uses the linear combination correction published in [1]_.

        The estimation is done with a modified kernel given by:

        .. math::

            K_r(x;X,h,L,U) = \frac{a_2(l,u) - a_1(-u, -l) z}{a_2(l,u)a_0(l,u) - a_1(-u,-l)^2} K(z)

            z = \frac{x-X}{h} \qquad l = \frac{L-x}{h} \qquad u = \frac{U-x}{h}

    5. Cyclic

        This method assumes cyclic boundary conditions and works only for closed boundaries.

        The estimation is done with a modified kernel given by:

        .. math::

            \hat{K}(x; X, h, L, U) = K\left(\frac{x-X}{h}\right) + K\left(\frac{x-X-(U-L)}{h}\right) + K\left(\frac{x-X+(U-L)}{h}\right)

        When computing grids, if the bandwidth is constant, the result is computing using FFT.

    .. [1] Jones, M. C. 1993. Simple boundary correction for kernel density estimation. Statistics and Computing 3: 135--146.

    """

    def __init__(self, xdata, **kwords):
        self._xdata = np.atleast_1d(xdata)
        self._upper = np.inf
        self._lower = -np.inf
        self._kernel = normal_kernel1d()

        self._bw_fct = None
        self._bw = None
        self._cov_fct = None
        self._covariance = None
        self._method = None

        self.weights = 1.
        self.lambdas = 1.

        for n in kwords:
            setattr(self, n, kwords[n])

        if self.covariance is None:
            self.covariance = scotts_bandwidth

        if self._method is None:
            self.method = 'renormalization'

    def update_bandwidth(self):
        """
        Re-compute the bandwidth if it was specified as a function.
        """
        if self._bw_fct:
            _bw = float(self._bw_fct(self._xdata))
            _cov = _bw*_bw
        elif self._cov_fct:
            _cov = float(self._cov_fct(self._xdata))
            _bw = np.sqrt(_cov)
        else:
            return
        self._covariance = _cov
        self._bw = _bw

    @property
    def xdata(self):
        return self._xdata

    @xdata.setter
    def xdata(self, xs):
        self._xdata = np.atleast_1d(xs)
        self.update_bandwidth()

    @property
    def kernel(self):
        r"""
        Kernel object. Should provide the following methods:

        ``kernel.pdf(xs)``
            Density of the kernel, denoted :math:`K(x)`

        ``kernel.cdf(z)``
            Cumulative density of probability, that is :math:`F^K(z) = \int_{-\infty}^z K(x) dx`

        ``kernel.pm1(z)``
            First partial moment, defined by :math:`\mathcal{M}^K_1(z) = \int_{-\infty}^z xK(x)dx`

        ``kernel.pm2(z)``
            Second partial moment, defined by :math:`\mathcal{M}^K_2(z) = \int_{-\infty}^z x^2K(x)dx`

        ``kernel.fft(z)``
            FFT of the kernel on the points of ``z``. The points will always be
            provided as a grid with :math:`2^n` points, representing the whole
            frequency range to be explored. For convenience, the second half of
            the points will be provided as negative values.

        ``kernel.dct(z)``
            DCT of the kernel on the points of ``z``. The points will always be
            provided as a grid with :math:`2^n` points, representing the whole
            frequency range to be explored.

        By default, the kernel is an instance of :py:class:`kernels.normal_kernel1d`
        """
        return self._kernel

    @kernel.setter
    def kernel(self, val):
        self._kernel = val

    @property
    def lower(self):
        r"""
        Lower bound of the density domain. If deleted, becomes set to :math:`-\infty`
        """
        return self._lower

    @lower.setter
    def lower(self, val):
        self._lower = float(val)

    @lower.deleter
    def lower(self):
        self._lower = -np.inf

    @property
    def upper(self):
        r"""
        Upper bound of the density domain. If deleted, becomes set to :math:`\infty`
        """
        return self._upper

    @upper.setter
    def upper(self, val):
        self._upper = float(val)

    @upper.deleter
    def upper(self):
        self._upper = np.inf

    @property
    def weights(self):
        """
        Weigths associated to each data point. It can be either a single value,
        or an array with a value per data point. If a single value is provided,
        the weights will always be set to 1.
        """
        return self._weights

    @weights.setter
    def weights(self, ws):
        try:
            ws = float(ws)
            self._weights = np.asarray(1.)
            self._total_weights = float(self.xdata.shape[0])
        except TypeError:
            ws = np.array(ws, dtype=float)
            ws.shape = self.xdata.shape
            self._total_weights = sum(ws)
            self._weights = ws

    @weights.deleter
    def weights(self):
        self._weights = np.asarray(1.)
        self._total_weights = float(self.xdata.shape[0])

    @property
    def total_weights(self):
        return self._total_weights

    @property
    def lambdas(self):
        """
        Scaling of the bandwidth, per data point. It can be either a single
        value or an array with one value per data point.

        When deleted, the lamndas are reset to 1.
        """
        return self._lambdas

    @lambdas.setter
    def lambdas(self, ls):
        try:
            self._lambdas = np.asarray(float(ls))
        except TypeError:
            ls = np.array(ls, dtype=float)
            ls.shape = self.xdata.shape
            self._lambdas = ls

    @lambdas.deleter
    def lambdas(self):
        self._lambdas = np.asarray(1.)

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
        self._bw_fct = None
        self._cov_fct = None
        if callable(bw):
            self._bw_fct = bw
            self.update_bandwidth()
        else:
            bw = float(bw)
            self._bw = bw
            self._covariance = bw*bw


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
        self._bw_fct = None
        self._cov_fct = None
        if callable(cov):
            self._cov_fct = cov
            self.update_bandwidth()
        else:
            cov = float(cov)
            self._covariance = cov
            self._bw = np.sqrt(cov)

    def evaluate_unbounded(self, points, output=None):
        """
        Method to use if there is, effectively, no bounds
        """
        xdata = self.xdata
        points = np.atleast_1d(points)[:,np.newaxis]

        bw = self.bandwidth * self.lambdas

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

    def evaluate_cyclic(self, points, output=None):
        if not self.closed:
            raise ValueError("Cyclic boundary conditions can only be used with closed domains.")

        xdata = self.xdata
        points = np.atleast_1d(points)[:,np.newaxis]

        bw = self.bandwidth * self.lambdas

        z = (points - xdata) / bw
        L = self.lower
        U = self.upper

        span = U-L

        kernel = self.kernel

        terms = kernel(z)
        terms += kernel(z - (span/bw))
        terms += kernel(z + (span/bw))

        terms *= self.weights / bw
        output = terms.sum(axis=1, out=output)
        output /= self.total_weights

        return output

    def evaluate_linear(self, points, output=None):
        xdata = self.xdata
        points = np.atleast_1d(points)[:,np.newaxis]

        bw = self.bandwidth * self.lambdas

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
        upper *= (self.weights / bw) * kernel(z)

        output = upper.sum(axis=1, out=output)
        output /= self.total_weights

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
            - ``cyclic``

        If the domain is unbounded (i.e. :math:`[-\infty;\infty]`), then
        the value is ``unbounded``.
        """
        if self.bounded:
            return self._method
        return "unbounded"

    @method.setter
    def method(self, m):
        _known_methods = { 'renormalization': self.evaluate_renorm,
                           'reflexion': self.evaluate_reflexion,
                           'linear_combination': self.evaluate_linear,
                           'cyclic': self.evaluate_cyclic}
        _known_grid = { 'renormalization': self.grid_eval,
                        'reflexion': self.grid_reflexion,
                        'linear_combination': self.grid_eval,
                        'cyclic': self.grid_cyclic }
        if m not in _known_methods:
            raise ValueError("Error, method must be one of 'renormalization', 'reflexion', 'cyclic' or 'linear_combination'")
        self._evaluate = _known_methods[m]
        self._grid_eval = _known_grid[m]
        self._method = m

    @property
    def closed(self):
        """
        Returns true if the density domain is closed (i.e. lower and upper are both finite)
        """
        return self.lower > -np.inf and self.upper < np.inf

    @property
    def bounded(self):
        """
        Returns true if the density domain is actually bounded
        """
        return self.lower > -np.inf or self.upper < np.inf


    def grid_eval(self, N = None):
        N = 2**10 if N is None else N
        lower = np.min(xdata) - 2*self.bandwidth if self.lower == -np.inf else self.lower
        upper = np.max(xdata) + 2*self.bandwidth if self.upper ==  np.inf else self.upper
        g = np.r_[lower:upper:N*1j]
        return g, self(g)

    def grid_cyclic(self, N):
        """
        FFT-based estimation of KDE estimation, i.e. with cyclic boundary conditions.
        This works only for closed domains, fixed bandwidth (i.e. lambdas = 1)
        and gaussian kernel.
        """
        if self.lambdas.shape:
            return self.grid_eval(N)
        if not self.closed:
            raise ValueError("Error, cyclic boundary conditions require a closed domain.")
        bw = self.bandwidth * self.lambdas
        data = self.xdata
        N = 2**14 if N is None else N
        lower = self.lower
        upper = self.upper
        R = upper - lower
        dN = 1/N
        mesh = np.r_[lower:upper+dN:(N+2)*1j]
        M = len(data)
        weights = self.weights
        if not weights.shape:
            weights = None
        DataHist, bin_edges = np.histogram(data, bins=mesh - dN/2, weights=weights)
        DataHist[0] += DataHist[-1]
        DataHist = DataHist/M
        FFTData = fftpack.fft(DataHist[:-1])
        if hasattr(self.kernel, 'fft'):
            t_star = (2*bw/R)
            gp = np.roll((np.arange(N)-N/2)*np.pi*t_star, N//2)
            smth = self.kernel.fft(gp)
        else:
            gp = np.roll((np.arange(N)-N/2)*R/N, N//2)
            smth = fftpack.fft(self.kernel(gp/bw) * (gp[1]-gp[0]) / bw)
        SmoothFFTData = FFTData * smth
        density = fftpack.ifft(SmoothFFTData) / (mesh[1]-mesh[0])
        return mesh[:-2], density.real

    def grid_reflexion(self, N = None):
        """
        DCT-based estimation of KDE estimation, i.e. with reflexion boundary
        conditions. This works only for fixed bandwidth (i.e. lambdas = 1) and
        gaussian kernel.

        For open domains, the grid is taken with 3 times the bandwidth as extra
        space to remove the boundary problems.
        """
        if self.lambdas.shape:
            return self.grid_eval(N)

        bw = self.bandwidth * self.lambdas
        data = self.xdata
        N = 2**14 if N is None else N
        lower = np.min(data) - 3*self.bandwidth if self.lower == -np.inf else self.lower
        upper = np.max(data) + 3*self.bandwidth if self.upper ==  np.inf else self.upper

        R = upper - lower

        # Histogram the data to get a crude first approximation of the density
        M = len(data)
        weights = self.weights
        if not weights.shape:
            weights = None
        DataHist, bins = np.histogram(data, bins=N, range=(lower,upper), weights = weights)
        DataHist = DataHist/M
        DCTData = fftpack.dct(DataHist, norm=None)

        if hasattr(self.kernel, 'dct'):
            t_star = bw/R
            gp = np.arange(N)*np.pi*t_star
            smth = self.kernel.dct(gp)
        else:
            gp = (np.arange(N)+0.5)*R/N
            smth = fftpack.dct(self.kernel(gp/bw) * (gp[1]-gp[0]) / bw)

        # Smooth the DCTransformed data using t_star
        SmDCTData = DCTData * smth
        # Inverse DCT to get density
        density = fftpack.idct(SmDCTData, norm=None)/(2*R)
        mesh = np.array([(bins[i]+bins[i+1])/2 for i in xrange(N)])

        return mesh, density


    def grid(self, N = None):
        """
        Evaluate the density on a grid of N points spanning the whole dataset.

        Currently, for cyclic, reflexion and unbouded methods, this used FFT
        and CDT, which are a lots faster on a regular grid. FFT and CDT cannot
        be used if the bandwidth vary depending on the sample though.

        :returns: a tuple with the mesh on which the density is evaluated and the density itself
        """
        if not self.bounded:
            return self.grid_reflexion(N)
        return self._grid_eval(N)

