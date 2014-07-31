"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

Module implementing kernel-based estimation of density of probability.
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from .kernels import normal_kernel1d
from .utils import namedtuple
from . import kde_methods
from .kde_bandwidth import variance_bandwidth, silverman_covariance, scotts_covariance, botev_bandwidth
from scipy import stats, optimize


def _kde_icdf(kde, points, output=None):
    """
    Compute the inverse cumulative distribution (quantile) function
    """
    xs, ys = kde.cdf_grid()
    coarse_result = np.interp(points, ys, xs, kde.lower, kde.upper)
    def find_inverse(p, approx):
        if approx >= xs[-1] or approx <= xs[0]:
            return approx
        def f(x):
            return kde.cdf(x) - p
        val = f(approx)
        idx = np.searchsorted(xs, approx)
        if val > 0:
            high = approx
            low = xs[idx-1]
        else:
            low = approx
            high = xs[idx]
        return optimize.brentq(f, low, high)
    fct = np.vectorize(find_inverse, [points.dtype])
    return fct(points, coarse_result)


def _kde_icdf_grid(kde, N=None, cut=None):
    """
    Compute the inverse cumulative distribution (quantile) function on a grid.
    """
    if N is None:
        N = 2**10
    xs, ys = kde.cdf_grid(4*N, cut)
    vals = np.linspace(0, 1, N)
    icdf = np.interp(vals, ys, xs, kde.lower, kde.upper)
    return vals, icdf


class KDE1D(object):
    r"""
    Perform a kernel based density estimation in 1D, possibly on a bounded
    domain :math:`[L,U]`.

    :param ndarray data: 1D array with the data points

    Any other named argument will be equivalent to setting the property
    after the fact. For example::

        >>> xs = [1,2,3]
        >>> k = KDE1D(xs, lower=0)

    will be equivalent to::

        >>> k = KDE1D(xs)
        >>> k.lower = 0

    The method rely on an estimator of kernel density given by:

    .. math::

        f(x) \triangleq \frac{1}{hW} \sum_{i=1}^n \frac{w_i}{\lambda_i}
        K\left(\frac{X-x}{h\lambda_i}\right)

        W = \sum_{i=1}^n w_i

    where :math:`h` is the bandwidth of the kernel (:py:attr:`bandwidth`), and :math:`K` is the kernel used for the 
    density estimation (:py:attr:`kernel`) and should follow the requirements set by 
    :py:class:`pyqt_fit.kernels.Kernel`, :math:`w_i` are the weights of the data points (:py:attr:`weights`) and 
    :math:`\lambda_i` are the adaptation factor of the kernel width (:py:attr:`lambdas`).

    If the domain of the density estimation is bounded to the interval
    :math:`[L,U]` (i.e. from :py:attr:`lower` to :py:attr:`upper`), the density
    is then estimated with:

    .. math::

        f(x) \triangleq \frac{1}{hW} \sum_{i=1}^n \frac{w_i}{\lambda_i}
        \hat{K}(x;X,\lambda_i h,L,U)

    Where :math:`\hat{K}` is a modified kernel that depends on the exact method
    used.

    The default methods are implemented in the `kde_methods` module.
    """

    def __init__(self, xdata, **kwords):
        self._xdata = None
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

        self._initialized = False

        for n in kwords:
            setattr(self, n, kwords[n])

        self.xdata = np.atleast_1d(xdata)

        has_bw = (self._bw is not None or self._bw_fct is not None or
                  self._covariance is not None or self._cov_fct is not None)
        if not has_bw:
            self.covariance = scotts_covariance

        if self._method is None:
            self.method = kde_methods.renormalization

        self._initialized = True
        self.update_bandwidth()

    @property
    def initialized(self):
        return self._initialized

    def copy(self):
        """
        Shallow copy of the KDE object
        """
        res = KDE1D.__new__(KDE1D)
        # Copy private members: start with a single '_'
        for m in self.__dict__:
            if len(m) > 1 and m[0] == '_' and m[1] != '_':
                setattr(res, m, getattr(self, m))
        return res

    def update_bandwidth(self):
        """
        Re-compute the bandwidth if it was specified as a function.
        """
        if self._xdata is None or not self.initialized:
            return
        if self._bw_fct:
            _bw = float(self._bw_fct(self._xdata, model=self))
            _cov = _bw * _bw
        elif self._cov_fct:
            _cov = float(self._cov_fct(self._xdata, model=self))
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
        Kernel object. See :py:class:`pyqt_fit.kernels.Kernel` for the requirements on the kernel.

        By default, the kernel is an instance of :py:class:`kernels.normal_kernel1d`
        """
        return self._kernel

    @kernel.setter
    def kernel(self, val):
        self._kernel = val

    @property
    def lower(self):
        r"""
        Lower bound of the density domain. If deleted, becomes set to
        :math:`-\infty`
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
        Upper bound of the density domain. If deleted, becomes set to
        :math:`\infty`
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
        except TypeError:
            ws = np.array(ws, dtype=float)
            self._weights = ws
        self._total_weights = None

    @weights.deleter
    def weights(self):
        self._weights = np.asarray(1.)
        self._total_weights = None

    @property
    def total_weights(self):
        if self._total_weights is None:
            if self._weights.shape:
                assert self._weights.shape == self._xdata.shape, \
                    "There must be as many weights as data points"
                self._total_weights = sum(self._weights)
            else:
                self._total_weights = len(self._xdata)
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
            self._lambdas = ls

    @lambdas.deleter
    def lambdas(self):
        self._lambdas = np.asarray(1.)

    @property
    def bandwidth(self):
        """
        Bandwidth of the kernel.
        Can be set either as a fixed value or using a bandwidth calculator,
        that is a function of signature ``w(xdata)`` that returns a single
        value.

        .. note::

            A ndarray with a single value will be converted to a floating point
            value.
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
            self._covariance = bw * bw

    @property
    def covariance(self):
        """
        Covariance of the gaussian kernel.
        Can be set either as a fixed value or using a bandwidth calculator,
        that is a function of signature ``w(xdata)`` that returns a single
        value.

        .. note::

            A ndarray with a single value will be converted to a floating point
            value.
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

    def evaluate(self, points, output=None):
        """
        Evaluate the kernel on the set of points ``points``
        """
        return self._method(self, points, output)

    def __call__(self, points, output=None):
        """
        This method is an alias for :py:meth:`BoundedKDE1D.evaluate`
        """
        return self.evaluate(points, output=output)

    def cdf(self, points, output=None):
        r"""
        Compute the cumulative distribution function defined as:

        .. math::

            cdf(x) = P(X \leq x) = \int_l^x p(t) dt

        where :math:`l` is the lower bound of the distribution domain and :math:`p` the density of probability.
        """
        return self.method.cdf(self, points, output)

    def cdf_grid(self, N=None, cut=None):
        """
        Compute the cdf from the lower bound to the points given as argument.
        """
        return self.method.cdf_grid(self, N, cut)

    def icdf(self, points, output=None):
        r"""
        Compute the inverse cumulative distribution (quantile) function.
        """
        return _kde_icdf(self, points, output)

    def icdf_grid(self, N=None, cut=None):
        """
        Compute the inverse cumulative distribution (quantile) function on a grid.
        """
        return _kde_icdf_grid(self, N, cut)

    def sf(self, points, output=None):
        r"""
        Compute the survival function.

        The survival function is defined as:

        .. math::

            sf(x) = P(X \geq x) = \int_x^u p(t) dt = 1 - cdf(x)

        where :math:`u` is the upper bound of the distribution domain and :math:`p` the density of probability.

        """
        output = self.cdf(points, output)
        output -= 1
        output *= -1
        return output

    def hazard(self, points, output=None):
        r"""
        Compute the hazard function evaluated on the points.

        The hazard function is defined as:

        .. math::

            h(x) = \frac{p(x)}{sf(x)}
        """
        output = self(points, output=output)
        sf = self.sf(points)
        output /= sf
        return output

    def cumhazard(self, points, output=None):
        r"""
        Compute the cumulative hazard function evaluated on the points.

        The cumulative hazard function is defined as:

        .. math::

            ch(x) = \int_l^x h(t) dt = -\ln sf(x)

        where :math:`l` is the lower bound of the domain, :math:`h` the hazard function and :math:`sf` the survival 
        function.
        """
        output = self.sf(points, output)
        np.log(output, out=output)
        output *= -1
        return output

    @property
    def method(self):
        """
        Select the method to use. Available methods in the :py:mod:`pyqt_fit.kde_methods` sub-module.

        The method is an object that should provide the following:

        ``method(kde, points, output)``
            Evaluate the KDE defined by the ``kde`` object on the ``points``. If ``output`` is provided, it should have 
            the right shape and the result should be written in it.

        ``method.grid(kde, N, cut)``
            Evaluate the KDE defined by the ``kde`` object on a grid. See :py:fct:`pyqt_fit.kde_methods.generate_grid` 
            for a detailed explanation on how the grid is computed.

        ``method.name``
            Return a user-readable name for the method

        ``str(method)``
            Should return the method's name
        """
        return self._method

    @method.setter
    def method(self, m):
        self._method = m

    @property
    def closed(self):
        """
        Returns true if the density domain is closed (i.e. lower and upper
        are both finite)
        """
        return self.lower > -np.inf and self.upper < np.inf

    @property
    def bounded(self):
        """
        Returns true if the density domain is actually bounded
        """
        return self.lower > -np.inf or self.upper < np.inf

    def grid(self, N=None, cut=None):
        """
        Evaluate the density on a grid of N points spanning the whole dataset.

        :returns: a tuple with the mesh on which the density is evaluated and
        the density itself
        """
        return self._method.grid(self, N, cut)

Transform = namedtuple('Tranform', ['__call__', 'inv', 'Dinv'])

LogTransform = Transform(np.log, np.exp, np.exp)
ExpTransform = Transform(np.exp, np.log, lambda x: 1. / x)

def transform_distribution(xs, ys, Dfct, output=None):
    """
    Transform a distribution into another one by a change a variable.

    Given a random variable :math:`X` of distribution :math:`f_X`, the random
    variable :math:`Y = g(X)` has a distribution :math:`f_Y` given by:

    .. math::

        f_Y(y) = \left| \frac{1}{g'(g^{-1}(y))} \right| \cdot f_X(g^{-1}(y))

    """
    return np.multiply(np.abs(1 / Dfct(xs)), ys, output)


def create_transform(obj, inv=None, Dinv=None):
    if isinstance(obj, Transform):
        return obj
    fct = obj.__call__
    if inv is None:
        if not hasattr(obj, 'inv'):
            raise AttributeError("Error, transform object must have a 'inv' "
                                 "attribute or you must specify 'inv'")
        inv = obj.inv if hasattr(obj, 'inv') else inv
    if Dinv is None:
        if hasattr(obj, Dinv):
            Dinv = obj.Dinv
        else:
            def Dinv(x):
                x = np.asfarray(x)
                dx = x * 1e-9
                dx[x == 0] = np.min(dx[x != 0])
                return (inv(x + dx) - inv(x - dx)) / (2 * dx)
    return Transform(fct, inv, Dinv)


class TransformKDE(object):
    r"""
    Compute the Kernel Density Estimate of a dataset, transforming it first to
    a domain where distances are "more meaningful".

    Often, KDE is best estimated in a different domain. This object takes a
    KDE1D object (or one compatible), and a transformation function.

    Given a random variable :math:`X` of distribution :math:`f_X`, the random
    variable :math:`Y = g(X)` has a distribution :math:`f_Y` given by:

    .. math::

        f_Y(y) = \left| \frac{1}{g'(g^{-1}(y))} \right| \cdot f_X(g^{-1}(y))

    In our term, :math:`Y` is the random variable the user is interested in,
    and :math:`X` the random variable we can estimate using the KDE. In this
    case, :math:`g` is the transform from :math:`Y` to :math:`X`.

    So to estimate the distribution on a set of points given in :math:`x`, we
    need a total of three functions:

        - Direct function: transform from the original space to the one in
          which the KDE will be perform (i.e. :math:`g^{-1}: y \mapsto x`)
        - Invert function: transform from the KDE space to the original one
          (i.e. :math:`g: x \mapsto y`)
        - Derivative of the invert function

    If the derivative is not provided, it will be estimated numerically.

    :param kde: KDE evaluation object
    :param trans: Either a simple function, or a function object with
        attributes `inv` and `Dinv` to use in case they are not provided
        as arguments.
    :param inv: Invert of the function. If not provided, `trans` must have
        it as attribute.
    :param Dinv: Derivative of the invert function.

    Any unknown member is forwarded to the underlying KDE object.
    """
    def __init__(self, kde, trans, inv=None, Dinv=None):
        d = self.__dict__
        trans = create_transform(trans, inv, Dinv)
        d['trans'] = trans
        d['kde'] = kde.copy()
        self._xdata = kde.xdata
        self.kde.xdata = trans(kde.xdata)
        self.kde.lower = trans(kde.lower)
        self.kde.upper = trans(kde.upper)
        self.kde.update_bandwidth()

    @property
    def xdata(self):
        """
        Input data
        """
        return self._xdata

    @xdata.setter
    def xdata(self, xs):
        self._xdata = np.atleast_1d(xs)
        self.kde.xdata = self.trans(self._xdata)

    @property
    def kernel(self):
        """
        Kernel used for the KDE estimation. See :py:class:`KDE1D` for details on the requirements of a kernel.
        """
        return self.kde.kernel

    @kernel.setter
    def kernel(self, k):
        self.kde.kernel = k

    @property
    def lower(self):
        """
        Lower bound of the input variable domain
        """
        return self.trans.inv(self.kde.lower)

    @lower.setter
    def lower(self, val):
        self.kde.lower = self.trans(val)

    @lower.deleter
    def lower(self):
        del self.kde.lower

    @property
    def upper(self):
        """
        Upper bound of the input variable domain
        """
        return self.trans.inv(self.kde.upper)

    @upper.setter
    def upper(self, val):
        self.kde.upper = self.trans(val)

    @upper.deleter
    def upper(self):
        del self.kde.upper

    @property
    def weights(self):
        """
        Weigths of the input data points
        """
        return self.kde.weights

    @weights.setter
    def weights(self, vals):
        self.kde.weights = vals

    @weights.deleter
    def weights(self):
        del self.kde.weights

    @property
    def lambdas(self):
        """
        Scaling of the bandwidth, per data point. It can be either a single
        value or an array with one value per data point.

        When deleted, the lamndas are reset to 1.
        """
        return self.kde.lambdas

    @lambdas.setter
    def lambdas(self, vals):
        self.kde.lambdas = vals

    @lambdas.deleter
    def lambdas(self):
        del self.kde.lambdas

    @property
    def bandwidth(self):
        """
        Set the bandwidth in the transformed domain
        """
        return self.kde.bandwidth

    @bandwidth.setter
    def bandwidth(self, val):
        self.kde.bandwidth = val

    @property
    def covariance(self):
        """
        Set the covariance in the transformed domain
        """
        return self.kde.covariance

    @covariance.setter
    def covariance(self, val):
        self.kde.covariance = val

    @property
    def method(self):
        """
        Computation method for the KDE
        """
        return self.kde.method

    @method.setter
    def method(self, m):
        self.kde.method = m

    @property
    def closed(self):
        """
        Check if the domain of the input variable is closed (i.e. bounded on both sides)
        """
        return self.lower > -np.inf and self.upper < np.inf

    @property
    def bounded(self):
        """
        Check if the domain of the input is bounded
        """
        return self.lower > -np.inf or self.upper < np.inf

    def copy(self):
        """
        Creates a shallow copy of the TransformKDE object
        """
        res = TransformKDE.__new__(TransformKDE)
        d = res.__dict__
        d['trans'] = self.trans
        d['kde'] = self.kde
        return res

    def evaluate(self, points, output=None):
        """
        Evaluate the KDE on a set of points
        """
        trans = self.trans
        pts = trans(points)
        output = self.kde(pts, output)
        return transform_distribution(pts, output, trans.Dinv, output)

    def __call__(self, points, output=None):
        """
        Evaluate the KDE on a set of points
        """
        return self.evaluate(points, output)

    def grid(self, N=None):
        """
        Evaluate the KDE on a grid of points with N points.

        The grid is regular *in the transformed domain*, so as to use FFT or
        CDT methods when applicable.
        """
        xs, ys = self.kde.grid(N)
        trans = self.trans
        return trans.inv(xs), transform_distribution(xs, ys, trans.Dinv)

    def cdf(self, points, output=None):
        """
        Evaluate the CDF on a set of points
        """
        pts = self.trans(points)
        return self.kde.cdf(pts, output)

    def cdf_grid(self, N=None, cut=None):
        """
        Implementation of :py:meth:`KDE1D.cdf_grid`
        """
        xs, ys = self.kde.cdf_grid(N, cut)
        return self.trans.inv(xs), ys

    def sf(self, points, output=None):
        """
        Implementation of :py:meth:`KDE1D.sf`
        """
        trans = self.trans
        pts = trans(points)
        return self.kde.sf(pts, output)

    def icdf(self, points, output=None):
        """
        Implementation of :py:meth:`KDE1D.icdf`
        """
        return _kde_icdf(self, points, output)

    def icdf_grid(self, N=None, cut=None):
        """
        Implementation of :py:meth:`KDE1D.icdf_grid`
        """
        return _kde_icdf_grid(self, N, cut)

