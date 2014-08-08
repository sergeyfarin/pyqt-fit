"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

Module implementing non-parametric regressions using kernel smoothing methods.
"""

from __future__ import division, absolute_import, print_function
from scipy import stats
from scipy.linalg import sqrtm, solve
import scipy
import numpy as np
from .compat import irange

from .cyth import HAS_CYTHON

local_linear = None


def useCython():
    """
    Switch to using Cython methods if available
    """
    global local_linear
    if HAS_CYTHON:
        from . import cy_local_linear
        local_linear = cy_local_linear


def usePython():
    """
    Switch to using the python implementation of the methods
    """
    global local_linear
    from . import py_local_linear
    local_linear = py_local_linear

if HAS_CYTHON:
    useCython()
else:
    usePython()

from .kde import scotts_covariance
from .kernels import normal_kernel, normal_kernel1d


class SpatialAverage(object):
    r"""
    Perform a Nadaraya-Watson regression on the data (i.e. also called
    local-constant regression) using a gaussian kernel.

    The Nadaraya-Watson estimate is given by:

    .. math::

        f_n(x) \triangleq \frac{\sum_i K\left(\frac{x-X_i}{h}\right) Y_i}
        {\sum_i K\left(\frac{x-X_i}{h}\right)}

    Where :math:`K(x)` is the kernel and must be such that :math:`E(K(x)) = 0`
    and :math:`h` is the bandwidth of the method.

    :param ndarray xdata: Explaining variables (at most 2D array)
    :param ndarray ydata: Explained variables (should be 1D array)

    :type  cov: ndarray or callable
    :param cov: If an ndarray, it should be a 2D array giving the matrix of
        covariance of the gaussian kernel. Otherwise, it should be a function
        ``cov(xdata, ydata)`` returning the covariance matrix.
    """

    def __init__(self, xdata, ydata, cov=scotts_covariance):
        self.xdata = np.atleast_2d(xdata)
        self.ydata = np.atleast_1d(ydata)

        self._bw = None
        self._covariance = None
        self._inv_cov = None

        self.covariance = cov

        self.d, self.n = self.xdata.shape
        self.correction = 1.

    @property
    def bandwidth(self):
        """
        Bandwidth of the kernel. It cannot be set directly, but rather should
        be set via the covariance attribute.
        """
        if self._bw is None and self._covariance is not None:
            self._bw = np.real(sqrtm(self._covariance))
        return self._bw

    @property
    def covariance(self):
        """
        Covariance of the gaussian kernel.
        Can be set either as a fixed value or using a bandwith calculator,
        that is a function of signature ``w(xdata, ydata)`` that returns
        a 2D matrix for the covariance of the kernel.
        """
        return self._covariance

    @covariance.setter  # noqa
    def covariance(self, cov):
        if callable(cov):
            _cov = np.atleast_2d(cov(self.xdata, self.ydata))
        else:
            _cov = np.atleast_2d(cov)
        self._bw = None
        self._covariance = _cov
        self._inv_cov = scipy.linalg.inv(_cov)

    def evaluate(self, points, result=None):
        """
        Evaluate the spatial averaging on a set of points

        :param ndarray points: Points to evaluate the averaging on
        :param ndarray result: If provided, the result will be put in this
            array
        """
        points = np.atleast_2d(points).astype(self.xdata.dtype)
        #norm = self.kde(points)
        d, m = points.shape
        if result is None:
            result = np.zeros((m,), points.dtype)
        norm = np.zeros((m,), points.dtype)

        # iterate on the internal points
        for i, ci in np.broadcast(irange(self.n),
                                  irange(self._correction.shape[0])):
            diff = np.dot(self._correction[ci],
                          self.xdata[:, i, np.newaxis] - points)
            tdiff = np.dot(self._inv_cov, diff)
            energy = np.exp(-np.sum(diff * tdiff, axis=0) / 2.0)
            result += self.ydata[i] * energy
            norm += energy

        result[norm > 0] /= norm[norm > 0]

        return result

    def __call__(self, *args, **kwords):
        """
        This method is an alias for :py:meth:`SpatialAverage.evaluate`
        """
        return self.evaluate(*args, **kwords)

    @property
    def correction(self):
        """
        The correction coefficient allows to change the width of the kernel
        depending on the point considered. It can be either a constant (to
        correct globaly the kernel width), or a 1D array of same size as the
        input.
        """
        return self._correction

    @correction.setter  # noqa
    def correction(self, value):
        self._correction = np.atleast_1d(value)

    def set_density_correction(self):
        """
        Add a correction coefficient depending on the density of the input
        """
        kde = stats.gaussian_kde(self.xdata)
        dens = kde(self.xdata)
        dm = dens.max()
        dens[dens < 1e-50] = dm
        self._correction = dm / dens


class LocalLinearKernel1D(object):
    r"""
    Perform a local-linear regression using a gaussian kernel.

    The local constant regression is the function that minimises, for each
    position:

    .. math::

        f_n(x) \triangleq \argmin_{a_0\in\mathbb{R}}
            \sum_i K\left(\frac{x-X_i}{h}\right)
            \left(Y_i - a_0 - a_1(x-X_i)\right)^2

    Where :math:`K(x)` is the kernel and must be such that :math:`E(K(x)) = 0`
    and :math:`h` is the bandwidth of the method.

    :param ndarray xdata: Explaining variables (at most 2D array)
    :param ndarray ydata: Explained variables (should be 1D array)

    :type  cov: float or callable
    :param cov: If an float, it should be a variance of the gaussian kernel.
        Otherwise, it should be a function ``cov(xdata, ydata)`` returning the
        variance.

    """
    def __init__(self, xdata, ydata, cov=scotts_covariance):
        self.xdata = np.atleast_1d(xdata)
        self.ydata = np.atleast_1d(ydata)
        self.n = self.xdata.shape[0]

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
        Can be set either as a fixed value or using a bandwith calculator,
        that is a function of signature ``w(xdata, ydata)`` that returns
        a single value.

        .. note::

            A ndarray with a single value will be converted to a floating
            point value.
        """
        return self._covariance

    @covariance.setter  # noqa
    def covariance(self, cov):
        if callable(cov):
            _cov = float(cov(self.xdata, self.ydata))
        else:
            _cov = float(cov)
        self._covariance = _cov
        self._bw = np.sqrt(_cov)

    def evaluate(self, points, out=None):
        """
        Evaluate the spatial averaging on a set of points

        :param ndarray points: Points to evaluate the averaging on
        :param ndarray result: If provided, the result will be put in this
            array
        """
        li2, out = local_linear.local_linear_1d(self._bw, self.xdata,
                                                   self.ydata, points, out)
        self.li2 = li2
        return out

    def __call__(self, *args, **kwords):
        """
        This method is an alias for :py:meth:`LocalLinearKernel1D.evaluate`
        """
        return self.evaluate(*args, **kwords)


class PolynomialDesignMatrix1D(object):
    def __init__(self, dim):
        self.dim = dim
        powers = np.arange(0, dim + 1).reshape((1, dim + 1))
        self.powers = powers

    def __call__(self, dX, out=None):
        return np.power(dX, self.powers, out)  # / self.frac


class LocalPolynomialKernel1D(object):
    r"""
    Perform a local-polynomial regression using a user-provided kernel
    (Gaussian by default).

    The local constant regression is the function that minimises, for each
    position:

    .. math::

        f_n(x) \triangleq \argmin_{a_0\in\mathbb{R}}
            \sum_i K\left(\frac{x-X_i}{h}\right)
            \left(Y_i - a_0 - a_1(x-X_i) - \ldots -
                a_q \frac{(x-X_i)^q}{q!}\right)^2

    Where :math:`K(x)` is the kernel such that :math:`E(K(x)) = 0`, :math:`q`
    is the order of the fitted polynomial  and :math:`h` is the bandwidth of
    the method. It is also recommended to have :math:`\int_\mathbb{R} x^2K(x)dx
    = 1`, (i.e. variance of the kernel is 1) or the effective bandwidth will be
    scaled by the square-root of this integral (i.e. the standard deviation of
    the kernel).

    :param ndarray xdata: Explaining variables (at most 2D array)
    :param ndarray ydata: Explained variables (should be 1D array)
    :param int q: Order of the polynomial to fit. **Default:** 3

    :type  cov: float or callable
    :param cov: If an float, it should be a variance of the gaussian kernel.
        Otherwise, it should be a function ``cov(xdata, ydata)`` returning
        the variance.
        **Default:** ``scotts_covariance``

    """
    def __init__(self, xdata, ydata, q=3, **kwords):
        self.xdata = np.atleast_1d(xdata)
        self.ydata = np.atleast_1d(ydata)
        self.n = self.xdata.shape[0]
        self.q = q

        self._kernel = None
        self._bw = None
        self._covariance = None
        self.designMatrix = None

        for n in kwords:
            setattr(self, n, kwords[n])

        if self.kernel is None:
            self.kernel = normal_kernel1d()
        if self.covariance is None:
            self.covariance = scotts_covariance
        if self.designMatrix is None:
            self.designMatrix = PolynomialDesignMatrix1D

    @property
    def bandwidth(self):
        """
        Bandwidth of the kernel.
        """
        return self._bw

    @bandwidth.setter  # noqa
    def bandwidth(self, bw):
        if callable(bw):
            _bw = float(bw(self.xdata, self.ydata))
        else:
            _bw = float(bw)
        self._bw = _bw
        self._covariance = _bw * _bw

    @property
    def covariance(self):
        """
        Covariance of the gaussian kernel.
        Can be set either as a fixed value or using a bandwith calculator,
        that is a function of signature ``w(xdata, ydata)`` that returns
        a single value.

        .. note::

            A ndarray with a single value will be converted to a floating
            point value.
        """
        return self._covariance

    @covariance.setter  # noqa
    def covariance(self, cov):
        if callable(cov):
            _cov = float(cov(self.xdata, self.ydata))
        else:
            _cov = float(cov)
        self._covariance = _cov
        self._bw = np.sqrt(_cov)

    @property
    def cov(self):
        """
        Covariance of the gaussian kernel.
        Can be set either as a fixed value or using a bandwith calculator,
        that is a function of signature ``w(xdata, ydata)`` that returns
        a single value.

        .. note::

            A ndarray with a single value will be converted to a floating
            point value.
        """
        return self.covariance

    @cov.setter  # noqa
    def cov(self, val):
        self.covariance = val

    @property
    def kernel(self):
        r"""
        Kernel object. Should provide the following methods:

        ``kernel.pdf(xs)``
            Density of the kernel, denoted :math:`K(x)`

        By default, the kernel is an instance of
        :py:class:`kernels.normal_kernel1d`
        """
        return self._kernel

    @kernel.setter  # noqa
    def kernel(self, val):
        self._kernel = val

    def evaluate(self, points, out=None):
        """
        Evaluate the spatial averaging on a set of points

        :param ndarray points: Points to evaluate the averaging on
        :param ndarray result: If provided, the result will be put
            in this array
        """
        xdata = self.xdata[:, np.newaxis]  # make it a column vector
        ydata = self.ydata[:, np.newaxis]  # make it a column vector
        q = self.q
        bw = self.bandwidth
        kernel = self.kernel
        designMatrix = self.designMatrix(q)
        if out is None:
            out = np.empty(points.shape, dtype=float)
        for i, p in enumerate(points):
            dX = (xdata - p)
            Wx = kernel(dX / bw)
            Xx = designMatrix(dX)
            WxXx = Wx * Xx
            XWX = np.dot(Xx.T, WxXx)
            Lx = solve(XWX, WxXx.T)[0]
            out[i] = np.dot(Lx, ydata)
        return out

    def __call__(self, *args, **kwords):
        """
        This method is an alias for :py:meth:`LocalLinearKernel1D.evaluate`
        """
        return self.evaluate(*args, **kwords)


class PolynomialDesignMatrix(object):
    """
    Class used to create a design matrix for polynomial regression
    """
    def __init__(self, dim, deg):
        self.dim = dim
        self.deg = deg

        self._designMatrixSize()

    def _designMatrixSize(self):
        """
        Compute the size of the design matrix for a n-D problem of order d.
        Can also compute the Taylors factors (i.e. the factors that would be
        applied for the taylor decomposition)

        :param int dim: Dimension of the problem
        :param int deg: Degree of the fitting polynomial
        :param bool factors: If true, the out includes the Taylor factors

        :returns: The number of columns in the design matrix and, if required,
            a ndarray with the taylor coefficients for each column of
            the design matrix.
        """
        dim = self.dim
        deg = self.deg
        init = 1
        dims = [0] * (dim + 1)
        cur = init
        prev = 0
        #if factors:
        #    fcts = [1]
        fact = 1
        for i in irange(deg):
            diff = cur - prev
            prev = cur
            old_dims = list(dims)
            fact *= (i + 1)
            for j in irange(dim):
                dp = diff - old_dims[j]
                cur += dp
                dims[j + 1] = dims[j] + dp
        #    if factors:
        #        fcts += [fact]*(cur-prev)
        self.size = cur
        #self.factors = np.array(fcts)

    def __call__(self, x, out=None):
        """
        Creates the design matrix for polynomial fitting using the points x.

        :param ndarray x: Points to create the design matrix.
            Shape must be (D,N) or (N,), where D is the dimension of
            the problem, 1 if not there.

        :param int deg: Degree of the fitting polynomial

        :param ndarray factors: Scaling factor for the columns of the design
            matrix. The shape should be (M,) or (M,1), where M is the number
            of columns of the out. This value can be obtained using
            the :py:func:`designMatrixSize` function.

        :returns: The design matrix as a (M,N) matrix.
        """
        dim, deg = self.dim, self.deg
        #factors = self.factors
        x = np.atleast_2d(x)
        dim = x.shape[0]
        if out is None:
            s = self._designMatrixSize(dim, deg)
            out = np.empty((s, x.shape[1]), dtype=x.dtype)
        dims = [0] * (dim + 1)
        out[0, :] = 1
        cur = 1
        for i in irange(deg):
            old_dims = list(dims)
            prev = cur
            for j in irange(x.shape[0]):
                dims[j] = cur
                for k in irange(old_dims[j], prev):
                    np.multiply(out[k], x[j], out[cur])
                    cur += 1
        #if factors is not None:
        #    factors = np.asarray(factors)
        #    if len(factors.shape) == 1:
        #        factors = factors[:,np.newaxis]
        #    out /= factors
        return out


class LocalPolynomialKernel(object):
    r"""
    Perform a local-polynomial regression in N-D using a user-provided kernel
    (Gaussian by default).

    The local constant regression is the function that minimises,
    for each position:

    .. math::

        f_n(x) \triangleq \argmin_{a_0\in\mathbb{R}}
            \sum_i K\left(\frac{x-X_i}{h}\right)
            \left(Y_i - a_0 - \mathcal{P}_q(X_i-x)\right)^2

    Where :math:`K(x)` is the kernel such that :math:`E(K(x)) = 0`, :math:`q`
    is the order of the fitted polynomial, :math:`\mathcal{P}_q(x)` is a
    polynomial of order :math:`d` in :math:`x` and :math:`h` is the bandwidth
    of the method.

    The polynomial :math:`\mathcal{P}_q(x)` is of the form:

    .. math::

        \mathcal{F}_d(k) = \left\{ \n \in \mathbb{N}^d \middle|
            \sum_{i=1}^d n_i = k \right\}

        \mathcal{P}_q(x_1,\ldots,x_d) = \sum_{k=1}^q
            \sum_{\n\in\mathcal{F}_d(k)} a_{k,\n}
            \prod_{i=1}^d x_i^{n_i}

    For example we have:

    .. math::

        \mathcal{P}_2(x,y) = a_{110} x + a_{101} y + a_{220} x^2 +
            a_{211} xy + a_{202} y^2

    :param ndarray xdata: Explaining variables (at most 2D array).
        The shape should be (N,D) with D the dimension of the problem
        and N the number of points. For 1D array, the shape can be (N,),
        in which case it will be converted to (N,1) array.
    :param ndarray ydata: Explained variables (should be 1D array). The shape
        must be (N,).
    :param int q: Order of the polynomial to fit. **Default:** 3
    :param callable kernel: Kernel to use for the weights. Call is
        ``kernel(points)`` and should return an array of values the same size
        as ``points``. If ``None``, the kernel will be ``normal_kernel(D)``.

    :type  cov: float or callable
    :param cov: If an float, it should be a variance of the gaussian kernel.
        Otherwise, it should be a function ``cov(xdata, ydata)`` returning
        the variance.
        **Default:** ``scotts_covariance``
    """
    def __init__(self, xdata, ydata, q=3, cov=scotts_covariance, kernel=None):
        self.xdata = np.atleast_2d(xdata)
        self.ydata = np.atleast_1d(ydata)
        self.d, self.n = self.xdata.shape
        self.q = q
        if kernel is None:
            kernel = normal_kernel(self.d)
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
        Can be set either as a fixed value or using a bandwith calculator,
        that is a function of signature ``w(xdata, ydata)`` that returns
        a DxD matrix.

        .. note::

            A ndarray with a single value will be converted to a floating
            point value.
        """
        return self._covariance

    @covariance.setter  # noqa
    def covariance(self, cov):
        if callable(cov):
            _cov = cov(self.xdata, self.ydata)
        else:
            _cov = np.atleast_2d(cov)
        self._covariance = _cov
        self._bw = np.real(sqrtm(_cov))

    def evaluate(self, points, out=None):
        """
        Evaluate the spatial averaging on a set of points

        :param ndarray points: Points to evaluate the averaging on
        :param ndarray out: Pre-allocated array for the result
        """
        xdata = self.xdata
        ydata = self.ydata[:, np.newaxis]  # make it a column vector
        points = np.atleast_2d(points)
        n = self.n
        q = self.q
        d = self.d
        designMatrix = PolynomialDesignMatrix(d, q)
        dm_size = designMatrix.size
        Xx = np.empty((dm_size, n), dtype=xdata.dtype)
        WxXx = np.empty(Xx.shape, dtype=xdata.dtype)
        XWX = np.empty((dm_size, dm_size), dtype=xdata.dtype)
        inv_bw = scipy.linalg.inv(self.bandwidth)
        kernel = self.kernel
        if out is None:
            out = np.empty((points.shape[1],), dtype=float)
        for i in irange(points.shape[1]):
            dX = (xdata - points[:, i:i + 1])
            Wx = kernel(np.dot(inv_bw, dX))
            designMatrix(dX, out=Xx)
            np.multiply(Wx, Xx, WxXx)
            np.dot(Xx, WxXx.T, XWX)
            Lx = solve(XWX, WxXx)[0]
            out[i] = np.dot(Lx, ydata)
        return out

    def __call__(self, *args, **kwords):
        """
        This method is an alias for :py:meth:`LocalLinearKernel1D.evaluate`
        """
        return self.evaluate(*args, **kwords)
