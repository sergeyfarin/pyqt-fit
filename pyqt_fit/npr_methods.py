"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

Module implementing non-parametric regressions using kernel methods.
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

def compute_bandwidth(reg):
    """
    Compute the bandwidth and covariance for the model, based of its xdata attribute
    """
    if reg.bandwidth_function:
        bw = np.atleast_2d(reg.bandwidth_function(reg.xdata, model=reg))
        cov = np.dot(bw, bw).real
    elif reg.covariance_function:
        cov = np.atleast_2d(reg.covariance_function(reg.xdata, model=reg))
        bw = sqrtm(cov)
    else:
        return reg.bandwidth, reg.covariance
    return bw, cov


class RegressionKernelMethod(object):
    r"""
    Base class for regression kernel methods
    """
    def fit(self, reg):
        reg.compute_bandwidth()


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


class LocalPolynomialKernel(RegressionKernelMethod):
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
    def __init__(self, reg, q=3):
        self.q = q

    def fit(self, reg):
        super(LocalPolynomialKernel, self).fit(reg)
        self.designMatrix = PolynomialDesignMatrix(reg.dim, self.q)

    def evaluate(self, reg, points, out):
        """
        Evaluate the spatial averaging on a set of points

        :param ndarray points: Points to evaluate the averaging on
        :param ndarray out: Pre-allocated array for the result
        """
        xdata = reg.xdata
        ydata = reg.ydata[:, np.newaxis]  # make it a column vector
        d, n = xdata.shape
        designMatrix = self.designMatrix
        dm_size = designMatrix.size
        Xx = np.empty((dm_size, n), dtype=xdata.dtype)
        WxXx = np.empty(Xx.shape, dtype=xdata.dtype)
        XWX = np.empty((dm_size, dm_size), dtype=xdata.dtype)
        inv_bw = scipy.linalg.inv(reg.bandwidth)
        kernel = reg.kernel
        for i in irange(points.shape[1]):
            dX = (xdata - points[:, i:i + 1])
            Wx = kernel(np.dot(inv_bw, dX))
            designMatrix(dX, out=Xx)
            np.multiply(Wx, Xx, WxXx)
            np.dot(Xx, WxXx.T, XWX)
            Lx = solve(XWX, WxXx)[0]
            out[i] = np.dot(Lx, ydata)
        return out

default_method = LocalPolynomialKernel

