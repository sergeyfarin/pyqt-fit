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
    if kde.bandwidth_function:
        bw = np.atleast_2d(reg.bandwidth_function(kde.xdata, model=kde))
        cov = np.dot(bw, bw).real
    elif kde.covariance_function:
        cov = np.atleast_2d(reg.covariance_function(kde.xdata, model=kde))
        bw = sqrtm(cov)
    else:
        return kde.bandwidth, kde.covariance
    return bw, cov


class RegressionKernelMethod(object):
    r"""
    Base class for regression kernel methods
    """
    def fit(self, reg):
        reg.bandwidth, reg.covariance = compute_bandwidth(kde)

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
    def __init__(self, q=3):
        self.q = q

    def evaluate(self, points, out):
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
        for i in irange(points.shape[1]):
            dX = (xdata - points[:, i:i + 1])
            Wx = kernel(np.dot(inv_bw, dX))
            designMatrix(dX, out=Xx)
            np.multiply(Wx, Xx, WxXx)
            np.dot(Xx, WxXx.T, XWX)
            Lx = solve(XWX, WxXx)[0]
            out[i] = np.dot(Lx, ydata)
        return out

default_method = LocalPolynomialKernel(q=1)

