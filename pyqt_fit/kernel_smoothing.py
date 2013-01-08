"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

Module implementing non-parametric regressions using kernel smoothing methods.
"""

from scipy.linalg import sqrtm
from scipy.stats import gaussian_kde
import numpy as np
import cyth
import cy_local_linear

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

def silverman_bandwidth(xdata, ydata):
    r"""
    The Silverman bandwidth is defined as a variance bandwidth with factor:

    .. math::

        \tau = \left( n \frac{d+2}{4} \right)^\frac{-1}{d+4}
    """
    n = ydata.shape[0]
    if len(xdata.shape) == 2:
        d = float(xdata.shape[0])
    else:
        d = 1.
    return variance_bandwidth(np.power(n*(d+2.)/4., -1./(d+4.)), xdata)

def scotts_bandwidth(xdata, ydata):
    r"""
    The Scotts bandwidth is defined as a variance bandwidth with factor:

    .. math::

        \tau = n^\frac{-1}{d+4}
    """
    n = ydata.shape[0]
    if len(xdata.shape) == 2:
        d = float(xdata.shape[0])
    else:
        d = 1.
    return variance_bandwidth(np.power(n, -1./(d+4.)), xdata)

class SpatialAverage(object):
    r"""
    Perform a Nadaraya-Watson regression on the data (i.e. also called local-constant regression) using a gaussian kernel.

    The Nadaraya-Watson estimate is given by:

    .. math::

        f_n(X) \triangleq \frac{\sum_i K\left(\frac{x-X_i}{h}\right) Y_i}{\sum_i K\left(\frac{x-X_i}{h}\right)}

    Where :math:`K(x)` is the kernel and must be such that :math:`E(K(x)) = 0` and :math:`h` is the bandwidth of the
    method.

    :param ndarray xdata: Explaining variables (at most 2D array)
    :param ndarray ydata: Explained variables (should be 1D array)

    :type  cov: ndarray or callable
    :param cov: If an ndarray, it should be a 2D array giving the matrix of covariance of the gaussian kernel.
        Otherwise, it should be a function ``cov(xdata, ydata)`` returning the covariance matrix.

    """

    def __init__(self, xdata, ydata, cov = scotts_bandwidth):
        self.xdata = np.atleast_2d(xdata)
        self.ydata = ydata

        self._bw = None
        self._covariance = None
        self._inv_cov = None

        self.covariance = cov

        self.d, self.n = self.xdata.shape
        self.correction = 1.

    @property
    def bandwidth(self):
        """
        Bandwidth of the kernel. It cannot be set directly, but rather should be set via the covariance attribute.
        """
        if self._bw is None and self._covariance is not None:
            self._bw = sqrtm(self._covariance)
        return self._bw

    @property
    def covariance(self):
        """
        Covariance of the gaussian kernel.
        Can be set either as a fixed value or using a bandwith calculator, that is a function
        of signature ``w(xdata, ydata)`` that returns a 2D matrix for the covariance of the kernel.
        """
        return self._covariance

    @covariance.setter
    def covariance(self, cov):
        if callable(cov):
            _cov = np.atleast_2d(bw(self.xdata, self.ydata))
        else:
            _cov = np.atleast_2d(bw)
        self._bw = None
        self._covariance = _cov
        self._inv_cov = np.linalg.inv(_cov)


    def evaluate(self, points, result = None):
        """
        Evaluate the spatial averaging on a set of points

        :param ndarray points: Points to evaluate the averaging on
        :param ndarray result: If provided, the result will be put in this array
        """
        points = np.atleast_2d(points).astype(self.xdata.dtype)
        #norm = self.kde(points)
        d, m = points.shape
        if result is None:
            result = np.zeros((m,), points.dtype)
        norm = np.zeros((m,), points.dtype)

        # iterate on the internal points
        for i,ci in np.broadcast(xrange(self.n), xrange(self._correction.shape[0])):
            diff = np.dot(self._correction[ci], self.xdata[:,i,np.newaxis] - points)
            tdiff = np.dot(self._inv_cov, diff)
            energy = np.exp(-np.sum(diff*tdiff,axis=0)/2.0)
            result += self.ydata[i]*energy
            norm += energy

        result[norm>1e-50] /= norm[norm>1e-50]

        return result

    def __call__(self, *args, **kwords):
        """
        This method is an alias for :py:meth:`SpatialAverage.evaluate`
        """
        return self.evaluate(*args, **kwords)

    @property
    def correction(self):
        """
        The correction coefficient allows to change the width of the kernel depending on the point considered.
        It can be either a constant (to correct globaly the kernel width), or a 1D array of same size as the input.
        """
        return self._correction

    @correction.setter
    def correction(self, value):
        self._correction = np.atleast_1d(value)

    def set_density_correction(self):
        """
        Add a correction coefficient depending on the density of the input
        """
        kde = gaussian_kde(xdata)
        dens = kde(xdata)
        dm = dens.max()
        dens[dens < 1e-50] = dm
        self._correction = dm/dens

class LocalLinearKernel1D(object):
    r"""
    Perform a local-linear regression using a gaussian kernel.

    The local constant regression is the function that minimises, for each position:

    .. math::

        \DeclareMathOperator{\argmin}{argmin}
        f_n(X) \triangleq \argmin_{a_0\in\mathbb{R}} \sum_i K\left(\frac{x-X_i}{h}\right)\left(Y_i - a_0 -
        a_1(x-X_i)\right)^2

    Where :math:`K(x)` is the kernel and must be such that :math:`E(K(x)) = 0` and :math:`h` is the bandwidth of the
    method.

    :param ndarray xdata: Explaining variables (at most 2D array)
    :param ndarray ydata: Explained variables (should be 1D array)

    :type  cov: float or callable
    :param cov: If an float, it should be a variance of the gaussian kernel.
        Otherwise, it should be a function ``cov(xdata, ydata)`` returning the variance.

    """
    def __init__(self, xdata, ydata, cov = scotts_bandwidth):
        self.xdata = np.atleast_1d(xdata)
        self.ydata = np.atleast_1d(ydata)
        self.n = xdata.shape[0]

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
        of signature ``w(xdata, ydata)`` that returns a single value.

        .. note::

            A ndarray with a single value will be converted to a floating point value.
        """
        return self._covariance

    @covariance.setter
    def covariance(self, cov):
        if callable(cov):
            _cov = float(cov(self.xdata, self.ydata))
        else:
            _cov = float(cov)
        self._covariance = _cov
        self._bw = np.sqrt(_cov)

    def evaluate(self, points, output=None):
        """
        Evaluate the spatial averaging on a set of points

        :param ndarray points: Points to evaluate the averaging on
        :param ndarray result: If provided, the result will be put in this array
        """
        li2, output = cy_local_linear.cy_local_linear_1d(self._bw, self.xdata, self.ydata, points, output)
        self.li2 = li2
        return output
        #points = np.atleast_1d(points).astype(self.xdata.dtype)
        #m = points.shape[0]
        #x0 = points - self.xdata[:,np.newaxis]
        #x02 = x0*x0
        #wi = np.exp(-self.inv_cov*x02/2.0)
        #X = np.sum(wi*x0, axis=0)
        #X2 = np.sum(wi*x02, axis=0)
        #wy = wi*self.ydata[:,np.newaxis]
        #Y = np.sum(wy, axis=0)
        #Y2 = np.sum(wy*x0, axis=0)
        #W = np.sum(wi, axis=0)
        #return np.divide(X2*Y-Y2*X, W*X2-X*X, output)

    def __call__(self, *args, **kwords):
        """
        This method is an alias for :py:meth:`LocalLinearKernel1D.evaluate`
        """
        return self.evaluate(*args, **kwords)

