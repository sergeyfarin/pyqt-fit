r"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module contains a set of methods to compute univariate KDEs. See the objects in the :py:mod:`pyqt_fit.kde` module 
for more details on these methods.

The definitions of the methods rely on the following definitions:

.. math::

   \begin{array}{rclc|crcl}
     a_0(l,u) &=& \int_l^u K(z) dz &\quad&\quad& z &=& \frac{x-X}{h} \\
     a_1(l,u) &=& \int_l^u zK(z) dz &\quad&\quad& l &=& \frac{L-X}{h} \\
     a_2(l,u) &=& \int_l^u z^2K(z) dz &\quad&\quad& u &=& \frac{U-X}{h}
   \end{array}

where :math:`x` is the point where the distribution is evaluated, :math:`X` is the vector of data points the 
distribution is evaluated from and :math:`(L,U)` are the bounds of the distribution's domain (may be infinite).

These definitions correspond to:

- :math:`a_0(l,u)` -- The cumulative distribution function
- :math:`a_1(l,u)` -- The first moment of the distribution. In particular, :math:`a_1(-\infty, \infty)` is the mean of 
  the kernel (i.e. and should be 0).
- :math:`a_2(l,u)` -- The second moment of the distribution. In particular, :math:`a_2(-\infty, \infty)` is the variance 
  of the kernel (i.e. which should be close to 1, unless using higher order kernel).
- :math:`z` -- The position of the points in the reference system of the kernel.
- :math:`l` -- The position of the lower bound of the distribution domain, in the reference system of the kernel
- :math:`u` -- The position of the upper bound of the distribution domain, in the reference system of the kernel

References:
```````````
.. [1] Jones, M. C. 1993. Simple boundary correction for kernel density
    estimation. Statistics and Computing 3: 135--146.
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from scipy import fftpack, integrate
from .compat import irange

def generate_grid(kde, N=None, cut=None):
    r"""
    Helper method returning a regular grid on the domain of the KDE.

    :param KDE1D kde: Object describing the KDE computation
    :param int N: Number of points in the grid
    :param float cut: for unbounded domains, how far past the maximum should the grid extend to, in term of KDE
        bandwidth

    :return: A vector of N regularly spaced points
    """
    if N is None:
        N = 2 ** 10
    if cut is None:
        cut = 3
    if kde.lower == -np.inf:
        lower = np.min(kde.xdata) - cut * kde.bandwidth
    else:
        lower = kde.lower
    if kde.upper == np.inf:
        upper = np.max(kde.xdata) + cut * kde.bandwidth
    else:
        upper = kde.upper
    return np.linspace(lower, upper, N)

class KDE1DMethod(object):
    """
    Base class providing a default grid method and a default method for unbounded evaluation.
    """

    @staticmethod
    def unbounded(kde, points, output):
        """
        Method to use if there is, effectively, no bounds
        """
        xdata = kde.xdata
        points = np.atleast_1d(points)[:, np.newaxis]

        bw = kde.bandwidth * kde.lambdas

        z = (points - xdata) / bw

        kernel = kde.kernel

        terms = kernel(z)

        terms *= kde.weights / bw

        output = terms.sum(axis=1, out=output)
        output /= kde.total_weights

        return output

    @staticmethod
    def unbounded_cdf(kde, points, output=None):
        """
        Compute the cdf in case the domain is fully unbounded.

        This function rely on the `kernel.cdf` method.
        """
        xdata = kde.xdata
        points = np.atleast_1d(points)[:, np.newaxis]
        bw = kde.bandwidth * kde.lambdas

        z = (points - xdata) / bw

        kernel = kde.kernel

        terms = kernel.cdf(z)
        terms *= kde.weights

        output = terms.sum(axis=1, out=output)
        output /= kde.total_weights

        return output

    @staticmethod
    def numeric_cdf(kde, points, output=None):
        """
        Provide a numeric approximation of the cdf based on integrating the pdf using :py:func:`scipy.integrate.quad`.
        """
        pts = np.atleast_1d(np.array(points, dtype=float))
        pts_shape = pts.shape
        pts = pts.ravel()

        pts[pts < kde.lower] = kde.lower
        pts[pts > kde.upper] = kde.upper

        ix = pts.argsort()

        sp = pts[ix]

        start = 0.0
        if sp[0] > kde.lower:
            start = integrate.quad(kde, kde.lower, sp[0])[0]

        parts = np.empty(sp.shape, dtype=float)
        parts[0] = start
        for i in range(1, len(parts)):
            parts[i] = integrate.quad(kde, sp[i-1], sp[i])[0]

        ints = parts.cumsum()
        if output is None:
            output = np.empty(pts_shape, dtype=float)

        output.put(ix, ints)
        return output

    @staticmethod
    def numeric_cdf_grid(kde, N=None, cut=None):
        """
        Compute the cdf on a grid using a trivial, but fast, numeric integration.

        Will return N+1 points over the whole domain.
        """
        if N is None:
            N = 2**10
        pts, pdf = kde.grid(N, cut)
        cdf_pts = (pts[1:] + pts[:-1]) / 2
        dx = pts[1] - pts[0]
        low_cdf = pts[0] - dx/2 if kde.lower == -np.inf else kde.lower
        high_cdf = pts[-1] + dx/2 if kde.upper == np.inf else kde.upper
        cdf_pts = np.r_[low_cdf, cdf_pts, high_cdf]
        cdf = np.empty(cdf_pts.shape, dtype=float)
        cdf[1:] = (pdf * (cdf_pts[1:] - cdf_pts[:-1])).cumsum()
        cdf[0] = 0.0
        return cdf_pts, cdf

    __call__ = unbounded

    def default_grid(self, kde, N=None, cut=None):
        """
        Evaluate the method on a grid spanning the whole domain of the KDE and containing N points.

        :param KDE1D kde: KDE object
        :param int N: Number of points of the grid
        :param float cut: Cutting points for the unbounded domain (see :py:func:`generate_grid`)

        :returns: A tuple with the grid points and the estimated values on these points
        """
        g = generate_grid(kde, N, cut)
        return g, self(kde, g)


    def default_cdf_grid(self, kde, N=None, cut=None):
        """
        Evaluate the method on a grid spanning the whole domain of the KDE and containing N points.

        :param KDE1D kde: KDE object
        :param int N: Number of points of the grid
        :param float cut: Cutting points for the unbounded domain (see :py:func:`generate_grid`)

        :returns: A tuple with the grid points and the estimated values on these points
        """
        g = generate_grid(kde, N, cut)
        return g, self.cdf(kde, g)

    def grid(self, kde, N=None, cut=None):
        """
        Call :py:meth:`default_grid`
        """
        return self.default_grid(kde, N, cut)

    def cdf_grid(self, kde, N=None, cut=None):
        """
        Call :py:meth:`default_cdf_grid`
        """
        return self.default_cdf_grid(kde, N, cut)

    def __str__(self):
        """
        Return the name of the method
        """
        return self.name

class RenormalizationMethod(KDE1DMethod):
    r"""
    This method consists in using the normal kernel method, but renormalize
    to only take into account the part of the kernel within the domain of
    the density [1]_.

    The kernel is then replaced with:

    .. math::

        \hat{K}(x;X,h,L,U) \triangleq \frac{1}{a_0(l,u)} K(z)

    See the :py:mod:`pyqt_fit.kde_methods` for a description of the various symbols.
    """

    name = 'renormalization'

    @staticmethod
    def __call__(kde, points, output=None):
        if not kde.bounded:
            return KDE1DMethod.unbounded(kde, points, output)

        xdata = kde.xdata
        points = np.atleast_1d(points)[:, np.newaxis]

        bw = kde.bandwidth * kde.lambdas

        l = (kde.lower - xdata) / bw
        u = (kde.upper - xdata) / bw
        z = (points - xdata) / bw

        kernel = kde.kernel

        a1 = (kernel.cdf(u) - kernel.cdf(l))

        terms = kernel(z) * ((kde.weights / bw) / a1)

        output = terms.sum(axis=1, out=output)
        output /= kde.total_weights

        return output

    @staticmethod
    def cdf(kde, points, output=None):
        if not kde.bounded:
            return KDE1DMethod.unbounded_cdf(kde, points, output)

        xdata = kde.xdata
        points = np.atleast_1d(points)[:, np.newaxis]

        bw = kde.bandwidth * kde.lambdas

        l = (kde.lower - xdata) / bw
        u = (kde.upper - xdata) / bw
        z = (points - xdata) / bw

        kernel = kde.kernel

        cl = kernel.cdf(l)
        cu = kernel.cdf(u)
        a1 = (cu - cl)

        terms = (kernel.cdf(z) - cl) * (kde.weights / a1)

        output = terms.sum(axis=1, out=output)
        output /= kde.total_weights

        return output


renormalization = RenormalizationMethod()

class ReflectionMethod(KDE1DMethod):
    r"""
    This method consist in simulating the reflection of the data left and
    right of the boundaries. If one of the boundary is infinite, then the
    data is not reflected in that direction. To this purpose, the kernel is
    replaced with:

    .. math::

        \hat{K}(x; X, h, L, U) \triangleq K(z)
        + K\left(\frac{x+X-2L}{h}\right)
        + K\left(\frac{x+X-2U}{h}\right)

    See the :py:mod:`pyqt_fit.kde_methods` for a description of the various symbols.

    When computing grids, if the bandwidth is constant, the result is
    computing using CDT.
    """

    name = 'reflection'

    @staticmethod
    def __call__(kde, points, output=None):
        if not kde.bounded:
            return KDE1DMethod.unbounded(kde, points, output)

        xdata = kde.xdata
        points = np.atleast_1d(points)[:, np.newaxis]

        # Make sure points are between the bounds, with reflection if needed
        if any(points < kde.lower) or any(points > kde.upper):
            span = kde.upper - kde.lower
            points = points - (kde.lower + span)
            points %= 2*span
            points -= kde.lower + span
            points = np.abs(points)

        bw = kde.bandwidth * kde.lambdas

        z = (points - xdata) / bw
        z1 = (points + xdata) / bw
        L = kde.lower
        U = kde.upper

        kernel = kde.kernel

        terms = kernel(z)

        if L > -np.inf:
            terms += kernel(z1 - (2 * L / bw))

        if U < np.inf:
            terms += kernel(z1 - (2 * U / bw))

        terms *= kde.weights / bw
        output = terms.sum(axis=1, out=output)
        output /= kde.total_weights

        return output

    @staticmethod
    def cdf(kde, points, output=None):
        if not kde.bounded:
            return KDE1DMethod.unbounded_cdf(kde, points, output)

        xdata = kde.xdata
        points = np.atleast_1d(points)[:, np.newaxis]

        # Make sure points are between the bounds, with reflection if needed
        if any(points < kde.lower) or any(points > kde.upper):
            span = kde.upper - kde.lower
            points = points - (kde.lower + span)
            points %= 2*span
            points -= kde.lower + span
            points = np.abs(points)

        bw = kde.bandwidth * kde.lambdas

        z = (points - xdata) / bw
        z1 = (points + xdata) / bw
        L = kde.lower
        U = kde.upper

        kernel = kde.kernel

        terms = kernel.cdf(z)

        if L > -np.inf:
            terms -= kernel.cdf((L - xdata) / bw) # Remove the truncated part on the left
            terms += kernel.cdf(z1 - (2 * L / bw)) # Add the reflected part
            terms -= kernel.cdf((xdata - L) / bw) # Remove the truncated part from the reflection

        if U < np.inf:
            terms += kernel.cdf(z1 - (2 * U / bw)) # Add the reflected part

        terms *= kde.weights
        output = terms.sum(axis=1, out=output)
        output /= kde.total_weights

        return output


    def grid(self, kde, N=None, cut=None):
        """
        DCT-based estimation of KDE estimation, i.e. with reflection boundary
        conditions. This works only for fixed bandwidth (i.e. lambdas = 1) and
        gaussian kernel.

        For open domains, the grid is taken with 3 times the bandwidth as extra
        space to remove the boundary problems.
        """
        if kde.lambdas.shape:
            return self.default_grid(kde, N, cut)

        bw = kde.bandwidth * kde.lambdas
        data = kde.xdata
        if N is None:
            N = 2 ** 14

        if kde.lower == -np.inf:
            lower = np.min(data) - 3 * kde.bandwidth
        else:
            lower = kde.lower
        if kde.upper == np.inf:
            upper = np.max(data) + 3 * kde.bandwidth
        else:
            upper = kde.upper

        R = upper - lower

        # Histogram the data to get a crude first approximation of the density
        weights = kde.weights
        if not weights.shape:
            weights = None
        DataHist, bins = np.histogram(data, bins=N, range=(lower, upper),
                                      weights=weights)
        DataHist = DataHist / kde.total_weights
        DCTData = fftpack.dct(DataHist, norm=None)

        t_star = bw / R
        gp = np.arange(N) * np.pi * t_star
        smth = kde.kernel.dct(gp)

        # Smooth the DCTransformed data using t_star
        SmDCTData = DCTData * smth
        # Inverse DCT to get density
        density = fftpack.idct(SmDCTData, norm=None) / (2 * R)
        mesh = np.array([(bins[i] + bins[i + 1]) / 2 for i in irange(N)])

        return mesh, density

reflection = ReflectionMethod()

class LinearCombinationMethod(KDE1DMethod):
    r"""
    This method uses the linear combination correction published in [1]_.

    The estimation is done with a modified kernel given by:

    .. math::

        \hat{K}(x;X,h,L,U) \triangleq \frac{a_2(l,u) - a_1(-u, -l) z}{a_2(l,u)a_0(l,u)
        - a_1(-u,-l)^2} K(z)

    See the :py:mod:`pyqt_fit.kde_methods` for a description of the various symbols.
    """

    name = 'linear combination'

    @staticmethod
    def __call__(kde, points, output=None):
        if not kde.bounded:
            return KDE1DMethod.unbounded(kde, points, output)

        xdata = kde.xdata
        points = np.atleast_1d(points)[:, np.newaxis]

        bw = kde.bandwidth * kde.lambdas

        l = (kde.lower - points) / bw
        u = (kde.upper - points) / bw
        z = (points - xdata) / bw

        kernel = kde.kernel

        a0 = kernel.cdf(u) - kernel.cdf(l)
        a1 = kernel.pm1(-l) - kernel.pm1(-u)
        a2 = kernel.pm2(u) - kernel.pm2(l)

        denom = a2 * a0 - a1 * a1
        upper = a2 - a1 * z

        upper /= denom
        upper *= (kde.weights / bw) * kernel(z)

        output = upper.sum(axis=1, out=output)
        output /= kde.total_weights

        return output

    @staticmethod
    def cdf(kde, points, output=None):
        if not kde.bounded:
            return KDE1DMethod.unbounded_cdf(kde, points, output)
        return KDE1DMethod.numeric_cdf(kde, points, output)

    @staticmethod
    def cdf_grid(kde, N=None, cut=None):
        if N is None:
            N = 2**10
        if not kde.bounded or N < 2**10:
            return KDE1DMethod.default_cdf_grid(kde, N, cut)
        return KDE1DMethod.numeric_cdf_grid(kde, N, cut)

linear_combination = LinearCombinationMethod()

class CyclicMethod(KDE1DMethod):
    r"""
    This method assumes cyclic boundary conditions and works only for
    closed boundaries.

    The estimation is done with a modified kernel given by:

    .. math::

        \hat{K}(x; X, h, L, U) \triangleq K(z)
        + K\left(z - \frac{U-L}{h}\right)
        + K\left(z + \frac{U-L}{h}\right)

    See the :py:mod:`pyqt_fit.kde_methods` for a description of the various symbols.

    When computing grids, if the bandwidth is constant, the result is
    computing using FFT.
    """

    name = 'cyclic'

    @staticmethod
    def __call__(kde, points, output=None):
        if not kde.closed:
            raise ValueError("Cyclic boundary conditions can only be used with "
                             "closed domains.")

        xdata = kde.xdata
        points = np.atleast_1d(points)[:, np.newaxis]

        # Make sure points are between the bounds
        if any(points < kde.lower) or any(points > kde.upper):
            points = points - kde.lower
            points %= kde.upper - kde.lower
            points += kde.lower

        bw = kde.bandwidth * kde.lambdas

        z = (points - xdata) / bw
        L = kde.lower
        U = kde.upper

        span = (U - L) / bw

        kernel = kde.kernel

        terms = kernel(z)
        terms += kernel(z + span) # Add points to the left
        terms += kernel(z - span) # Add points to the right

        terms *= kde.weights / bw
        output = terms.sum(axis=1, out=output)
        output /= kde.total_weights

        return output


    @staticmethod
    def cdf(kde, points, output=None):
        if not kde.closed:
            raise ValueError("Cyclic boundary conditions can only be used with "
                             "closed domains.")

        xdata = kde.xdata
        points = np.atleast_1d(points)[:, np.newaxis]

        # Make sure points are between the bounds
        if any(points < kde.lower) or any(points > kde.upper):
            points = points - kde.lower
            points %= kde.upper - kde.lower
            points += kde.lower

        bw = kde.bandwidth * kde.lambdas

        z = (points - xdata) / bw
        L = kde.lower
        U = kde.upper

        span = (U - L) / bw

        kernel = kde.kernel

        terms = kernel.cdf(z)
        terms -= kernel.cdf((L - xdata) / bw) # Remove the parts left of the lower bound

        terms += kernel.cdf(z + span) # Repeat on the left
        terms -= kernel.cdf((L - xdata) / bw + span) # Remove parts left of lower bounds

        terms += kernel.cdf(z - span) # Repeat on the right

        terms *= kde.weights
        output = terms.sum(axis=1, out=output)
        output /= kde.total_weights

        return output

    def grid(self, kde, N=None, cut=None):
        """
        FFT-based estimation of KDE estimation, i.e. with cyclic boundary
        conditions. This works only for closed domains, fixed bandwidth
        (i.e. lambdas = 1) and gaussian kernel.
        """
        if kde.lambdas.shape:
            return self.default_grid(kde, N, cut)
        if not kde.closed:
            raise ValueError("Error, cyclic boundary conditions require "
                             "a closed domain.")
        bw = kde.bandwidth * kde.lambdas
        data = kde.xdata
        if N is None:
            N = 2 ** 14
        lower = kde.lower
        upper = kde.upper
        R = upper - lower
        dN = 1 / N
        mesh = np.r_[lower:upper + dN:(N + 2) * 1j]
        weights = kde.weights
        if not weights.shape:
            weights = None
        DataHist, bin_edges = np.histogram(data, bins=mesh - dN / 2,
                                           weights=weights)
        DataHist[0] += DataHist[-1]
        DataHist = DataHist / kde.total_weights
        FFTData = fftpack.fft(DataHist[:-1])

        t_star = (2 * bw / R)
        gp = np.roll((np.arange(N) - N / 2) * np.pi * t_star, N // 2)
        smth = kde.kernel.fft(gp)

        SmoothFFTData = FFTData * smth
        density = fftpack.ifft(SmoothFFTData) / (mesh[1] - mesh[0])
        return mesh[:-2], density.real

cyclic = CyclicMethod()

