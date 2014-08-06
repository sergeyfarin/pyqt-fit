"""
Pure Python implementation of the kernel functions
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.special import erf
from .utils import numpy_trans, numpy_trans_idx

s2pi = np.sqrt(2.0 * np.pi)
s2 = np.sqrt(2.0)


@numpy_trans
def norm1d_pdf(z, out):
    """
    Full-python implementation of :py:func:`normal_kernel1d.pdf`
    """
    z = np.atleast_1d(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    np.multiply(z, z, out)
    out *= -0.5
    np.exp(out, out)
    out /= s2pi
    return out


@numpy_trans
def norm1d_cdf(z, out):
    """
    Full-python implementation of :py:func:`normal_kernel1d.cdf`
    """
    np.divide(z, s2, out)
    erf(out, out)
    out *= 0.5
    out += 0.5
    return out


@numpy_trans
def norm1d_pm1(z, out):
    """
    Full-python implementation of :py:func:`normal_kernel1d.pm1`
    """
    np.multiply(z, z, out)
    out *= -0.5
    np.exp(out, out)
    out /= -s2pi
    return out


@numpy_trans_idx
def norm1d_pm2(z, out):
    """
    Full-python implementation of :py:func:`normal_kernel1d.pm2`
    """
    np.divide(z, s2, out)
    erf(out, out)
    out /= 2
    if z.shape:
        zz = np.isfinite(z)
        sz = z[zz]
        out[zz] -= sz * np.exp(-0.5 * sz * sz) / s2pi
    elif np.isfinite(z):
        out -= z * np.exp(-0.5 * z * z) / s2pi
    out += 0.5
    return out


tricube_width = np.sqrt(35. / 243)


@numpy_trans_idx
def tricube_pdf(z, out=None):
    np.multiply(z, tricube_width, out)
    sel = (out > -1) & (out < 1)
    out[~sel] = 0
    out[sel] = 70. / 81 * (1 - abs(out[sel]) ** 3.) ** 3. * tricube_width
    return out


@numpy_trans_idx
def tricube_cdf(z, out=None):
    np.multiply(z, tricube_width, out)
    sel_down = out <= -1
    sel_up = out >= 1
    sel_neg = (out < 0) & (~sel_down)
    sel_pos = (out >= 0) & (~sel_up)
    out[sel_up] = 1
    out[sel_down] = 0
    out[sel_pos] = 1. / 162 * \
        (60 * (out[sel_pos] ** 7) - 7. *
         (2 * (out[sel_pos] ** 10) + 15 * (out[sel_pos] ** 4)) +
         140 * out[sel_pos] + 81)
    out[sel_neg] = 1. / 162 * \
        (60 * (out[sel_neg] ** 7) + 7. *
         (2 * (out[sel_neg] ** 10) + 15 * (out[sel_neg] ** 4)) +
         140 * out[sel_neg] + 81)
    return out


@numpy_trans_idx
def tricube_pm1(z, out=None):
    np.multiply(z, tricube_width, out)
    out[out < 0] = -out[out < 0]
    sel = out < 1
    out[~sel] = 0
    out[sel] = 7 / (3564 * tricube_width) * \
        (165 * out[sel] ** 8 - 8 * (5 * out[sel] ** 11 + 33 * out[sel] ** 5) +
         220 * out[sel] ** 2 - 81)
    return out


@numpy_trans_idx
def tricube_pm2(z, out=None):
    np.multiply(z, tricube_width, out)
    sel_down = out <= -1
    sel_up = out >= 1
    sel_neg = (out < 0) & ~sel_down
    sel_pos = (out >= 0) & ~sel_up
    out[sel_down] = 0
    out[sel_up] = 1
    out[sel_pos] = 35. / (tricube_width * tricube_width * 486) * \
        (4 * out[sel_pos] ** 9 - (out[sel_pos] ** 12 + 6 * out[sel_pos] ** 6) +
         4 * out[sel_pos] ** 3 + 1)
    out[sel_neg] = 35. / (tricube_width * tricube_width * 486) * \
        (4 * out[sel_neg] ** 9 + (out[sel_neg] ** 12 + 6 * out[sel_neg] ** 6) +
         4 * out[sel_neg] ** 3 + 1)
    return out

epanechnikov_width = 1. / np.sqrt(5.)

@numpy_trans_idx
def epanechnikov_pdf(z, out=None):
    np.multiply(z, epanechnikov_width, out)
    sel = (out > -1) & (out < 1)
    out[~sel] = 0
    out[sel] = (.75 * epanechnikov_width) * (1 - out[sel] ** 2)
    return out


@numpy_trans_idx
def epanechnikov_cdf(z, out=None):
    np.multiply(z, epanechnikov_width, out)
    sel_up = out >= 1
    sel_down = out <= -1
    out[sel_up] = 1
    out[sel_down] = 0
    sel = ~(sel_up | sel_down)
    out[sel] = .25 * (2 + 3 * out[sel] - out[sel] ** 3)
    return out


@numpy_trans_idx
def epanechnikov_pm1(z, out=None):
    np.multiply(z, epanechnikov_width, out)
    sel = (out > -1) & (out < 1)
    out[~sel] = 0
    out[sel] = -3 / (16 * epanechnikov_width) * \
        (1 - 2 * out[sel] ** 2 + out[sel] ** 4)
    return out


@numpy_trans_idx
def epanechnikov_pm2(z, out=None):
    np.multiply(z, epanechnikov_width, out)
    sel_up = out >= 1
    sel_down = out <= -1
    out[sel_up] = 1
    out[sel_down] = 0
    sel = ~(sel_up | sel_down)
    out[sel] = .25 * (2 + 5 * out[sel] ** 3 - 3 * out[sel] ** 5)
    return out


@numpy_trans
def normal_o4_pdf(z, out=None):
    norm1d_pdf(z, out)
    out *= (3 - z ** 2) / 2
    return out


@numpy_trans_idx
def normal_o4_cdf(z, out=None):
    norm1d_cdf(z, out)
    sel = np.isfinite(z)
    out[sel] += z[sel] * norm1d_pdf(z[sel]) / 2
    return out


@numpy_trans_idx
def normal_o4_pm1(z, out=None):
    norm1d_pdf(z, out)
    out -= normal_o4_pdf(z)
    out[~np.isfinite(z)] = 0
    return out


@numpy_trans_idx
def normal_o4_pm2(z, out=None):
    np.power(z, 3, out)
    out *= norm1d_pdf(z) / 2
    out[~np.isfinite(z)] = 0
    return out


@numpy_trans_idx
def epanechnikov_o4_pdf(z, out=None):
    np.power(z, 2., out)
    out *= -15 / 8.
    out += 9. / 8.
    out[(z < -1) | (z > 1)] = 0
    return out


@numpy_trans_idx
def epanechnikov_o4_cdf(z, out=None):
    np.power(z, 3, out)
    out *= -5. / 8.
    out += (4 + 9 * z) / 8.
    out[z > 1] = 1
    out[z < -1] = 0
    return out


@numpy_trans_idx
def epanechnikov_o4_pm1(z, out=None):
    out = np.power(z, 4, out)
    out *= -15. / 32.
    out += 1. / 32. * (18 * z ** 2 - 3)
    out[(z < -1) | (z > 1)] = 0
    return out


@numpy_trans_idx
def epanechnikov_o4_pm2(z, out=None):
    out = np.power(z, 3, out)
    out *= .375
    out -= .375 * np.power(z, 5)
    out[(z < -1) | (z > 1)] = 0
    return out
