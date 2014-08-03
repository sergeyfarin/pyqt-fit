"""
Pure Python implementation of the kernel functions
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.special import erf


s2pi = np.sqrt(2.0 * np.pi)
s2 = np.sqrt(2.0)


def norm1d_pdf(z, out=None):
    """
    Full-python implementation of :py:func:`normal_kernel1d.pdf`
    """
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    np.multiply(z, z, out)
    out *= -0.5
    np.exp(out, out)
    out /= s2pi
    return out


def norm1d_cdf(z, out=None):
    """
    Full-python implementation of :py:func:`normal_kernel1d.cdf`
    """
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    np.divide(z, s2, out)
    erf(out, out)
    out *= 0.5
    out += 0.5
    return out


def norm1d_pm1(z, out=None):
    """
    Full-python implementation of :py:func:`normal_kernel1d.pm1`
    """
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    np.multiply(z, z, out)
    out *= -0.5
    np.exp(out, out)
    out /= -s2pi
    return out


def norm1d_pm2(z, out=None):
    """
    Full-python implementation of :py:func:`normal_kernel1d.pm2`
    """
    z = np.asfarray(z, dtype=float)
    if out is None:
        out = np.empty(z.shape)
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


def tricube_pdf(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    np.multiply(z, tricube_width, out)
    if out.shape:
        sel = (out > -1) & (out < 1)
        out[~sel] = 0
        out[sel] = 70. / 81 * (1 - abs(out[sel]) ** 3.) ** 3. * tricube_width
    elif out > -1 and out < 1:
        out.setfield(70. / 81 * (1 - abs(out) ** 3.) ** 3. * tricube_width, dtype=float)
    else:
        out.setfield(0.0, dtype=float)
    return out


def tricube_cdf(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    np.multiply(z, tricube_width, out)
    if out.shape:
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
    elif out < -1:
        out.setfield(0, dtype=float)
    elif out > 1:
        out.setfield(1, dtype=float)
    elif out > 0:
        out.setfield(1. / 162 *
                     (60 * (out ** 7) - 7. *
                     (2 * (out ** 10) + 15 * (out ** 4)) +
                     140 * out + 81), dtype=float)
    else:
        out.setfield(1. / 162 *
                     (60 * (out ** 7) + 7. *
                     (2 * (out ** 10) + 15 * (out ** 4)) +
                     140 * out + 81), dtype=float)
    return out


def tricube_pm1(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    np.multiply(z, tricube_width, out)
    if out.shape:
        out[out < 0] = -out[out < 0]
        sel = out < 1
        out[~sel] = 0
        out[sel] = 7 / (3564 * tricube_width) * \
            (165 * out[sel] ** 8 - 8 * (5 * out[sel] ** 11 + 33 * out[sel] ** 5) +
             220 * out[sel] ** 2 - 81)
    else:
        if out < 0:
            out *= -1
        if out < 1:
            out.setfield(7 / (3564 * tricube_width) *
                         (165 * out ** 8 - 8 * (5 * out ** 11 + 33 * out ** 5) +
                          220 * out ** 2 - 81), dtype=float)
        else:
            out.setfield(0.0, dtype=float)
    return out


def tricube_pm2(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    np.multiply(z, tricube_width, out)
    if out.shape:
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
    elif out <= -1:
        out.setfield(0.0, dtype=float)
    elif out >= 1:
        out.setfield(1.0, dtype=float)
    elif out > 0:
        out.setfield(35. / (tricube_width * tricube_width * 486) *
                     (4 * out ** 9 - (out ** 12 + 6 * out ** 6) +
                      4 * out ** 3 + 1), dtype=float)
    else:
        out.setfield(35. / (tricube_width * tricube_width * 486) * \
                     (4 * out ** 9 + (out ** 12 + 6 * out ** 6) +
                      4 * out ** 3 + 1), dtype=float)
    return out

epanechnikov_width = 1. / np.sqrt(5.)

def epanechnikov_pdf(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    np.multiply(z, epanechnikov_width, out)
    if out.shape:
        sel = (out > -1) & (out < 1)
        out[~sel] = 0
        out[sel] = (.75 * epanechnikov_width) * (1 - out[sel] ** 2)
    elif abs(out) > 1:
        out.setfield(0.0, dtype=float)
    else:
        out.setfield((.75 * epanechnikov_width) * (1 - out ** 2), dtype=float)
    return out


def epanechnikov_cdf(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    np.multiply(z, epanechnikov_width, out)
    if out.shape:
        sel_up = out >= 1
        sel_down = out <= -1
        out[sel_up] = 1
        out[sel_down] = 0
        sel = ~(sel_up | sel_down)
        out[sel] = .25 * (2 + 3 * out[sel] - out[sel] ** 3)
    elif out >= 1:
        out.setfield(1.0, dtype=float)
    elif out <= -1:
        out.setfield(0.0, dtype=float)
    else:
        out.setfield(.25 * (2 + 3 * out - out ** 3), dtype=float)
    return out


def epanechnikov_pm1(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    np.multiply(z, epanechnikov_width, out)
    if out.shape:
        sel = (out > -1) & (out < 1)
        out[~sel] = 0
        out[sel] = -3 / (16 * epanechnikov_width) * \
            (1 - 2 * out[sel] ** 2 + out[sel] ** 4)
    elif abs(out) > 1:
        out.setfield(0.0, dtype=float)
    else:
        out.setfield(-3 / (16 * epanechnikov_width) * \
                     (1 - 2 * out ** 2 + out ** 4), dtype=float)
    return out


def epanechnikov_pm2(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    np.multiply(z, epanechnikov_width, out)
    if out.shape:
        sel_up = out >= 1
        sel_down = out <= -1
        out[sel_up] = 1
        out[sel_down] = 0
        sel = ~(sel_up | sel_down)
        out[sel] = .25 * (2 + 5 * out[sel] ** 3 - 3 * out[sel] ** 5)
    elif out >= 1:
        out.setfield(1, dtype=float)
    elif out <= -1:
        out.setfield(0, dtype=float)
    else:
        out.setfield(.25 * (2 + 5 * out ** 3 - 3 * out ** 5), dtype=float)
    return out


def normal_o4_pdf(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    norm1d_pdf(z, out)
    out *= (3 - z ** 2) / 2
    return out


def normal_o4_cdf(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    norm1d_cdf(z, out)
    if out.shape:
        sel = np.isfinite(z)
        out[sel] += z[sel] * norm1d_pdf(z[sel]) / 2
    elif np.isfinite(z):
        out += z * norm1d_pdf(z) / 2
    return out


def normal_o4_pm1(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    norm1d_pdf(z, out)
    out -= normal_o4_pdf(z)
    if out.shape:
        out[~np.isfinite(z)] = 0
    elif not np.isfinite(z):
        out.setfield(0, dtype=float)
    return out


def normal_o4_pm2(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    np.power(z, 3, out)
    out *= norm1d_pdf(z) / 2
    if out.shape:
        out[~np.isfinite(z)] = 0
    elif not np.isfinite(z):
        out.setfield(0, dtype=float)
    return out


def epanechnikov_o4_pdf(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    np.power(z, 2., out)
    out *= -15 / 8.
    out += 9. / 8.
    if out.shape:
        out[(z < -1) | (z > 1)] = 0
    elif abs(z) > 1:
        out.setfield(0.0, dtype=float)
    return out


def epanechnikov_o4_cdf(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    np.power(z, 3, out)
    out *= -5. / 8.
    out += (4 + 9 * z) / 8.
    if out.shape:
        out[z > 1] = 1
        out[z < -1] = 0
    elif z > 1:
        out.setfield(1.0, dtype=float)
    elif z < -1:
        out.setfield(0.0, dtype=float)
    return out


def epanechnikov_o4_pm1(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    out = np.power(z, 4, out)
    out *= -15. / 32.
    out += 1. / 32. * (18 * z ** 2 - 3)
    if out.shape:
        out[(z < -1) | (z > 1)] = 0
    elif abs(z) > 1:
        out.setfield(0.0, dtype=float)
    return out


def epanechnikov_o4_pm2(z, out=None):
    z = np.asfarray(z)
    if out is None:
        out = np.empty(z.shape, dtype=z.dtype)
    out = np.power(z, 3, out)
    out *= .375
    out -= .375 * np.power(z, 5)
    if out.shape:
        out[(z < -1) | (z > 1)] = 0
    elif abs(z) > 1:
        out.setfield(0.0, dtype=float)
    return out
