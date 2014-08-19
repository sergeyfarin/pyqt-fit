#cython profile=True
"""
cython -a fast_linbin.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/ -o fast_linbin.so fast_linbin.c
"""

cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport floor

ctypedef np.float64_t DOUBLE
ctypedef np.int_t INT

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def fast_linbin(np.ndarray[DOUBLE] X, double a, double b, int M, np.ndarray[DOUBLE] weights = None, int cyclic=0):
    r"""
    Linear Binning as described in Fan and Marron (1994)

    :param X ndarray: Input data
    :param a float: Lowest value to consider
    :param b float: Highest valus to consider
    :param M int: Number of bins
    :param weights ndarray: Array of same size as X with weights for each point, or None if all weights are 1
    :param cyclic bool: Consider the data cyclic or not

    :Returns: The weights in each bin

    For a point :math:`x` between bins :math:`b_i` and :math:`b_{i+1}` at positions :math:`p_i` and :math:`p_{i+1}`, the 
    bins will be updated as:

    .. math::

        b_i = b_i + \frac{b_{i+1} - x}{b_{i+1} - b_i}

        b_{i+1} = b_{i+1} + \frac{x - b_i}{b_{i+1} - b_i}

    By default the bins will be placed at :math:`\{a+\delta/2, \ldots, a+k \delta + \delta/1, \ldots b-\delta/2\}` with 
    :math:`delta = \frac{M-1}{b-a}`.

    If cyclic is true, then the bins are placed at :math:`\{a, \ldots, a+k \delta, \ldots, b-\delta\}` with 
    :math:`\delta = \frac{M}{b-a}` and there is a virtual bin in :math:`b` which is fused with :math:`a`.

    """
    cdef:
        Py_ssize_t i, li_i
        int nobs = X.shape[0]
        np.ndarray[DOUBLE] gcnts = np.zeros(M, np.float)
        np.ndarray[DOUBLE] mesh
        double delta = (b - a) / M
        double inv_delta = 1 / delta
        double shift
        double rem
        double val
        double lower
        double upper
        double w
        int base_idx
        int N
        int has_weights = weights is not None

    if has_weights:
        assert weights.shape[0] == X.shape[0], "Error, the weights must be None or an array of same size as X"

    if cyclic:
        shift = -a
        lower = 0
        upper = M
    else:
        shift = -a-delta/2
        lower = -0.5
        upper = M-0.5

    for i in range(nobs):
        val = (X[i] + shift) * inv_delta
        if val >= lower and val <= upper:
            base_idx = <int> floor(val);
            if not cyclic and val < 0:
                rem = 1
            elif not cyclic and val > M-1:
                rem = 0
            else:
                rem = val - base_idx
            if has_weights:
                w = weights[i]
            else:
                w = 1.
            if base_idx == M:
                gcnts[0] = w
            else:
                if base_idx >= 0:
                    gcnts[base_idx] += (1 - rem)*w
                if base_idx < M-1:
                    gcnts[base_idx+1] += rem*w
                elif base_idx == M-1:
                    gcnts[0] += rem

    if cyclic:
        mesh = np.linspace(a, b-delta, M)
    else:
        mesh = np.linspace(a+delta/2, b-delta/2, M)

    #print("cyclic = {2} : delta = {0} - mesh[1] - mesh[0] = {1}".format(delta, mesh[1]-mesh[0], cyclic))

    return gcnts, mesh

