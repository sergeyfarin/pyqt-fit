#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
from math cimport isfinite, erf
from libc.math cimport exp, sqrt, M_PI

np.import_array()

ctypedef np.npy_float64 float64_t

cdef float64_t S2PI = sqrt(2.0*M_PI)
cdef float64_t S2 = sqrt(2.0)

s2pi = S2PI
s2 = S2

cdef void _vectorize(object z,
                     object out,
                     float64_t (*fct)(float64_t v)):
    cdef np.broadcast it = np.broadcast(z, out)
    while np.PyArray_MultiIter_NOTDONE(it):
        (<float64_t*> np.PyArray_MultiIter_DATA(it, 1))[0] = fct((<float64_t*> np.PyArray_MultiIter_DATA(it, 0))[0])
        np.PyArray_MultiIter_NEXT(it)

cdef object vectorize(object z,
                      object out,
                      float64_t (*fct)(float64_t v)):
    if z.shape:
        if out is None:
            out = np.empty(z.shape, dtype=np.float64)
        _vectorize(z, out, fct)
        return out
    else:
        return fct(<float64_t>z)

cdef float64_t _norm1d_pdf(float64_t z):
    return exp(-z*z/2)/S2PI

def norm1d_pdf(np.ndarray z, object out = None):
    return vectorize(z, out, _norm1d_pdf)

cdef float64_t _norm1d_cdf(float64_t z):
    return erf(z/S2) / 2 + 0.5

def norm1d_cdf(np.ndarray z, object out = None):
    return vectorize(z, out, _norm1d_cdf)

cdef float64_t _norm1d_pm1(float64_t z):
    return -exp(-z*z/2) / S2PI

def norm1d_pm1(np.ndarray z, object out = None):
    return vectorize(z, out, _norm1d_pm1)

cdef float64_t _norm1d_pm2(float64_t z):
    if isfinite(z):
        return 0.5*erf(z/S2) + 0.5 - z/S2PI*exp(-z*z/2)
    return 0.5*erf(z/S2)+0.5

def norm1d_pm2(np.ndarray z, object out = None):
    return vectorize(z, out, _norm1d_pm2)

