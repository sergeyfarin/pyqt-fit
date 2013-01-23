#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
from math cimport erf, isfinite
from libc.math cimport exp, sqrt, M_PI

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef double S2PI = sqrt(2*<double>M_PI)
cdef double S2 = sqrt(2)

cdef void _vectorize(object z,
                     object out,
                     DTYPE_t (*fct)(DTYPE_t v)):
    cdef np.broadcast it = np.broadcast(z, out)
    while np.PyArray_MultiIter_NOTDONE(it):
        (<double*> np.PyArray_MultiIter_DATA(it, 1))[0] = fct((<double*> np.PyArray_MultiIter_DATA(it, 0))[0])
        np.PyArray_MultiIter_NEXT(it)

cdef object vectorize(object z,
                      object out,
                      DTYPE_t (*fct)(DTYPE_t v)):
    if z.shape:
        if out is None:
            out = np.empty(z.shape, dtype=np.float64)
        _vectorize(z, out, fct)
        return out
    else:
        return fct(<double>z)

cdef double _norm1d_pdf(double z):
    return exp(-z*z/2)/S2PI

def norm1d_pdf(np.ndarray z, object out = None):
    return vectorize(z, out, _norm1d_pdf)

cdef double _norm1d_cdf(double z):
    return erf(z/S2) / 2 + 0.5

def norm1d_cdf(np.ndarray z, object out = None):
    return vectorize(z, out, _norm1d_cdf)

cdef double _norm1d_pm1(double z):
    return -exp(-z*z/2) / S2PI

def norm1d_pm1(np.ndarray z, object out = None):
    return vectorize(z, out, _norm1d_pm1)

cdef double _norm1d_pm2(double z):
    if isfinite(z):
        return 0.5*erf(z/S2) + 0.5 - z/S2PI*exp(-z*z/2)
    return 0.5*erf(z/S2)+0.5

def norm1d_pm2(np.ndarray z, object out = None):
    return vectorize(z, out, _norm1d_pm2)

