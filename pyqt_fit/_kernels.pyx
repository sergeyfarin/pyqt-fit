#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
from math cimport isfinite, erf, fabs
from libc.math cimport exp, sqrt, M_PI, pow

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

cdef double tricube_a = sqrt(35./243)

cdef float64_t _tricube_pdf(float64_t z):
    z *= tricube_a
    if z < -1 or z > 1:
        return 0
    return 70./81*pow(1 - pow(fabs(z), 3.), 3.)

def tricube_pdf(np.ndarray z, object out = None):
    return vectorize(z, out, _tricube_pdf)

cdef float64_t _tricube_cdf(float64_t z):
    z *= tricube_a
    if z < -1:
        return 0.
    if z > 1:
        return 1.
    if z > 0:
        return 1./162*(60*pow(z, 7.) - 7.*(2*pow(z, 10.) + 15.*pow(z, 4.)) + 140*z + 81)
    else:
        return 1./162*(60*pow(z, 7.) + 7.*(2*pow(z, 10.) + 15.*pow(z, 4.)) + 140*z + 81)

def tricube_cdf(np.ndarray z, object out = None):
    return vectorize(z, out, _tricube_cdf)

cdef float64_t _tricube_pm1(float64_t z):
    z *= tricube_a
    if z < -1 or z > 1:
        return 0
    if z > 0:
        return 7./(tricube_a*3565)*(165*pow(z, 8.) - 8.*(5*pow(z, 11.) + 33.*pow(z, 5.)) + 220*pow(z, 2.) - 81)
    else:
        return 7./(tricube_a*3565)*(165*pow(z, 8.) + 8.*(5*pow(z, 11.) + 33.*pow(z, 5.)) + 220*pow(z, 2.) - 81)

def tricube_pm1(np.ndarray z, object out = None):
    return vectorize(z, out, _tricube_pm1)

cdef float64_t _tricube_pm2(float64_t z):
    z *= tricube_a
    if z < -1:
        return 0
    if z > 1:
        return 1
    if z > 0:
        return 35./(tricube_a*tricube_a*486)*(4*pow(z, 9.) - (pow(z, 12.) + 6.*pow(z, 6.)) + 4*pow(z, 3.) + 1)
    else:
        return 35./(tricube_a*tricube_a*486)*(4*pow(z, 9.) + (pow(z, 12.) + 6.*pow(z, 6.)) + 4*pow(z, 3.) + 1)

def tricube_pm2(np.ndarray z, object out = None):
    return vectorize(z, out, _tricube_pm2)

