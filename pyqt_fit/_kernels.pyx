#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
from math cimport isfinite, erf, fabs
from libc.math cimport exp, sqrt, M_PI, pow

np.import_array()

ctypedef np.npy_float64 float64_t
#ctypedef np.npy_float128 float128_t

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
    cdef np.ndarray zz = np.asfarray(z)
    if zz.ndim > 0:
        if out is None:
            out = np.PyArray_EMPTY(zz.ndim, zz.shape, np.NPY_FLOAT64, False)
        _vectorize(<object>zz, out, fct)
        return out
    else:
        return fct(<float64_t>zz)

cdef float64_t _norm1d_pdf(float64_t z):
    return exp(-z*z/2)/S2PI

def norm1d_pdf(object z, object out = None):
    return vectorize(z, out, _norm1d_pdf)

cdef float64_t _norm1d_cdf(float64_t z):
    return erf(z/S2) / 2 + 0.5

def norm1d_cdf(object z, object out = None):
    return vectorize(z, out, _norm1d_cdf)

cdef float64_t _norm1d_pm1(float64_t z):
    return -exp(-z*z/2) / S2PI

def norm1d_pm1(object z, object out = None):
    return vectorize(z, out, _norm1d_pm1)

cdef float64_t _norm1d_pm2(float64_t z):
    if isfinite(z):
        return 0.5*erf(z/S2) + 0.5 - z/S2PI*exp(-z*z/2)
    return 0.5*erf(z/S2)+0.5

def norm1d_pm2(object z, object out = None):
    return vectorize(z, out, _norm1d_pm2)

cdef float64_t tricube_a = sqrt(35./243)
#cdef float128_t tricube_al = sqrtl(35./243)
tricube_width = tricube_a

cdef float64_t _tricube_pdf(float64_t z):
    z *= tricube_a
    if z < -1 or z > 1:
        return 0
    return 70./81*pow(1 - pow(fabs(z), 3.), 3.) * tricube_a

def tricube_pdf(object z, object out = None):
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

def tricube_cdf(object z, object out = None):
    return vectorize(z, out, _tricube_cdf)

cdef float64_t _tricube_pm1(float64_t zc):
    cdef float64_t z = zc
    z *= tricube_a
    if z < -1 or z > 1:
        return 0
    if z < 0:
        z = -z
    cdef float64_t z2 = z*z
    cdef float64_t z3 = z2*z
    cdef float64_t z5 = z3*z2
    cdef float64_t z8 = z5*z3
    cdef float64_t z11 = z8*z3
    return 7./(tricube_a*3564)*(165*z8 - 8.*(5*z11 + 33.*z5) + 220*z2 - 81)

def tricube_pm1(object z, object out = None):
    return vectorize(z, out, _tricube_pm1)

cdef float64_t _tricube_pm2(float64_t z):
    z *= tricube_a
    if z < -1:
        return 0
    if z > 1:
        return 1
    cdef float64_t z3 = z*z*z
    cdef float64_t z6 = z3*z3
    cdef float64_t z9 = z6*z3
    cdef float64_t z12 = z9*z3
    if z > 0:
        return 35./(tricube_a*tricube_a*486)*(4*z9 - (z12 + 6.*z6) + 4*z3 + 1)
    else:
        return 35./(tricube_a*tricube_a*486)*(4*z9 + (z12 + 6.*z6) + 4*z3 + 1)

def tricube_pm2(object z, object out = None):
    return vectorize(z, out, _tricube_pm2)

cdef float64_t epanechnikov_a = 1./sqrt(5.)
epanechnikov_width = epanechnikov_a

cdef float64_t _epanechnikov_pdf(float64_t z):
    z *= epanechnikov_a
    if z < -1:
        return 0
    if z > 1:
        return 0
    return .75*(1-z*z)*epanechnikov_a

def epanechnikov_pdf(object z, object out = None):
    return vectorize(z, out, _epanechnikov_pdf)

cdef float64_t _epanechnikov_cdf(float64_t z):
    z *= epanechnikov_a
    if z < -1:
        return 0
    if z > 1:
        return 1
    return .25*(2+3*z-z*z*z)

def epanechnikov_cdf(object z, object out = None):
    return vectorize(z, out, _epanechnikov_cdf)

cdef float64_t _epanechnikov_pm1(float64_t z):
    z *= epanechnikov_a
    if z < -1:
        return 0
    if z > 1:
        return 0
    cdef float64_t z2 = z*z
    return -3./16.*(1-2*z2+z2*z2)/epanechnikov_a

def epanechnikov_pm1(object z, object out = None):
    return vectorize(z, out, _epanechnikov_pm1)

cdef float64_t _epanechnikov_pm2(float64_t z):
    z *= epanechnikov_a
    if z < -1:
        return 0
    if z > 1:
        return 1
    cdef float64_t z3 = z*z*z
    return 0.25*(2+5*z3-3*z3*z*z)

def epanechnikov_pm2(object z, object out = None):
    return vectorize(z, out, _epanechnikov_pm2)

cdef float64_t _epanechnikov_o4_pdf(float64_t z):
    if z < -1:
        return 0
    if z > 1:
        return 0
    return 0.125*(9-15*z*z)

def epanechnikov_o4_pdf(object z, object out = None):
    return vectorize(z, out, _epanechnikov_o4_pdf)

cdef float64_t _epanechnikov_o4_cdf(float64_t z):
    if z < -1:
        return 0
    if z > 1:
        return 1
    return .125*(4+9*z-5*z*z*z)

def epanechnikov_o4_cdf(object z, object out = None):
    return vectorize(z, out, _epanechnikov_o4_cdf)

cdef float64_t _epanechnikov_o4_pm1(float64_t z):
    if z < -1:
        return 0
    if z > 1:
        return 0
    cdef float64_t z2 = z*z
    return 1./32.*(18*z2-3-15*z2*z2)

def epanechnikov_o4_pm1(object z, object out = None):
    return vectorize(z, out, _epanechnikov_o4_pm1)

cdef float64_t _epanechnikov_o4_pm2(float64_t z):
    if z < -1:
        return 0
    if z > 1:
        return 0
    cdef float64_t z2 = z*z
    cdef float64_t z3 = z2*z
    return .375*(z3 - z2*z3)

def epanechnikov_o4_pm2(object z, object out = None):
    return vectorize(z, out, _epanechnikov_o4_pm2)

cdef float64_t _normal_o4_pdf(float64_t z):
    return (3-z*z)*_norm1d_pdf(z)/2

def normal_o4_pdf(object z, object out = None):
    return vectorize(z, out, _normal_o4_pdf)

cdef float64_t _normal_o4_cdf(float64_t z):
    return _norm1d_cdf(z)+z*_norm1d_pdf(z)/2

def normal_o4_cdf(object z, object out = None):
    return vectorize(z, out, _normal_o4_cdf)

cdef float64_t _normal_o4_pm1(float64_t z):
    return _norm1d_pdf(z) - _normal_o4_pdf(z)

def normal_o4_pm1(object z, object out = None):
    return vectorize(z, out, _normal_o4_pm1)

cdef float64_t _normal_o4_pm2(float64_t z):
    return z*z*z/2*_norm1d_pdf(z)

def normal_o4_pm2(object z, object out = None):
    return vectorize(z, out, _normal_o4_pm2)

