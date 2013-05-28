#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
from math cimport isfinite, powl, expl, sqrtl, lgammal, logl
from libc.math cimport erf, exp, sqrt, M_PI

ctypedef np.npy_float64 float64_t
ctypedef np.npy_float128 float128_t
ctypedef np.npy_int int_t
ctypedef np.npy_int64 int64_t

cdef float128_t M_PIl = 3.1415926535897932384626433832795029

cdef float64_t S2PI = sqrt(2.0*M_PI)
cdef float128_t S2PIl = sqrtl(2.0*M_PIl)
cdef float64_t S2 = sqrt(2.0)
cdef float128_t S2l = sqrtl(2.0)

s2pi = S2PI
s2 = S2

def f_comp_(object I, object a2, float128_t s, float128_t t):
    return np.float128(f_comp(I, a2, s, t))

def lgamma(long double v):
    return lgammal(v)

cdef long double _odd_factorial(int k):
    """
    Odd factorial of (2*k-1) => (2k-1)!! = (2k)!/(2^k k!)
    """
    return expl(lgammal(2.*k+1.)-((logl(2.) * k) + lgammal(k+1.)))

def odd_factorial(int k):
    return _odd_factorial(k)

def ref_odd_factorial(int k):
    cdef int result = 1
    for i from 2 <= i <= k:
        result *= 2*i-1
    return result

cdef float128_t _fixed_point(float64_t t, float128_t M, np.ndarray[float128_t] I, np.ndarray[float128_t] a2):
    cdef int l=7
    cdef int s
    cdef long double K0 = _odd_factorial(l)/S2PIl
    cdef long double cst, time
    cdef float128_t f = f_comp(I, a2, l, t)
    for s from l >= s > 1:
        cst = (1. + powl(0.5, s + .5))/3.0
        time=powl(2*cst*K0/M/f,2./(3.+2.*s))
        f=f_comp(I, a2, s, time)
        K0 /= 2*s-1
    return t-powl(2*M*sqrtl(M_PI)*f,-2./5)

def fixed_point(object t, object M_, object I_, object a2_):
    cdef np.ndarray[float128_t] I = np.float128(I_)
    cdef float128_t M = M_
    cdef np.ndarray[float128_t] a2 = np.float128(a2_)
    return np.float128(_fixed_point(t, M, I, a2))


