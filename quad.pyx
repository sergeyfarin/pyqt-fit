import cython
import numpy as np
from numpy import linalg
cimport numpy as np

DTYPE = np.float64
ctypedef np.float_t DTYPE_t

def quadratic(np.ndarray[DTYPE_t, ndim=1] x, ps):
    cdef double p0 = ps[0]
    cdef double p1 = ps[1]
    cdef double p2 = ps[2]
    cdef unsigned int s = x.shape[0]
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros((s,), dtype=DTYPE)
    for i in range(s):
        result[i] = p0 + p1*x[i] + p2*x[i]*x[i]
    return result

