import numpy as np
cimport numpy as np
from libc.math cimport exp, log

cdef extern from "math.h" nogil:
    double HUGE_VAL

DTYPE=np.float
ctypedef np.float_t DTYPE_t

from numpy import argsort


def standard(np.ndarray[DTYPE_t, ndim=1] y1, np.ndarray[DTYPE_t, ndim=1] y0):
    """
    Name: Standard
    Formula: y_1 - y_0
    Invert: add_standard
    """
    cdef unsigned int s = y1.shape[0]
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros((s,), dtype=DTYPE)
    for i in range(s):
        result[i] = y1[i]-y0[i]
    return result

def add_standard(np.ndarray y, np.ndarray res):
    """
    Add the residual to the value
    """
    return y+res

def log_residual(np.ndarray[DTYPE_t, ndim=1] y1, np.ndarray[DTYPE_t, ndim=1] y0):
    """
    Name: Difference of the logs
    Formula: log(y1/y0)
    Invert: add_log_residual
    """
    cdef unsigned int s = y1.shape[0]
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros((s,), dtype=DTYPE)
    for i in range(s):
        result[i] = log(y1[i]/y0[i])
    return result

def add_log_residual(np.ndarray y, np.ndarray res):
    """
    Multiply the value by the exponential of the residual
    """
    return y*np.exp(res)
