import numpy as np
cimport numpy as np
from libc.math cimport pow, exp, log

cdef extern from "math.h" nogil:
    double HUGE_VAL

DTYPE=np.float
ctypedef np.float_t DTYPE_t

from numpy import argsort

def linear(np.ndarray[DTYPE_t, ndim=1] x, ps):
    """
    Function: y = a x + b
    Parameters: a b
    Name: Linear
    ParametersEstimate: linearParams
    """
    cdef double a = ps[0]
    cdef double b = ps[1]
    cdef unsigned int s = x.shape[0]
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros((s,), dtype=DTYPE)
    for i in range(s):
        result[i] = a*x[i] + b
    return result

def linearParams(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y):
    b = y.min()
    a = (y.max() - y.min())/(x.max() - x.min())
    return (a,b)

def exponential(np.ndarray[DTYPE_t, ndim=1] x, ps):
    """
    Function: y = A e^{k(x-x_0)} + y_0
    Parameters: A k x_0 y_0
    Name: Exponential
    ParametersEstimate: exponentialParams
    """
    cdef double A = ps[0]
    cdef double k = ps[1]
    cdef double x0 = ps[2]
    cdef double y0 = ps[3]
    cdef unsigned int s = x.shape[0]
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros((s,), dtype=DTYPE)
    for i in range(s):
        result[i] = A*exp(k*(x[i]-x0))+y0
    return result

def exponentialParams(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y):
    x0 = (x.max() + x.min())/2
    IX = argsort(x)
    xs = x[IX]
    ys = y[IX]
    k = log(ys[-1]/ys[0])/(xs[-1]-xs[0])
    A = ys[-1]/(exp(k*(xs[-1]-x0)))
    return (A,k,x0,ys[0])

def power_law(np.ndarray[DTYPE_t, ndim=1] x, ps):
    """
    Function: y = A (x-x_0)^k + y_0
    Name: Power law
    Parameters: A x_0 k y_0
    ParametersEstimate: power_lawParams
    """
    cdef double A = ps[0]
    cdef double x0 = ps[1]
    cdef double k = ps[2]
    cdef double y0 = ps[3]
    cdef unsigned int s = x.shape[0]
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros((s,), dtype=DTYPE)
    for i in range(s):
        result[i] = A*pow(x[i]-x0,k) + y0
    return result

def power_lawParams(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y):
    x0 = x.min()
    y0 = y.min()
    A = (y.max()-y0)/(x.max()-x0)
    return (A, x0, 1, y0)

def logistic(np.ndarray[DTYPE_t, ndim=1] x, ps):
    """
    Function: y = A / (1 + e^{-k (x-x_0)}) + y_0
    Parameters: A k x_0 y_0
    Name: Logistic
    ParametersEstimate: logisticParams
    """
    cdef double A = ps[0]
    cdef double k = ps[1]
    cdef double x0 = ps[2]
    cdef double y0 = ps[3]
    cdef unsigned int s = x.shape[0]
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros((s,), dtype=DTYPE)
    for i in range(s):
        result[i] = A/(1.+exp(k*(x0-x[i]))) + y0
    return result

def logisticParams(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y):
    x0 = (x.max()+x.min())/2
    k = log(y.min())/(x.min()-x0)
    return (y.max(), k, x0, y.min())

