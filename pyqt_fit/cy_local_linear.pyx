import numpy as np
cimport numpy as np
from libc.math cimport exp

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef void cy_li(double bw, np.ndarray[DTYPE_t, ndim=1] xdata, np.ndarray[DTYPE_t, ndim=1] ydata, np.ndarray[DTYPE_t, ndim=1] points,
                np.ndarray[DTYPE_t, ndim=1] li2, np.ndarray[DTYPE_t, ndim=1] output):
    cdef unsigned int nx = xdata.shape[0]
    cdef unsigned int npts = points.shape[0]
    cdef unsigned int i,j
    cdef np.ndarray[DTYPE_t, ndim=1] X0 = np.zeros((nx,), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] wi = np.zeros((nx,), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] bi = np.zeros((nx,), dtype=DTYPE)
    cdef double* xp  = <double*>xdata.data
    cdef double* X0p  = <double*>X0.data
    cdef double* yp  = <double*>ydata.data
    cdef double* pp  = <double*>points.data
    cdef double* wip = <double*>wi.data
    cdef double* bip = <double*>bi.data
    cdef double* Op  = <double*>output.data
    cdef double wy, x0, X, X2, lwi, lbi, sbi, li
    cdef double exp_factor = -1.0/(2.0*bw*bw)
    for j from 0 <= j < npts:
        X = 0
        X2 = 0
        for i from 0 <= i < nx:
            x0 = xp[i] - pp[j]
            X0p[i] = x0
            lwi = exp(exp_factor*x0*x0)
            wip[i] = lwi
            X += lwi*x0
            X2 += lwi*x0*x0
        sbi = 0
        for i from 0 <= i < nx:
            lbi = wip[i] * (X2 - X0p[i]*X)
            bip[i] = lbi
            sbi += lbi
        Op[j] = 0
        for i from 0 <= i < nx:
            li = bip[i] / sbi
            li2[j] += li*li
            Op[j] += li * yp[i]

def local_linear_1d(bw, xdata, ydata, points, output = None):
    bw = float(bw)
    xdata = np.ascontiguousarray(xdata, dtype=np.float)
    ydata = np.ascontiguousarray(ydata, dtype=np.float)
    points = np.ascontiguousarray(points, dtype=np.float)
    li2 = np.empty(points.shape, dtype=float)
    if output is None:
        output = np.empty(points.shape, dtype=float)
    else:
        output = np.ascontiguousarray(output, dtype=np.float)
    cy_li(bw, xdata, ydata, points, li2, output)
    return li2, output
