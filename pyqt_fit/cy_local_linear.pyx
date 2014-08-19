cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp

DTYPE = np.float
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cy_li(double bw, np.ndarray[DTYPE_t, ndim=1] xdata, np.ndarray[DTYPE_t, ndim=1] ydata,
           np.ndarray[DTYPE_t, ndim=1] points,
           np.ndarray[DTYPE_t, ndim=1] li2,
           np.ndarray[DTYPE_t, ndim=1] out):
    cdef:
         unsigned int nx = xdata.shape[0]
         unsigned int npts = points.shape[0]
         unsigned int i, j
         np.ndarray[DTYPE_t, ndim=1] X0 = np.zeros((nx,), dtype=DTYPE)
         np.ndarray[DTYPE_t, ndim=1] wi = np.zeros((nx,), dtype=DTYPE)
         np.ndarray[DTYPE_t, ndim=1] bi = np.zeros((nx,), dtype=DTYPE)
         double x0, X, X2, lwi, lbi, sbi, li
         double exp_factor = -1.0/(2.0*bw*bw)

    for j in range(npts):
        it_in = np.broadcast(xdata, X0, wi)
        X = 0
        X2 = 0
        for i in range(nx):
            x0 = xdata[i] - points[j]
            X0[i] = x0
            lwi = exp(exp_factor*x0*x0)
            wi[i] = lwi
            X += lwi*x0
            X2 += lwi*x0*x0

        sbi = 0
        for i in range(nx):
            lbi = wi[i] * (X2 - X0[i]*X)
            bi[i] = lbi
            sbi += lbi

        out[j] = 0
        li2[i] = 0
        if sbi != 0: # The total weight is 0 only if all weights are 0
            for i in range(nx):
                li = bi[i] / sbi
                li2[j] += li * li
                out[j] += li * ydata[i]

@cython.embedsignature(True)
def local_linear_1d(bw, xdata, ydata, points, kernel, out):
    bw = float(bw)
    li2 = np.empty(points.shape, dtype=float)
    cy_li(bw, xdata, ydata, points, li2, out)
    return li2, out
