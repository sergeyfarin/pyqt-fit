#cython: nonecheck=True
#  --> check for None argument and returns an AttributeError
import cython
import numpy as np
from numpy import linalg
cimport numpy as np
from libc.math cimport exp, pow

DTYPE = np.float
ctypedef np.float_t DTYPE_t
#ctypedef np.ndarray[DTYPE_t, ndim=2] ndarray2d_t
#ctypedef np.ndarray[DTYPE_t, ndim=1] ndarray1d_t

cdef class SpatialAverage:
    cdef public np.ndarray xdata
    cdef public np.ndarray ydata
    cdef public unsigned int d, n
    cdef public double factor
    cdef public np.ndarray covariance
    cdef public np.ndarray inv_cov
    cdef public double _norm_factor

    def __init__(self, np.ndarray xdata, np.ndarray[DTYPE_t, ndim=1] ydata):
        self.xdata = np.atleast_2d(xdata).astype(float)
        self.ydata = ydata
        self.d = self.xdata.shape[0]
        self.n = self.xdata.shape[1]
        self._compute_covariance()

    cdef double scotts_factor(self):
        return pow(<double>self.n, -1./(<double>self.d+4.0))

    cdef double silverman_factor(self):
        return pow(<double>self.n*(<double>self.d+2.0)/4.0, -1./(<double>self.d+4.0))

    #covariance_factor = scotts_factor

    cpdef _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor
        """
        self.factor = self.scotts_factor() # self.covariance_factor()
        self.covariance = np.atleast_2d(np.cov(self.xdata, rowvar=1, bias=False) *
            self.factor * self.factor)
        self.inv_cov = linalg.inv(self.covariance)
        self._norm_factor = np.sqrt(linalg.det(2*np.pi*self.covariance)) * self.n

    cdef dens_evaluate(self, np.ndarray[DTYPE_t, ndim=2] points):
        if points.shape[0] != self.d:
            raise ValueError("Error, `points` must have as many dimension as the input data set")
        cdef unsigned int m = points.shape[1]
        cdef unsigned int d = self.d
        cdef unsigned int n = self.n
        cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros((m,), dtype=float)
        cdef np.ndarray[DTYPE_t, ndim=1] norm_res = np.zeros((m,), dtype=float)
        cdef unsigned int i, j, k, l
        cdef np.ndarray[DTYPE_t, ndim=2] diff = np.zeros((d,m), dtype=float)
        cdef np.ndarray[DTYPE_t, ndim=2] tdiff = np.zeros((d,m), dtype=float)
        #cdef np.nbarray[DTYPE_t, ndim=1] energy = np.zeros((m,), dtype=float)
        cdef double energy
        cdef np.ndarray[DTYPE_t, ndim=2] inv_cov = self.inv_cov
        cdef np.ndarray[DTYPE_t, ndim=2] xdata = self.xdata

        for i in range(n):
            #diff = xdata[:,i,np.newaxis] - points
            for j in range(d):
                for k in range(m):
                    diff[j,k] = xdata[j,i] - points[j,k]
            #tdiff = np.dot(inv_cov, diff)
            for j in range(d):
                for k in range(m):
                    tdiff[j,k] = 0.0
                    for l in range(d):
                        tdiff[j,k] += inv_cov[j,l]*diff[l,k]
            for j in range(m):
                energy = 0.0
                for k in range(d):
                    energy += diff[k,j]*tdiff[k,j]
                energy /= 2.0
                result[j] += exp(-energy)

        for i in range(m):
            result[i] /= self._norm_factor

        return result

    @cython.boundscheck(False)
    @cython.cdivision(False)
    @cython.wraparound(False)
    cdef evaluate(self, np.ndarray[DTYPE_t, ndim=2] points, double eps = 1e-50):
        if points.shape[0] != self.d:
            raise ValueError("Error, `points` must have as many dimension as the input data set")
        cdef unsigned int m = points.shape[1]
        cdef unsigned int d = self.d
        cdef unsigned int n = self.n
        cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros((m,), dtype=float)
        cdef np.ndarray[DTYPE_t, ndim=1] norm_res = np.zeros((m,), dtype=float)
        cdef unsigned int i, j, k, l
        cdef np.ndarray[DTYPE_t, ndim=2] diff = np.zeros((d,m), dtype=float)
        cdef np.ndarray[DTYPE_t, ndim=2] tdiff = np.zeros((d,m), dtype=float)
        #cdef np.nbarray[DTYPE_t, ndim=1] energy = np.zeros((m,), dtype=float)
        cdef double energy
        cdef np.ndarray[DTYPE_t, ndim=2] inv_cov = self.inv_cov
        cdef np.ndarray[DTYPE_t, ndim=2] xdata = self.xdata
        cdef np.ndarray[DTYPE_t, ndim=1] ydata = self.ydata

        for i in range(n):
            #diff = xdata[:,i,np.newaxis] - points
            for j in range(d):
                for k in range(m):
                    diff[j,k] = xdata[j,i] - points[j,k]
            tdiff = np.dot(inv_cov, diff)
            for j in range(m):
                energy = 0.0
                for k in range(d):
                    energy += diff[k,j]*tdiff[k,j]
                energy /= 2.0
                result[j] += ydata[i]*exp(-energy)
                norm_res[j] += exp(-energy)

        for i in range(m):
            if norm_res[i] > eps:
                result[i] /= norm_res[i]
            else:
                result[i] = 0.0

        return result

    def density(self, points):
        return self.dens_evaluate(np.atleast_2d(points))

    def __call__(self, points, double eps = 1e-50):
        return self.evaluate(np.atleast_2d(points), eps)

