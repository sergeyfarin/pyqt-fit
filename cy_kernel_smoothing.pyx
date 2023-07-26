#cython: nonecheck=True
##cython: profile=True
#  --> check for None argument and returns an AttributeError
import cython
import numpy as np
from numpy import linalg
cimport numpy as np
from libc.math cimport exp, pow

DTYPE = np.float64
ctypedef np.float_t DTYPE_t
#ctypedef np.ndarray[DTYPE_t, ndim=2] ndarray2d_t
#ctypedef np.ndarray[DTYPE_t, ndim=1] ndarray1d_t

np.import_array()

@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
cdef void point_diff(unsigned int d,
                     unsigned int m,
                     unsigned int i,
                     np.ndarray[DTYPE_t, ndim=2] diff,
                     np.ndarray[DTYPE_t, ndim=2] xdata,
                     np.ndarray[DTYPE_t, ndim=2] points):
    cdef unsigned int j,k
    cdef unsigned int ds0 = diff.strides[0]
    cdef unsigned int ds1 = diff.strides[1]
    cdef unsigned int xs0 = xdata.strides[0]
    cdef unsigned int xs1 = xdata.strides[1]
    cdef unsigned int ps0 = points.strides[0]
    cdef unsigned int ps1 = points.strides[1]
    cdef char* d_data = diff.data
    cdef char* x_data = xdata.data + i*xs1
    cdef char* p_data = points.data
    #for j in range(d):
    #    for k in range(m):
    for j from 0 <= j < d:
        for k from 0 <= k < m:
            (<double*>d_data)[0] = (<double*>x_data)[0] - (<double*>p_data)[0]
            d_data += ds1
            p_data += ps1
        d_data += ds0
        p_data += ps0
        x_data += xs0
            #diff[j,k] = xdata[j,i] - points[j,k]
    return

@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
cdef void apply_energy(unsigned int d,
                       unsigned int m,
                       np.ndarray[DTYPE_t, ndim=1] result,
                       np.ndarray[DTYPE_t, ndim=2] diff,
                       np.ndarray[DTYPE_t, ndim=2] tdiff):
    #result += np.exp(-np.sum(diff*tdiff, axis=0)/2.0)
    cdef unsigned int j,k
    cdef double energy
    cdef unsigned int r_stride = result.strides[0]
    cdef unsigned int diff_s0 = diff.strides[0]
    cdef unsigned int diff_s1 = diff.strides[1]
    cdef unsigned int tdiff_s0 = tdiff.strides[0]
    cdef unsigned int tdiff_s1 = tdiff.strides[1]
    cdef char* r_data = result.data
    cdef char* d_data = diff.data
    cdef char* t_data = tdiff.data
    cdef char *d_it, *t_it
    for j from 0 <= j < m:
        energy = 0.0
        d_it = d_data + j*diff_s1
        t_it = t_data + j*tdiff_s1
        for k from 0 <= k < d:
            energy += (<double*>d_it)[0]*(<double*>t_it)[0]
            d_it += diff_s0
            t_it += tdiff_s0
            #energy += diff[k,j]*tdiff[k,j]
        energy /= 2.0
        (<double*>r_data)[0] += exp(-energy)
        r_data += r_stride
        #result[j] += exp(-energy)
    return

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
            point_diff(d,m,i,diff,xdata,points)
            #tdiff = np.dot(inv_cov, diff)
            tdiff = np.dot(inv_cov, diff)
            apply_energy(d,m,result,diff,tdiff)
            #result += np.exp(-np.sum(diff*tdiff, axis=0)/2.0)

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
        cdef np.ndarray[DTYPE_t, ndim=2] tdiff
        cdef np.ndarray[DTYPE_t, ndim=1] energy
        #cdef double energy
        cdef np.ndarray[DTYPE_t, ndim=2] inv_cov = self.inv_cov
        cdef np.ndarray[DTYPE_t, ndim=2] xdata = self.xdata
        cdef np.ndarray[DTYPE_t, ndim=1] ydata = self.ydata

        for i in range(n):
            #diff = xdata[:,i,np.newaxis] - points
            point_diff(d,m,i,diff,xdata,points)
            tdiff = np.dot(inv_cov, diff)
            energy = np.exp(-np.sum(diff*tdiff, axis=0)/2.0)
            result += ydata[i]*energy
            norm_res += energy

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

