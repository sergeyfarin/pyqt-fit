from scipy.stats import gaussian_kde
from numpy import atleast_2d, atleast_1d, zeros, newaxis, dot, sum, exp, broadcast, asarray, var, power, sqrt, divide
import cyth
import cy_local_linear

class SpatialAverage(object):
    def __init__(self, xdata, ydata):
        self.xdata = atleast_2d(xdata)
        self.ydata = ydata
        kde = gaussian_kde(xdata)
        self.covariance = kde.covariance
        self.inv_cov = kde.inv_cov
        self.d, self.n = self.xdata.shape
        self.correction = 1.

    def evaluate(self, points, result = None):
        points = atleast_2d(points).astype(self.xdata.dtype)
        #norm = self.kde(points)
        d, m = points.shape
        if result is None:
            result = zeros((m,), points.dtype)
        norm = zeros((m,), points.dtype)

        # iterate on the internal points
        for i,ci in broadcast(xrange(self.n), xrange(self._correction.shape[0])):
            diff = dot(self._correction[ci], self.xdata[:,i,newaxis] - points)
            tdiff = dot(self.inv_cov, diff)
            energy = exp(-sum(diff*tdiff,axis=0)/2.0)
            result += self.ydata[i]*energy
            norm += energy

        result[norm>1e-50] /= norm[norm>1e-50]

        return result

    def __call__(self, new_x):
        return self.evaluate(new_x)

    def set_correction(self, value):
        self._correction = atleast_1d(value)

    def get_correction(self):
        return self._correction

    correction = property(get_correction, set_correction)

    def set_density_correction(self):
        dens = self.kde(self.xdata)
        dm = dens.max()
        dens[dens < 1e-50] = dm
        self._correction = dm/dens

class LocalLinearKernel1D(object):
    def __init__(self, xdata, ydata):
        self.xdata = atleast_1d(xdata)
        self.ydata = atleast_1d(ydata)
        self.n = xdata.shape[0]
        self.compute_bandwidth()

    def evaluate(self, points, output=None):
        li2, output = cy_local_linear.cy_local_linear_1d(self.bandwidth, self.xdata, self.ydata, points, output)
        self.li2 = li2
        return output
        #points = atleast_1d(points).astype(self.xdata.dtype)
        #m = points.shape[0]
        #x0 = points - self.xdata[:,newaxis]
        #x02 = x0*x0
        #wi = exp(-self.inv_cov*x02/2.0)
        #X = sum(wi*x0, axis=0)
        #X2 = sum(wi*x02, axis=0)
        #wy = wi*self.ydata[:,newaxis]
        #Y = sum(wy, axis=0)
        #Y2 = sum(wy*x0, axis=0)
        #W = sum(wi, axis=0)
        #return divide(X2*Y-Y2*X, W*X2-X*X, output)

    def variance_bandwidth(self, factor):
        self.factor = factor
        self.sq_bandwidth = var(self.xdata)*self.factor*self.factor
        self.bandwidth = sqrt(self.sq_bandwidth)
        self.inv_cov = 1/self.sq_bandwidth

    def silverman_bandwidth(self):
        self.variance_bandwidth(power(0.75*self.n, -0.2))

    def scotts_bandwidth(self):
        self.variance_bandwidth(power(self.n, -0.2))

    compute_bandwidth = scotts_bandwidth

    __call__ = evaluate

