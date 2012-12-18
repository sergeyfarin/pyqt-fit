from scipy.stats import gaussian_kde
from numpy import atleast_2d, atleast_1d, zeros, newaxis, dot, sum, exp, broadcast, asarray

class SpatialAverage(object):
    def __init__(self, xdata, ydata):
        self.xdata = atleast_2d(xdata)
        self.ydata = ydata
        self.kde = gaussian_kde(xdata)
        self.d, self.n = self.xdata.shape
        self.correction = 1.

    def evaluate(self, points):
        points = atleast_2d(points).astype(self.xdata.dtype)
        #norm = self.kde(points)
        d, m = points.shape
        result = zeros((m,), points.dtype)
        norm = zeros((m,), points.dtype)

        # iterate on the internal points
        for i,ci in broadcast(xrange(self.n), xrange(self._correction.shape[0])):
            diff = dot(self._correction[ci], self.xdata[:,i,newaxis] - points)
            tdiff = dot(self.kde.inv_cov, diff)
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
        self.xdata = asarray(xdata)
        self.ydata = asarray(ydata)
        kde = gaussian_kde(xdata)
        self.inv_cov = kde.inv_cov[0,0]
        self.n = xdata.shape[0]

    def evaluate(self, points):
        points = asarray(points, dtype=self.xdata.dtype)
        m = points.shape[0]
        x0 = points - self.xdata[:,newaxis]
        x02 = x0*x0
        wi = exp(-self.inv_cov*x02/2.0)
        X = sum(wi*x0, axis=0)
        X2 = sum(wi*x02, axis=0)
        wy = wi*self.ydata[:,newaxis]
        Y = sum(wy, axis=0)
        Y2 = sum(wy*x0, axis=0)
        W = sum(wi, axis=0)
        return (X2*Y-Y2*X)/(W*X2-X**2)

    __call__ = evaluate

