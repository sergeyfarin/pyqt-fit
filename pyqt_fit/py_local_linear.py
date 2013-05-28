from __future__ import division, absolute_import, print_function
import numpy as np

def local_linear_1d(bw, xdata, ydata, points, output = None):
    points = np.atleast_1d(points).astype(self.xdata.dtype)
    m = points.shape[0]
    x0 = points - self.xdata[:,np.newaxis]
    x02 = x0*x0
    wi = np.exp(-self.inv_cov*x02/2.0)
    X = np.sum(wi*x0, axis=0)
    X2 = np.sum(wi*x02, axis=0)
    wy = wi*self.ydata[:,np.newaxis]
    Y = np.sum(wy, axis=0)
    Y2 = np.sum(wy*x0, axis=0)
    W = np.sum(wi, axis=0)
    return None, np.divide(X2*Y-Y2*X, W*X2-X*X, output)
