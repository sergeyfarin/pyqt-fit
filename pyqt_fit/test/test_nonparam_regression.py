from __future__ import division, absolute_import, print_function

from .. import nonparam_regression, npr_methods
import numpy as np

from ..compat import irange, izip

def fct(xs):
    """
    Function to be estimated
    """
    return np.cos(xs) + xs ** 2

methods = [ npr_methods.SpatialAverage,
            npr_methods.LocalLinearKernel1D,
            npr_methods.LocalPolynomialKernel1D ]

methods_args = [dict(q=0),
                dict(q=1),
                dict(q=2),
                ]


class TestConvergence1D(object):
    @classmethod
    def setupClass(cls):
        cls.xx = np.r_[0:3:256j]
        cls.yy = fct(cls.xx)
        cls.nb_samples = 100
        sizes = [2 ** i for i in range(5, 8)]
        cls.sizes = sizes
        xs = [ np.tile(np.linspace(0.01, 3, s), cls.nb_samples) for s in sizes ]
        ys = [fct(x) for x in xs]
        cls.xs = [x.reshape((cls.nb_samples, s)) for x, s in izip(xs, sizes)]
        cls.ys = [y.reshape((cls.nb_samples, s)) for y, s in izip(ys, sizes)]

    def make_regressor(self, i, x, y, method, args):
        est = nonparam_regression.NonParamRegression(x[i], y[i])
        est.method = npr_methods.LocalPolynomialKernel(**args)
        est.fit()
        assert isinstance(est.fitted_method, method), "Failed to resolve the correct instance"
        return est

    def convergence(self, method, args):
        diff = np.empty(len(self.xs), dtype=float)
        xx = self.xx
        yy = self.yy
        for i, (x, y) in enumerate(izip(self.xs, self.ys)):
            ests = [ self.make_regressor(i, x, y, method, args) for j in irange(self.nb_samples) ]
            res = [est(xx) for est in ests]
            diff[i] = np.std([(yy - r) ** 2 for r in res])
        assert all(diff[1:] < diff[:-1]), 'Diff is not strictly decreasing: {}'.format(diff)

    def testConvergence(self):
        for m, a in zip(methods, methods_args):
            yield self.convergence, m, a
        npr_methods.usePython()
        yield self.convergence, npr_methods.LocalLinearKernel1D, dict(q=1)
        npr_methods.useCython()

