from __future__ import division, absolute_import, print_function

from .. import kernel_smoothing
import numpy as np

from ..compat import irange, izip


def fct(xs):
    """
    Function to be estimated
    """
    return np.cos(xs) + xs ** 2

methods = [kernel_smoothing.SpatialAverage,
           kernel_smoothing.LocalLinearKernel1D,
           kernel_smoothing.LocalPolynomialKernel1D,
           kernel_smoothing.LocalPolynomialKernel1D
           ]

methods_args = [{},
                {},
                {'q': 1},
                {'q': 2}
                ]


class TestConvergence1D(object):
    @classmethod
    def setupClass(cls):
        cls.xx = np.r_[0:3:256j]
        cls.yy = fct(cls.xx)
        cls.nb_samples = 100
        sizes = [2 ** i for i in range(5, 8)]
        cls.sizes = sizes
        xs = [3 * np.random.rand(cls.nb_samples * s) for s in sizes]
        noise = cls.yy.max() / 10
        ys = [fct(x) + noise * np.random.randn(len(x)) for x in xs]
        cls.xs = [x.reshape((cls.nb_samples, s)) for x, s in izip(xs, sizes)]
        cls.ys = [y.reshape((cls.nb_samples, s)) for y, s in izip(ys, sizes)]

    def convergence(self, method, args):
        # mods = [method(x, y, **args) for (x, y) in izip(self.xs, self.ys)]
        diff = np.empty(len(self.xs), dtype=float)
        xx = self.xx
        yy = self.yy
        for i, (x, y) in enumerate(izip(self.xs, self.ys)):
            res = [method(x[j], y[j], **args)(xx) for j in irange(self.nb_samples)]
            diff[i] = np.std([(yy - r) ** 2 for r in res])
        assert all(diff[1:] < diff[:-1]), 'Diff is not strictly decreasing: {}'.format(diff)

    def testConvergence(self):
        for m, a in zip(methods, methods_args):
            yield self.convergence, m, a
        kernel_smoothing.usePython()
        yield self.convergence, kernel_smoothing.LocalLinearKernel1D, {}
        kernel_smoothing.useCython()
