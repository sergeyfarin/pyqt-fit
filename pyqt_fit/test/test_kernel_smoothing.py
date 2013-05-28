from __future__ import division, absolute_import, print_function

from .. import kernel_smoothing
import numpy as np

from ..compat import irange, izip

def fct(xs):
    return np.cos(xs) + xs**2

methods = [ kernel_smoothing.SpatialAverage, 
            kernel_smoothing.LocalLinearKernel1D,
            kernel_smoothing.LocalPolynomialKernel1D,
            kernel_smoothing.LocalPolynomialKernel1D,
            kernel_smoothing.LocalPolynomialKernel1D,
            kernel_smoothing.LocalPolynomialKernel1D
            ]

methods_args = [ {},
                 {},
                 {'q': 1},
                 {'q': 2},
                 {'q': 3},
                 {'q': 4}
               ]

class TestConvergence1D(object):
    @classmethod
    def setupClass(cls):
        cls.xx = np.r_[0:3:1024j]
        cls.yy = fct(cls.xx)
        cls.xs = [ 3*np.random.rand(2**i) for i in irange(8, 12) ]
        cls.ys = [ fct(x) for x in cls.xs ]

    def convergence(self, method, args):
        mods = [ method(x, y, **args) for (x,y) in izip(self.xs, self.ys) ]
        res = [ m(self.xx) for m in mods ]
        diff = np.array([ sum((self.yy - r)**2) for r in res ])
        assert all(diff[1:] < diff[:-1])

    def testConvergence(self):
        for m,a in zip(methods, methods_args):
            yield self.convergence, m, a
        kernel_smoothing.usePython()
        yield self.convergence, kernel_smoothing.LocalLinearKernel1D, {}
        kernel_smoothing.useCython()
