import numpy as np
from .. import kde_methods
from ..utils import namedtuple
from ..compat import irange
from scipy import stats, special
from .. import kernels

def generate(dist, N, low, high):
    start = dist.cdf(low)
    end = dist.cdf(high)
    xs = np.linspace(1-start, 1-end, N)
    return dist.isf(xs)

def setupClass_norm(cls):
    cls.dist = stats.norm(0, 1)
    cls.sizes = np.array([512, 1024, 2048])
    cls.vs = [generate(cls.dist, s, -5, 5) for s in cls.sizes]
    cls.args = {}
    cls.weights = [ cls.dist.pdf(v) for v in cls.vs ]
    cls.lambdas = [ 1 - ws for ws in cls.weights ]
    cls.xs = np.r_[-5:5:1024j]
    cls.lower = -5
    cls.upper = 5

def setupClass_lognorm(cls):
    cls.dist = stats.lognorm(1)
    cls.sizes = np.array([512, 1024, 2048])
    cls.args = {}
    cls.vs = [ generate(cls.dist, s, 0.001, 20) for s in cls.sizes ]
    cls.vs = [ v[v < 20] for v in cls.vs ]
    cls.xs = np.r_[0:20:1024j]
    cls.weights = [ cls.dist.pdf(v) for v in cls.vs ]
    cls.lambdas = [ 1 - ws for ws in cls.weights ]
    cls.lower = 0
    cls.upper = 20

test_method = namedtuple('test_method', ['instance', 'accuracy', 'grid_accuracy', 'bound_low', 'bound_high'])

methods = [ test_method(None, 1e-5, 1e-4, False, False)
          , test_method(kde_methods.reflection, 1e-5, 1e-4, True, True)
          , test_method(kde_methods.cyclic, 1e-5, 1e-4, True, True)
          , test_method(kde_methods.renormalization, 1e-5, 1e-4, True, True)
          , test_method(kde_methods.linear_combination, 1e-1, 1e-1, True, False)
          ]
methods_log = [test_method(kde_methods.transformKDE1D(kde_methods.LogTransform), 1e-5, 1e-4, True, False)]

test_kernel = namedtuple('test_kernel', ['cls', 'precision_factor', 'var', 'positive'])

kernels1d = [ test_kernel(kernels.normal_kernel1d, 1, 1, True)
            , test_kernel(kernels.tricube, 1, 1, True)
            , test_kernel(kernels.Epanechnikov, 1, 1, True)
            , test_kernel(kernels.normal_order4, 10, 0, False) # Bad for precision because of high frequencies
            , test_kernel(kernels.Epanechnikov_order4, 100, 0, False) # Bad for precision because of high frequencies
            ]

class KDETester(object):

    def test_methods(self):
        for m in methods:
            for i in irange(len(self.sizes)):
                yield self.method_works, i, m

    def test_grid_methods(self):
        for m in methods:
            for i in irange(len(self.sizes)):
                yield self.grid_method_works, i, m

    def test_weights_methods(self):
        for m in methods:
            for i in irange(len(self.sizes)):
                yield self.weights_method_works, i, m

    def test_weights_grid_methods(self):
        for m in methods:
            for i in irange(len(self.sizes)):
                yield self.weights_grid_method_works, i, m

    def test_lambdas_methods(self):
        for m in methods:
            for i in irange(len(self.sizes)):
                yield self.lambdas_method_works, i, m

    def test_lambdas_grid_methods(self):
        for m in methods:
            for i in irange(len(self.sizes)):
                yield self.lambdas_grid_method_works, i, m

    def kernel_works_(self, k):
        self.kernel_works(k)
        if kernels.HAS_CYTHON:
            try:
                kernels.usePython()
                self.kernel_works(k)
            finally:
                kernels.useCython()

    def test_kernels(self):
        for k in kernels1d:
            yield self.kernel_works_, k

    def grid_kernel_works_(self, k):
        self.grid_kernel_works(k)
        if kernels.HAS_CYTHON:
            try:
                kernels.usePython()
                self.grid_kernel_works(k)
            finally:
                kernels.useCython()

    def test_grid_kernels(self):
        for k in kernels1d:
            yield self.grid_kernel_works_, k

