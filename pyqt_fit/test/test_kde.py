from __future__ import division, absolute_import, print_function

from .. import kde
from .. import kde_methods
from .. import kernels
import numpy as np
from numpy import newaxis
from numpy.random import randn
from scipy import integrate
from ..compat import irange
from . import kde_utils

class TestBandwidth(object):
    @classmethod
    def setUpClass(cls):
        cls.ratios = np.array([1., 2., 5.])
        d = randn(500)
        cls.vs = cls.ratios[:, newaxis]*np.array([d, d, d])
        cls.ss = np.var(cls.vs, axis=1)

    def variance_methods(self, m):
        bws = np.asfarray([m(v) for v in self.vs])
        assert bws.shape == (3, 1, 1)
        rati = bws[:, 0, 0] / self.ss
        assert sum((rati - rati[0])**2) < 1e-6
        rati = bws[:, 0, 0] / bws[0, 0, 0]
        assert sum((rati - self.ratios**2)**2) < 1e-6

    def test_variance_methods(self):
        yield self.variance_methods, kde.silverman_covariance
        yield self.variance_methods, kde.scotts_covariance

    def test_botev(self):
        class FakeModel(object):
            lower = -np.inf
            upper = np.inf
            weights = np.asarray(1.)
        bws = np.array([kde.botev_bandwidth()(v, model=FakeModel()) for v in self.vs])
        assert bws.shape == (3,)
        rati = bws**2 / self.ss
        assert sum((rati - rati[0])**2) < 1e-6
        rati = bws / bws[0]
        assert sum((rati - self.ratios)**2) < 1e-6


class TestKDE1D(kde_utils.KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
        cls.methods = kde_utils.methods

    def createKDE(self, data, method, **args):
        all_args = dict(self.args)
        all_args.update(args)
        k = kde.KDE1D(data, **all_args)
        if method.instance is None:
            del k.method
        else:
            k.method = method.instance
        if method.bound_low:
            k.lower = self.lower
        else:
            del k.lower
        if method.bound_high:
            k.upper = self.upper
        else:
            del k.upper
        assert k.fitted is False
        return k

    #def test_converge(self):
        #xs = np.r_[-3:3:512j]
        #ys = self.dist.pdf(xs)
        #ks = [ self.createKDE(v, **self.args) for v in self.vs ]

    def method_works(self, i, method):
        k = self.createKDE(self.vs[i], method, **self.args)
        k.fit()
        tot = integrate.quad(k.pdf, k.lower, k.upper, limit=100)[0]
        assert abs(tot - 1) < method.accuracy, "Error, {} should be close to 1".format(tot)

    def grid_method_works(self, i, method):
        k = self.createKDE(self.vs[i], method, **self.args)
        xs, ys = k.grid(4000)
        tot = integrate.simps(ys, xs)
        assert abs(tot - 1) < method.grid_accuracy, "Error, {} should be close to 1".format(tot)

    def weights_method_works(self, i, method):
        weights = self.weights[i]
        k = self.createKDE(self.vs[i], method, weights=weights, **self.args)
        tot = integrate.quad(k.pdf, k.lower, k.upper, limit=100)[0]
        assert abs(tot - 1) < method.accuracy, "Error, {} should be close to 1".format(tot)
        del k.weights
        k.fit()
        assert k.total_weights == len(self.vs[i])

    def weights_grid_method_works(self, i, method):
        weights = self.weights[i]
        k = self.createKDE(self.vs[i], method, weights=weights, **self.args)
        xs, ys = k.grid(2048)
        tot = integrate.simps(ys, xs)
        assert abs(tot - 1) < method.grid_accuracy, "Error, {} should be close to 1".format(tot)

    def lambdas_method_works(self, i, method):
        lambdas = self.lambdas[i]
        k = self.createKDE(self.vs[i], method, lambdas=lambdas, **self.args)
        tot = integrate.quad(k.pdf, k.lower, k.upper, limit=100)[0]
        assert abs(tot - 1) < method.accuracy, "Error, {} should be close to 1".format(tot)
        del k.lambdas
        k.fit()
        assert k.lambdas == 1

    def lambdas_grid_method_works(self, i, method):
        lambdas = self.lambdas[i]
        k = self.createKDE(self.vs[i], method, lambdas=lambdas, **self.args)
        xs, ys = k.grid(2048)
        tot = integrate.simps(ys, xs)
        assert abs(tot - 1) < method.accuracy, "Error, {} should be close to 1".format(tot)

    def test_copy(self):
        k = self.createKDE(self.vs[0], self.methods[0])
        k.covariance = kde.silverman_covariance
        xs = np.r_[self.xs.min():self.xs.max():512j]
        ys = k(xs)
        k1 = k.copy()
        ys1 = k1(xs)
        np.testing.assert_allclose(ys1, ys, 1e-8, 1e-8)

    def test_bandwidths(self):
        k = self.createKDE(self.vs[0], self.methods[0])
        k.covariance = kde.silverman_covariance
        assert k.fitted is not None
        k.fit()
        k.covariance = 0.01
        assert k.fitted is not None
        k.fit()
        np.testing.assert_almost_equal(k.bandwidth, 0.1)
        k.bandwidth = 0.1
        assert k.fitted is not None
        k.fit()
        np.testing.assert_almost_equal(k.covariance, 0.01)
        k.bandwidth = kde.botev_bandwidth()
        assert k.fitted is not None
        k.fit()

    def kernel_works(self, ker):
        method = self.methods[0]
        k = self.createKDE(self.vs[1], method)
        k.kernel = ker.cls()
        tot = integrate.quad(k.pdf, k.lower, k.upper, limit=100)[0]
        acc = method.grid_accuracy * ker.precision_factor
        assert abs(tot - 1) < acc, "Error, {} should be close to 1".format(tot)

    def grid_kernel_works(self, ker):
        method = self.methods[0]
        k = self.createKDE(self.vs[1], method)
        xs, ys = k.grid()
        tot = integrate.simps(ys, xs)
        acc = method.grid_accuracy * ker.precision_factor
        assert abs(tot - 1) < acc, "Error, {} should be close to 1".format(tot)

class LogTestKDE1D(kde_utils.KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        cls.methods = kde_utils.methods_log

