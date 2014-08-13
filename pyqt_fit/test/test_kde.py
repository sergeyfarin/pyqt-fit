from __future__ import division, absolute_import, print_function

from .. import kde
from .. import kde_methods
import numpy as np
from numpy.random import randn
from scipy import integrate
from . import kde_utils

class TestBandwidth(object):
    @classmethod
    def setUpClass(cls):
        cls.ratios = np.array([1., 2., 5.])
        d = randn(500)
        cls.vs = cls.ratios[:, np.newaxis]*np.array([d, d, d])
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

    #def test_converge(self):
        #xs = np.r_[-3:3:512j]
        #ys = self.dist.pdf(xs)
        #ks = [ self.createKDE(v, **self.args) for v in self.vs ]

    def method_works(self, k, method, name):
        k.fit()
        tot = integrate.quad(k.pdf, k.lower, k.upper, limit=100)[0]
        acc = method.normed_accuracy
        assert abs(tot - 1) < acc, "Error, {} should be close to 1".format(tot)
        del k.weights
        del k.lambdas
        k.fit()
        assert k.total_weights == len(k.xdata)
        assert k.lambdas == 1.

    def grid_method_works(self, k, method, name):
        xs, ys = k.grid(4000)
        tot = integrate.simps(ys, xs)
        acc = max(method.normed_accuracy, method.grid_accuracy)
        assert abs(tot - 1) < acc, "Error, {} should be close to 1".format(tot)

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

    def kernel_works(self, ker, name):
        method = self.methods[0]
        k = self.createKDE(self.vs[1], method)
        k.kernel = ker.cls()
        tot = integrate.quad(k.pdf, k.lower, k.upper, limit=100)[0]
        acc = method.normed_accuracy * ker.precision_factor
        assert abs(tot - 1) < acc, "Error, {} should be close to 1".format(tot)

    def grid_kernel_works(self, ker, name):
        method = self.methods[0]
        k = self.createKDE(self.vs[1], method)
        xs, ys = k.grid()
        tot = integrate.simps(ys, xs)
        acc = max(method.grid_accuracy, method.normed_accuracy) * ker.precision_factor
        assert abs(tot - 1) < acc, "Error, {} should be close to 1".format(tot)

class LogTestKDE1D(TestKDE1D):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        cls.methods = kde_utils.methods_log

class TestSF(kde_utils.KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
        cls.methods = kde_utils.methods
        del cls.sizes[1:]

    def method_works(self, k, method, name):
        k.fit()
        xs = kde_methods.generate_grid(k)
        sf = k.sf(xs)
        cdf = k.cdf(xs)
        np.testing.assert_allclose(sf, 1-cdf, method.accuracy, method.accuracy)

    def grid_method_works(self, k, method, name):
        xs, sf = k.sf_grid()
        _, cdf = k.cdf_grid()
        np.testing.assert_allclose(sf, 1-cdf, method.accuracy, method.accuracy)

    def kernel_works(self, ker, name):
        pass

    def grid_kernel_works(self, ker, name):
        pass

class TestLogSF(TestSF):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        cls.methods = kde_utils.methods_log
        del cls.sizes[1:]

class TestISF(kde_utils.KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
        cls.methods = kde_utils.methods
        del cls.sizes[1:]

    def method_works(self, k, method, name):
        sf = np.linspace(0, 1, 64)
        sf_xs = k.isf(sf)
        cdf_xs = k.icdf(1-sf)
        acc = max(method.accuracy, method.normed_accuracy)
        np.testing.assert_allclose(sf_xs, cdf_xs, acc, acc)

    def grid_method_works(self, k, method, name):
        comp_sf, xs = k.isf_grid()
        ref_sf = k.sf(xs)
        acc = max(method.grid_accuracy, method.normed_accuracy)
        np.testing.assert_allclose(comp_sf, ref_sf, acc, acc)

    def kernel_works(self, ker, name):
        pass

    def grid_kernel_works(self, ker, name):
        pass

class TestLogISF(TestISF):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        del cls.sizes[1:]

class TestICDF(kde_utils.KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
        cls.methods = kde_utils.methods
        del cls.sizes[1:]

    def method_works(self, k, method, name):
        quant = np.linspace(0, 1, 64)
        xs = k.icdf(quant)
        cdf_quant = k.cdf(xs)
        acc = max(method.accuracy, method.normed_accuracy)
        np.testing.assert_allclose(cdf_quant, quant,  acc, acc)

    def grid_method_works(self, k, method, name):
        comp_cdf, xs = k.icdf_grid()
        ref_cdf = k.cdf(xs)
        acc = max(method.grid_accuracy, method.normed_accuracy)
        np.testing.assert_allclose(comp_cdf, ref_cdf, acc, acc)

    def kernel_works(self, ker, name):
        pass

    def grid_kernel_works(self, ker, name):
        pass

class TestLogICDF(TestICDF):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        cls.methods = kde_utils.methods_log
        del cls.sizes[1:]


class TestHazard(kde_utils.KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
        cls.methods = kde_utils.methods
        del cls.sizes[1:]

    def method_works(self, k, method, name):
        k.fit()
        xs = kde_methods.generate_grid(k)
        h_comp = k.hazard(xs)
        sf = k.sf(xs)
        h_ref = k.pdf(xs)
        sf = k.sf(xs)
        sf[sf < 0] = 0 # Some methods can produce negative sf
        h_ref /= sf
        sel = sf > np.sqrt(method.accuracy)
        np.testing.assert_allclose(h_comp[sel], h_ref[sel], method.accuracy, method.accuracy)

    def grid_method_works(self, k, method, name):
        xs, h_comp = k.hazard_grid()
        xs, sf = k.sf_grid()
        sf[sf < 0] = 0 # Some methods can produce negative sf
        h_ref = k.grid()[1]
        h_ref /= sf
        sel = sf > np.sqrt(method.accuracy)
        # Only tests for sf big enough or error is too large
        np.testing.assert_allclose(h_comp[sel], h_ref[sel], method.accuracy, method.accuracy)

    def kernel_works(self, ker, name):
        pass

    def grid_kernel_works(self, ker, name):
        pass

class TestLogHazard(TestHazard):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        del cls.sizes[1:]
        cls.methods = kde_utils.methods_log

class TestCumHazard(kde_utils.KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)
        cls.methods = kde_utils.methods
        del cls.sizes[1:]

    def method_works(self, k, method, name):
        k.fit()
        xs = kde_methods.generate_grid(k)
        h_comp = k.cumhazard(xs)
        sf = k.sf(xs)
        sf[sf < 0] = 0 # Some methods can produce negative sf
        h_ref = -np.log(sf)
        sel = sf > np.sqrt(method.accuracy)
        np.testing.assert_allclose(h_comp[sel], h_ref[sel], method.accuracy, method.accuracy)

    def grid_method_works(self, k, method, name):
        xs, h_comp = k.cumhazard_grid()
        xs, sf = k.sf_grid()
        sf[sf < 0] = 0 # Some methods can produce negative sf
        h_ref = -np.log(sf)
        sel = sf > np.sqrt(method.accuracy)
        # Only tests for sf big enough or error is too large
        np.testing.assert_allclose(h_comp[sel], h_ref[sel], method.accuracy, method.accuracy)

    def kernel_works(self, ker, name):
        pass

    def grid_kernel_works(self, ker, name):
        pass

class TestLogCumHazard(TestCumHazard):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)
        del cls.sizes[1:]
        cls.methods = kde_utils.methods_log


