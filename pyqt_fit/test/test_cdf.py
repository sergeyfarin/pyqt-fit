from __future__ import division, absolute_import, print_function

from .. import kde
from .. import kde_methods
import numpy as np
from numpy import newaxis
from numpy.random import randn
from scipy import stats, integrate, interpolate
from ..compat import irange

class TestUnboundedCDF(object):
    @classmethod
    def setUpClass(cls):
        cls.dist = stats.lognorm(1)
        cls.sizes = np.r_[1000:5000:5j]
        cls.args = {}
        cls.vs = [cls.dist.rvs(s) for s in cls.sizes]
        cls.vs = [ v[v < 20] for v in cls.vs]
        cls.xs = np.r_[0:20:1024j]
        cls.accuracy = 1e-4

    def createKDE(self, data, **args):
        all_args = dict(self.args)
        all_args.update(args)
        return kde.KDE1D(data, **all_args)

    def is_normed(self, i):
        k = self.createKDE(self.vs[i])
        tot = float(k.cdf(k.upper))
        assert abs(tot - 1) < self.accuracy, "Error, k.cdf({0}) = {1} should be close to 1".format(k.upper, tot)

    def same_numeric(self, i):
        k = self.createKDE(self.vs[i])
        ys = k.cdf(self.xs)
        xxs, yys = k.method.numeric_cdf_grid(k, N=2**15)
        interp = interpolate.interp1d(xxs, yys)
        sel = self.xs <= xxs.max()
        xs = self.xs[sel]
        ys2 = interp(xs)
        error = max(abs(ys2 - ys[sel]))
        assert error < self.accuracy, "Error, max(abs(ys2 - ys)) = {} should be close to 0".format(error)

    def grid_same_numeric(self, i):
        k = self.createKDE(self.vs[i])
        xs, ys = k.grid_cdf()
        xxs, yys = k.method.numeric_cdf_grid(k, N=2**15)
        interp = interpolate.interp1d(xxs, yys)
        ys2 = interp(self.xs)
        error = max(abs(ys2 - ys))
        assert error < self.accuracy, "Error, max(abs(ys2 - ys)) = {} should be close to 0".format(error)

    def numeric_cdf(self, i):
        k = self.createKDE(self.vs[i])
        ys = np.empty(self.xs.shape, dtype=float)
        k.method.numeric_cdf(k, self.xs, output=ys)
        xxs, yys = k.method.numeric_cdf_grid(k, N=2**15)
        interp = interpolate.interp1d(xxs, yys)
        ys2 = interp(self.xs)
        error = max(abs(ys2 - ys))
        assert error < self.accuracy, "Error, max(abs(ys2 - ys)) = {} should be close to 0".format(error)

    def test_normed(self):
        for i in irange(len(self.sizes)):
            yield self.is_normed, i

    def test_same_numeric(self):
        for i in irange(len(self.sizes)):
            yield self.same_numeric, i


class TestReflectionCDF(TestUnboundedCDF):
    @classmethod
    def setUpClass(cls):
        TestUnboundedCDF.setUpClass()
        cls.args = dict(lower=0, upper=20, method=kde_methods.reflection)


class TestCyclicCDF(TestUnboundedCDF):
    @classmethod
    def setUpClass(cls):
        TestUnboundedCDF.setUpClass()
        cls.args = dict(lower=0, upper=20, method=kde_methods.cyclic)
        cls.accuracy = 1e-3

    def test_numeric(self):
        self.numeric_cdf(0)

class TestRenormCDF(TestUnboundedCDF):
    @classmethod
    def setUpClass(cls):
        TestUnboundedCDF.setUpClass()
        cls.args = dict(lower=0, upper=20, method=kde_methods.renormalization)


class TestLCCDF(TestUnboundedCDF):
    @classmethod
    def setUpClass(cls):
        TestUnboundedCDF.setUpClass()
        cls.args = dict(lower=0, upper=20, method=kde_methods.linear_combination)
        cls.accuracy = 1e-2 # This method makes large approximation on boundaries

    def test_normed(self):
        pass # Skip this one as we don't expect the result to be normed

