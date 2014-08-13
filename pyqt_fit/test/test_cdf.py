from __future__ import division, absolute_import, print_function

import numpy as np
from . import kde_utils

class TestCDF(kde_utils.KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_lognorm(cls)

    def method_works(self, k, method, name):
        begin, last = k.cdf([k.lower, k.upper])
        assert abs(last - 1) < method.accuracy, "Error, k.cdf({0}) = {1} should be close to 1".format(k.upper, last)
        assert abs(begin) < method.accuracy, "Error, k.cdf({0}) = {1} should be close to 0".format(k.lower, begin)

    def same_numeric(self, i, method):
        k = self.createKDE(self.vs[i], method)
        ys = k.cdf(self.xs)
        ys2 = np.empty(self.xs.shape, dtype=float)
        k.method.numeric_cdf(k, self.xs, ys2)
        np.testing.assert_allclose(ys, ys2, method.accuracy, method.accuracy)

    def grid_same_numeric(self, i, method):
        k = self.createKDE(self.vs[i], method)
        xs, ys = k.cdf_grid()
        ys2 = np.empty(self.xs.shape, dtype=float)
        k.method.numeric_cdf(k, xs, out=ys2)
        np.testing.assert_allclose(ys, ys2, method.grid_accuracy, method.grid_accuracy)

    def numeric_cdf(self, i, method):
        k = self.createKDE(self.vs[i], method)
        k.fit()
        ys = np.empty(self.xs.shape, dtype=float)
        k.method.numeric_cdf(k, self.xs, out=ys)
        xxs, yys = k.method.numeric_cdf_grid(k, N=2**12)
        ys2 = np.interp(self.xs, xxs, yys)
        np.testing.assert_allclose(ys, ys2, 100*method.accuracy, 100*method.accuracy)

    def test_same_numeric(self):
        self.same_numeric(0, kde_utils.methods[0])

    def test_grid_same_numeric(self):
        self.grid_same_numeric(0, kde_utils.methods[0])

    def test_numeric_cdf(self):
        self.numeric_cdf(0, kde_utils.methods[0])

    def grid_method_works(self, k, method, name):
        xs, ys = k.cdf_grid(64)
        acc = method.accuracy
        assert np.all(ys >= -acc), "Some negative values"
        assert np.all(ys <= 1+acc), "CDF must be below one"
        assert np.all(ys[1:] - ys[:-1] >= -acc), "The CDF must be strictly growing."

    def kernel_works(self, ker, name):
        method = kde_utils.methods[0]
        k = self.createKDE(self.vs[1], method)
        k.kernel = ker.cls()
        begin, last = k.cdf([k.lower, k.upper])
        acc = method.accuracy * ker.precision_factor
        assert abs(last - 1) < acc, "Error, k.cdf({0}) = {1} should be close to 1".format(k.upper, last)
        assert abs(begin) < acc, "Error, k.cdf({0}) = {1} should be close to 0".format(k.lower, begin)

    def grid_kernel_works(self, ker, name):
        method = kde_utils.methods[0]
        k = self.createKDE(self.vs[1], method)
        k.kernel = ker.cls()
        xs, ys = k.cdf_grid(cut=5)
        acc = method.accuracy * ker.precision_factor
        if ker.positive: # This is true only if the kernel is a probability ... that is not higher order!
            assert np.all(ys >= -acc), "Some negative values"
            assert np.all(ys <= 1+acc), "CDF must be below one"
            assert np.all(ys[1:] - ys[:-1] >= -acc), "The CDF must be strictly growing."
        else:
            assert abs(ys[0]) < acc, "Error, k.cdf({0}) = {1} should be close to 0".format(xs[0], ys[0])
            assert abs(ys[-1]-1) < acc, "Error, k.cdf({0}) = {1} should be close to 1".format(xs[-1], ys[-1])

