from __future__ import division, absolute_import, print_function

from .. import kde
from .. import kde_methods
import numpy as np
from numpy import newaxis
from numpy.random import randn
from scipy import stats, integrate, interpolate
from ..compat import irange
from ..utils import make_ufunc

from . import kde_utils

class TestCDF(kde_utils.KDETester):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)

    def createKDE(self, data, method, **args):
        all_args = dict(self.args)
        all_args.update(args)
        k = kde.KDE1D(data, **all_args)
        if method.cls is None:
            del k.method
        else:
            k.method = method.cls()
        if method.bounded:
            k.lower = self.lower
            k.upper = self.upper
        else:
            del k.lower
            del k.upper
        assert k.fitted is False
        return k

    def method_works(self, i, method, **args):
        k = self.createKDE(self.vs[i], method, **args)
        begin, last = k.cdf([k.lower, k.upper])
        assert abs(last - 1) < method.accuracy, "Error, k.cdf({0}) = {1} should be close to 1".format(k.upper, last)
        assert abs(begin) < method.accuracy, "Error, k.cdf({0}) = {1} should be close to 0".format(k.lower, begin)

    def same_numeric(self, i, method):
        k = self.createKDE(self.vs[i], method)
        ys = k.cdf(self.xs)
        ys2 = k.method.numeric_cdf(k, self.xs)
        np.testing.assert_allclose(ys, ys2, method.accuracy, method.accuracy)

    def grid_same_numeric(self, i, method):
        k = self.createKDE(self.vs[i], method)
        xs, ys = k.cdf_grid()
        ys2 = k.method.numeric_cdf(k, xs)
        np.testing.assert_allclose(ys, ys2, method.grid_accuracy, method.grid_accuracy)

    def numeric_cdf(self, i, method):
        k = self.createKDE(self.vs[i], method)
        ys = k.method.numeric_cdf(k, self.xs)
        xxs, yys = k.method.numeric_cdf_grid(k, N=2**12)
        ys2 = np.interp(self.xs, xxs, yys)
        np.testing.assert_allclose(ys, ys2, 100*method.accuracy, 100*method.accuracy)

    def test_same_numeric(self):
        self.same_numeric(0, kde_utils.methods[0])

    def test_grid_same_numeric(self):
        self.grid_same_numeric(0, kde_utils.methods[0])

    def test_numeric_cdf(self):
        self.numeric_cdf(0, kde_utils.methods[0])

    def weights_method_works(self, i, method):
        return self.method_works(i, method, weights = self.weights[i])

    def lambdas_method_works(self, i, method):
        return self.method_works(i, method, lambdas = self.lambdas[i])

    def grid_method_works(self, i, method):
        k = self.createKDE(self.vs[i], method)
        xs, ys = k.cdf_grid(64)
        acc = method.accuracy
        assert np.all(ys >= -acc), "Some negative values"
        assert np.all(ys <= 1+acc), "CDF must be below one"
        assert np.all(ys[1:] - ys[:-1] >= -acc), "The CDF must be strictly growing."

    def weights_grid_method_works(self, i, method):
        weights = self.weights[i]
        k = self.createKDE(self.vs[i], method)
        k.weights = self.weights[i]
        xs, ys = k.cdf_grid(64)
        acc = method.accuracy
        assert np.all(ys >= -acc), "Some negative values"
        assert np.all(ys <= 1+acc), "CDF must be below one"
        assert np.all(ys[1:] - ys[:-1] >= -acc), "The CDF must be strictly growing."

    def lambdas_grid_method_works(self, i, method):
        k = self.createKDE(self.vs[i], method)
        k.lambdas = self.lambdas[i]
        xs, ys = k.cdf_grid(64)
        acc = method.accuracy
        assert np.all(ys >= -acc), "Some negative values"
        assert np.all(ys <= 1+acc), "CDF must be below one"
        assert np.all(ys[1:] - ys[:-1] >= -acc), "The CDF must be strictly growing."

    def kernel_works(self, ker):
        method = kde_utils.methods[0]
        k = self.createKDE(self.vs[1], method)
        k.kernel = ker.cls()
        begin, last = k.cdf([k.lower, k.upper])
        acc = method.accuracy * ker.precision_factor
        assert abs(last - 1) < acc, "Error, k.cdf({0}) = {1} should be close to 1".format(k.upper, last)
        assert abs(begin) < acc, "Error, k.cdf({0}) = {1} should be close to 0".format(k.lower, begin)

    def grid_kernel_works(self, ker):
        method = kde_utils.methods[0]
        k = self.createKDE(self.vs[1], method)
        k.kernel = ker.cls()
        xs, ys = k.cdf_grid()
        acc = method.accuracy * ker.precision_factor
        assert np.all(ys >= -acc), "Some negative values"
        assert np.all(ys <= 1+acc), "CDF must be below one"
        assert np.all(ys[1:] - ys[:-1] >= -acc), "The CDF must be strictly growing."

