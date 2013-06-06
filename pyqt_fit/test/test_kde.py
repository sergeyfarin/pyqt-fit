from __future__ import division, absolute_import, print_function

from .. import kde
import numpy as np
from numpy import newaxis
from numpy.random import randn
from scipy import stats, integrate
from ..compat import irange


class TestBandwidth(object):

    @classmethod
    def setUpClass(cls):
        cls.ratios = np.array([1., 2., 5.])
        d = randn(500)
        cls.vs = cls.ratios[:, newaxis]*np.array([d, d, d])
        cls.ss = np.var(cls.vs, axis=1)

    def variance_methods(self, m):
        bws = np.array([m(v) for v in self.vs])
        assert bws.shape == (3, 1, 1)
        rati = bws[:, 0, 0] / self.ss
        assert sum((rati - rati[0])**2) < 1e-6
        rati = bws[:, 0, 0] / bws[0, 0, 0]
        assert sum((rati - self.ratios**2)**2) < 1e-6

    def test_variance_methods(self):
        yield self.variance_methods, kde.silverman_bandwidth
        yield self.variance_methods, kde.scotts_bandwidth

    def test_botev(self):
        class FakeModel(object):
            lower = -np.inf
            upper = np.inf
        bws = np.array([kde.botev_bandwidth()(v, model=FakeModel()) for v in self.vs])
        assert bws.shape == (3,)
        rati = bws**2 / self.ss
        assert sum((rati - rati[0])**2) < 1e-6
        rati = bws / bws[0]
        assert sum((rati - self.ratios)**2) < 1e-6


class TestUnboundedKDE1D(object):
    @classmethod
    def setUpClass(cls):
        cls.dist = stats.norm(0, 1)
        cls.sizes = np.r_[1000:5000:5j]
        cls.vs = [cls.dist.rvs(s) for s in cls.sizes]
        cls.args = {}
        cls.grid_accuracy = 1e-6
        cls.accuracy = 1e-3

    def createKDE(self, data, **args):
        return kde.KDE1D(data, **args)

    #def test_converge(self):
        #xs = np.r_[-3:3:512j]
        #ys = self.dist.pdf(xs)
        #ks = [ self.createKDE(v, **self.args) for v in self.vs ]

    def is_normed(self, i):
        k = self.createKDE(self.vs[i], **self.args)
        xs, ys = k._grid_eval(2048)
        tot = integrate.simps(ys, xs)
        assert abs(tot - 1) < self.accuracy, "Error, {} should be close to 1".format(tot)

    def is_grid_normed(self, i):
        k = self.createKDE(self.vs[i], **self.args)
        xs, ys = k.grid(2048)
        tot = integrate.simps(ys, xs)
        assert abs(tot - 1) < self.grid_accuracy, "Error, {} should be close to 1".format(tot)

    def test_normed(self):
        for i in irange(len(self.sizes)):
            yield self.is_normed, i

    def test_grid_normed(self):
        for i in irange(len(self.sizes)):
            yield self.is_grid_normed, i

    def is_ws_normed(self, i):
        ws = np.r_[1:2:self.sizes[i]*1j]
        k = self.createKDE(self.vs[i], weights=ws, **self.args)
        xs, ys = k._grid_eval(2048)
        tot = integrate.simps(ys, xs)
        assert abs(tot - 1) < self.accuracy, "Error, {} should be close to 1".format(tot)

    def is_ws_grid_normed(self, i):
        ws = np.r_[1:2:self.sizes[i]*1j]
        k = self.createKDE(self.vs[i], weights=ws, **self.args)
        xs, ys = k.grid(2048)
        tot = integrate.simps(ys, xs)
        assert abs(tot - 1) < self.grid_accuracy, "Error, {} should be close to 1".format(tot)

    def test_ws_normed(self):
        for i in irange(len(self.sizes)):
            yield self.is_ws_normed, i

    def test_ws_grid_normed(self):
        for i in irange(len(self.sizes)):
            yield self.is_ws_grid_normed, i

    def is_ls_normed(self, i):
        ws = np.r_[1:2:self.sizes[i]*1j]
        k = self.createKDE(self.vs[i], lambdas=ws, **self.args)
        xs, ys = k._grid_eval(2048)
        tot = integrate.simps(ys, xs)
        assert abs(tot - 1) < self.accuracy, "Error, {} should be close to 1".format(tot)

    def test_ls_normed(self):
        for i in irange(len(self.sizes)):
            yield self.is_ls_normed, i


class TestReflexionKDE1D(TestUnboundedKDE1D):
    @classmethod
    def setUpClass(cls):
        cls.dist = stats.norm(0, 1)
        cls.sizes = np.r_[1000:5000:5j]
        cls.vs = [cls.dist.rvs(s) for s in cls.sizes]
        cls.args = dict(lower=-5, upper=5, method='reflexion')
        cls.grid_accuracy = 1e-5
        cls.accuracy = 1e-3


class TestCyclicKDE1D(TestUnboundedKDE1D):
    @classmethod
    def setUpClass(cls):
        cls.dist = stats.norm(0, 1)
        cls.sizes = np.r_[1000:5000:5j]
        cls.vs = [cls.dist.rvs(s) for s in cls.sizes]
        cls.args = dict(lower=-5, upper=5, method='cyclic')
        cls.grid_accuracy = 1e-5
        cls.accuracy = 1e-3


class TestRenormKDE1D(TestUnboundedKDE1D):
    @classmethod
    def setUpClass(cls):
        cls.dist = stats.norm(0, 1)
        cls.sizes = np.r_[1000:5000:5j]
        cls.vs = [cls.dist.rvs(s) for s in cls.sizes]
        cls.args = dict(lower=-5, upper=5, method='renormalization')
        cls.grid_accuracy = 1e-4
        cls.accuracy = 1e-3


class TestLCKDE1D(TestUnboundedKDE1D):
    @classmethod
    def setUpClass(cls):
        cls.dist = stats.norm(0, 1)
        cls.sizes = np.r_[1000:5000:5j]
        cls.vs = [cls.dist.rvs(s) for s in cls.sizes]
        cls.args = dict(lower=-5, upper=5, method='linear_combination')
        cls.grid_accuracy = 1e-4
        cls.accuracy = 1e-3
