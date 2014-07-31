from __future__ import division, absolute_import, print_function

from .. import kde
from .. import kde_methods
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


class TestUnboundedKDE1D(object):
    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)

    def createKDE(self, data, **args):
        return kde.KDE1D(data, **args)

    #def test_converge(self):
        #xs = np.r_[-3:3:512j]
        #ys = self.dist.pdf(xs)
        #ks = [ self.createKDE(v, **self.args) for v in self.vs ]

    def is_normed(self, i):
        k = self.createKDE(self.vs[i], **self.args)
        xs, ys = k.grid(2048)
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

    def is_weights_normed(self, i):
        weights = self.weights[i]
        k = self.createKDE(self.vs[i], weights=weights, **self.args)
        xs, ys = k.grid(2048)
        tot = integrate.simps(ys, xs)
        assert abs(tot - 1) < self.accuracy, "Error, {} should be close to 1".format(tot)

    def is_weights_grid_normed(self, i):
        weights = self.weights[i]
        k = self.createKDE(self.vs[i], weights=weights, **self.args)
        xs, ys = k.grid(2048)
        tot = integrate.simps(ys, xs)
        assert abs(tot - 1) < self.grid_accuracy, "Error, {} should be close to 1".format(tot)

    def test_weights_normed(self):
        """
        Test with weights
        """
        for i in irange(len(self.sizes)):
            yield self.is_weights_normed, i

    def test_weights_grid_normed(self):
        for i in irange(len(self.sizes)):
            yield self.is_weights_grid_normed, i

    def is_lambdas_normed(self, i):
        lambdas = self.lambdas[i]
        k = self.createKDE(self.vs[i], lambdas=lambdas, **self.args)
        xs, ys = k.grid(2048)
        tot = integrate.simps(ys, xs)
        assert abs(tot - 1) < self.accuracy, "Error, {} should be close to 1".format(tot)

    def test_lambdas_normed(self):
        """
        Test with lambdas
        """
        for i in irange(len(self.sizes)):
            yield self.is_lambdas_normed, i


class ToTestBoundedKDE1D(TestUnboundedKDE1D):
    @classmethod
    def mainSetup(cls):
        kde_utils.setupClass_norm(cls)
        cls.args = dict(lower=-5, upper=5, method=cls.method())


for met in kde_utils.methods:
    cls_name = 'Test{}KDE1D'.format(met.cls.name.replace(' ', '_'))
    template = r"""
class {0}(ToTestBoundedKDE1D):
    @classmethod
    def setUpClass(cls):
        cls.mainSetup()
        cls.accuracy = {1}
        cls.grid_accuracy = {2}
""".format(cls_name, met.accuracy, met.grid_accuracy)
    exec(template, globals())
    cls = globals()[cls_name]
    setattr(cls, 'method', met.cls)

