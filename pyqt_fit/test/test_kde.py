from __future__ import division, absolute_import, print_function

import unittest
from .. import kde
import numpy as np
from numpy import newaxis
from numpy.random import randn
from scipy import stats

class TestBandwidth(object):

    @classmethod
    def setUpClass(cls):
        cls.ratios = np.array([1.,2.,5.])
        d = randn(500)
        cls.vs = cls.ratios[:,newaxis]*np.array([d,d,d])
        cls.ss = np.var(cls.vs, axis=1)

    def variance_methods(self, m):
        bws = np.array([ m(v) for v in self.vs ])
        assert bws.shape == (3,1,1)
        rati = bws[:,0,0] / self.ss
        assert sum((rati - rati[0])**2) < 1e-6
        rati = bws[:,0,0] / bws[0,0,0]
        assert sum((rati - self.ratios**2)**2) < 1e-6

    def test_variance_methods(self):
        yield self.variance_methods, kde.silverman_bandwidth
        yield self.variance_methods, kde.scotts_bandwidth

    def test_botev(self):
        bws = np.array([ kde.botev_bandwidth()(v) for v in self.vs ])
        assert bws.shape == (3,)
        rati = bws**2 / self.ss
        assert sum((rati - rati[0])**2) < 1e-6
        rati = bws / bws[0]
        assert sum((rati - self.ratios)**2) < 1e-6


class TestUnboundedKDE1D(object):

    @classmethod
    def setUpClass(cls):
        self.dist = stats.norm(0,1)
        cls.sizes = np.r_[1000:5000:5j]
        cls.vs = [self.dist.rvs(s) for s in cls.sizes]

    def test_converge(self):
        xs = np.r_[-3:3:512j]
        ys = self.dist.pdf(xs)
        ks = [ kde.KDE1D(v) for v in cls.vs ]
