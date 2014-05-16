from __future__ import division, absolute_import, print_function

from .. import utils
from nose.tools import raises


class TestNamedTuple(object):
    def test_normal(self):
        nt = utils.namedtuple('nt', 'x y z')
        n = nt(1, 2, 3)
        assert hasattr(n, 'x')
        assert n.x == 1
        assert hasattr(n, 'y')
        assert n.y == 2
        assert hasattr(n, 'z')
        assert n.z == 3

    def test_replace(self):
        nt = utils.namedtuple('nt', 'x y z')
        n = nt(1, 2, 3)
        n1 = n._replace(x=10)
        assert n1.x == 10
        assert n1.y == 2
        assert n1.z == 3

    def test_extended(self):
        nt = utils.namedtuple('nt', 'x __call__ __tested__')
        n = nt(1, lambda x: x, 3)
        assert hasattr(n, 'x')
        assert hasattr(n, '__call__')
        assert hasattr(n, '__tested__')
        assert n(True)

    @raises(ValueError)
    def test_badname(self):
        utils.namedtuple('nt', 'x y _test1')

    @raises(ValueError)
    def reserved(self, n):
        utils.namedtuple('nt', n)

    def test_reserved_names(self):
        for n in ['__init__', '__slots__', '__new__', '__repr__', '__getnewargs__']:
            yield self.reserved, n

    @raises(ValueError)
    def test_keyword(self):
        utils.namedtuple('nt', 'is')

    @raises(ValueError)
    def test_duplicate(self):
        utils.namedtuple('nt', 'x y x')

    @raises(ValueError)
    def test_number(self):
        utils.namedtuple('nt', 'x 12 x')

    @raises(ValueError)
    def test_symbol(self):
        utils.namedtuple('nt', 'x y-t x')

import numpy as np


def square(vals):
    return vals ** 2


class TestJacobian(object):
    def test_call(self):
        res = utils.approx_jacobian(np.array([0, 0]), square, 1)
        np.testing.assert_array_equal(res, np.eye(2, dtype=res.dtype))

    def test_call2(self):
        res = utils.approx_jacobian(np.array([1, 1]), square, 1e-6)
        np.testing.assert_array_almost_equal(res, 2 * np.eye(2, dtype=res.dtype), 6)

    def call_prec(self, p, dt=float):
        res = utils.approx_jacobian(np.array([1, 1], dtype=dt), square, 10 ** (-p))
        np.testing.assert_array_almost_equal(res, 2 * np.eye(2, dtype=res.dtype), p)

    def test_call_prec(self):
        for p in (3, 4, 5, 6, 8):  # Above 8, we hit precision limit of floating point
            yield self.call_prec, p

    def test_call_high_prec(self):
        for p in (5, 8, 9):  # Above 9, we hit precision limit of floating point
            yield self.call_prec, p, utils.large_float
