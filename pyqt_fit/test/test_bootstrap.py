from __future__ import division, absolute_import, print_function

import unittest
from .. import bootstrap
import sys

import numpy as np

class TestBootstrapMethods(unittest.TestCase):
    def setUp(self):
        self.xdata = np.r_[0:1000:129j]
        self.res = np.r_[-4:5:129j]
        self.centered_res = self.res - np.mean(self.res)
        self.ydata = self.xdata + self.res

    def base_residuals(self, fct, real_res, *args, **kwords):
        def noop(x):
            return x
        for rep in (20, 129, 500):
            new_x, new_y = fct(noop, self.xdata, self.ydata, repeats = rep, *args, **kwords)
            used_res = new_y - new_x
            self.assertEqual(used_res.shape, self.xdata.shape[:-1] + (rep,) + self.xdata.shape[-1:])
            diff = (np.abs(np.subtract.outer(used_res, real_res)) < 1e-10).sum(axis=2)
            np.testing.assert_array_equal(diff, np.ones(dtype=diff.dtype, shape=diff.shape))

    def test_residuals_simple(self):
        self.base_residuals(bootstrap.bootstrap_residuals, self.centered_res)

    def test_residuals_res_fct(self):
        def residuals(y1, y0):
            return y1 - y0 + 1

        def add_residual(y, r):
            return y + r - 1

        self.base_residuals(bootstrap.bootstrap_residuals, self.centered_res-1, residuals=residuals, add_residual = add_residual)

    def test_residuals_res_values(self):
        residuals = self.res + 1

        def add_residual(y, r):
            return y + r - 1

        self.base_residuals(bootstrap.bootstrap_residuals, self.centered_res-1, residuals=residuals, add_residual = add_residual)

    def base_regression(self, fct, *args, **kwords):
        def noop(x):
            return x
        for rep in (20, 129, 500):
            new_x, new_y = fct(noop, self.xdata, self.ydata, repeats = rep, *args, **kwords)
            used_res = new_y - new_x
            self.assertEqual(used_res.shape, self.xdata.shape[:-1] + (rep,) + self.xdata.shape[-1:])
            diff = (np.abs(np.subtract.outer(new_y, self.ydata)) < 1e-10).sum(axis=2)
            np.testing.assert_array_equal(diff, np.ones(dtype=diff.dtype, shape=diff.shape))

    def test_regression_simple(self):
        self.base_regression(bootstrap.bootstrap_regression)

class VoidFittingFct(object):
    def __init__(self, y, *args, **kwords):
        self.args = np.array(args)
        print(y[0], file=sys.stderr)
        self.y = y[0]

    def __call__(self, x):
        return x+self.y

class VoidFitting(object):
    def __init__(self, test, *args, **kwords):
        self.test = test
        self.args = args
        self.kwords = kwords

    def __call__(self, xdata, ydata, *args, **kwords):
        self.test.assertTupleEqual(args, self.args)
        self.test.assertDictEqual(kwords, self.kwords)
        return VoidFittingFct(ydata, *args, **kwords)

class CountShuffleSparse(object):
    def __init__(self, test, *args, **kwords):
        self.test = test
        self.args = args
        self.kwords = kwords

    def __call__(self, fct, xdata, ydata, *args, **kwords):
        repeats = kwords.pop('repeats', 3000)
        self.test.assertTupleEqual(args, self.args)
        self.test.assertDictEqual(kwords, self.kwords)
        return xdata[...,np.newaxis,:], np.zeros(ydata.shape) + np.r_[1:repeats:repeats*1j][:,np.newaxis]

class CountShuffleFull(object):
    def __init__(self, test, *args, **kwords):
        self.test = test
        self.args = args
        self.kwords = kwords

    def __call__(self, fct, xdata, ydata, *args, **kwords):
        repeats = kwords.pop('repeats', 3000)
        self.test.assertTupleEqual(args, self.args)
        self.test.assertDictEqual(kwords, self.kwords)
        return np.repeat(xdata[...,np.newaxis,:], repeats, axis=-2), zeros(ydata.shape) + np.r_[1:repeats:repeats*1j][:,np.newaxis]

class TestBoostrap(unittest.TestCase):
    def setUp(self):
        self.xdata = np.r_[0:1000:129j]
        self.ydata = np.zeros((129,), dtype=float)
        self.eval_points = np.r_[10:900:65j]

    def simple(self, fit, shuffle, CI, extra_attrs = (), other_tests = lambda x: None, *args, **kwords):
        for rep in (20, 129, 500):
            result = bootstrap.bootstrap(fit, self.xdata, self.ydata, CI=CI, extra_attrs = extra_attrs,
                                         eval_points = self.eval_points, repeats = rep,
                                         shuffle_method = shuffle, *args, **kwords)
            np.testing.assert_array_equal(result.y_est, self.xdata+1)
            np.testing.assert_array_equal(result.y_eval, self.eval_points+1)
            self.assertEqual(len(result.CIs), 1 + len(extra_attrs))
            for ci in result.CIs:
                self.assertTupleEqual(ci.shape[:-1], (len(CI),2))
            self.assertEqual(result.CIs[0].shape[-1], len(self.eval_points))

            if result.shuffled_xs is not None:
                self.assertIn(result.shuffled_xs.shape[-2], (1, rep))
                self.assertIn(result.shuffled_ys.shape[-2], (1, rep))
                self.assertTupleEqual(result.full_results.shape, (rep+1, len(self.eval_points)))
                np.testing.assert_array_equal(result.full_results, self.eval_points[np.newaxis,:] + r_[0:rep:(rep+1)*1j])
            other_tests(result)

    def test_simple_sparse(self):
        self.simple(VoidFitting(self, 1, 2, a=3, b=4), CountShuffleSparse(self, 10, 12, d=4, e=6), CI=(95,),
                shuffle_args = (10, 12), shuffle_kwrds = {'d': 4, 'e': 6}, fit_args=(1,2), fit_kwrds = {'a': 3, 'b': 4}, nb_workers=1)

    def test_simple_attrs(self):
        self.simple(VoidFitting(self, 1, 2, a=3, b=4), CountShuffleSparse(self, 10, 12, d=4, e=6), CI=(95,), extra_attrs = ('args',),
                shuffle_args = (10, 12), shuffle_kwrds = {'d': 4, 'e': 6}, fit_args=(1,2), fit_kwrds = {'a': 3, 'b': 4}, nb_workers=1)

    def test_simple_full(self):
        self.simple(VoidFitting(self), CountShuffleSparse(self), CI=(95,), nb_workers=1, full_results=True)

