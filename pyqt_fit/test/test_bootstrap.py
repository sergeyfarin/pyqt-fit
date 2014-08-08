from __future__ import division, absolute_import, print_function

from .. import bootstrap

import numpy as np


class TestBootstrapMethods(object):

    @classmethod
    def setUpClass(cls):
        cls.xdata = np.r_[0:1000:129j]
        cls.res = np.r_[-4:5:129j]
        cls.centered_res = cls.res - np.mean(cls.res)
        cls.ydata = cls.xdata + cls.res

    def base_residuals(self, rep, fct, real_res, args, kwords):
        def noop(x):
            return x
        new_x, new_y = fct(noop, self.xdata, self.ydata, repeats=rep, *args, **kwords)
        used_res = new_y - new_x
        assert used_res.shape == self.xdata.shape[:-1] + (rep,) + self.xdata.shape[-1:]
        diff = (np.abs(np.subtract.outer(used_res, real_res)) < 1e-10).sum(axis=2)
        np.testing.assert_array_equal(diff, np.ones(dtype=diff.dtype, shape=diff.shape))

    def iterate_base_residuals(self, fct, real_res, *args, **kwords):
        for rep in (20, 129, 500):
            yield self.base_residuals, rep, fct, real_res, args, kwords

    def test_residuals_simple(self):
        for args in self.iterate_base_residuals(bootstrap.bootstrap_residuals, self.centered_res):
            yield args

    def test_residuals_res_fct(self):
        def residuals(y1, y0):
            return y1 - y0 + 1

        def add_residual(y, r):
            return y + r - 1

        for args in self.iterate_base_residuals(bootstrap.bootstrap_residuals,
                                                self.centered_res - 1, residuals=residuals,
                                                add_residual=add_residual):
            yield args

    def test_residuals_res_values(self):
        residuals = self.res + 1

        def add_residual(y, r):
            return y + r - 1

        for args in self.iterate_base_residuals(bootstrap.bootstrap_residuals,
                                                self.centered_res - 1, residuals=residuals,
                                                add_residual=add_residual):
            yield args

    def base_regression(self, rep, fct, args, kwords):
        def noop(x):
            return x
        new_x, new_y = fct(noop, self.xdata, self.ydata, repeats=rep, *args, **kwords)
        used_res = new_y - new_x
        assert used_res.shape == self.xdata.shape[:-1] + (rep,) + self.xdata.shape[-1:]
        diff = (np.abs(np.subtract.outer(new_y, self.ydata)) < 1e-10).sum(axis=2)
        np.testing.assert_array_equal(diff, np.ones(dtype=diff.dtype, shape=diff.shape))

    def test_regression_simple(self):
        for rep in (20, 129, 500):
            yield self.base_regression, rep, bootstrap.bootstrap_regression, (), {}


class VoidFittingFct(object):
    def __init__(self, y, *args, **kwords):
        self.args = np.array(args)
        #print(y[0], file=sys.stderr)
        self.y = y[0]

    def __call__(self, x):
        return x + self.y

    def fit(self):
        pass


class VoidFitting(object):
    def __init__(self, *args, **kwords):
        self.args = args
        self.kwords = kwords

    def __call__(self, xdata, ydata, *args, **kwords):
        assert args == self.args
        assert kwords == self.kwords
        return VoidFittingFct(ydata, *args, **kwords)


class CountShuffleSparse(object):
    def __init__(self, *args, **kwords):
        self.args = args
        self.kwords = kwords

    def __call__(self, fct, xdata, ydata, *args, **kwords):
        repeats = kwords.pop('repeats', 3000)
        assert args == self.args
        assert kwords == self.kwords
        return (xdata[..., np.newaxis, :],
                np.zeros(ydata.shape) + np.r_[1:repeats:repeats * 1j][:, np.newaxis])


class CountShuffleFull(object):
    def __init__(self, *args, **kwords):
        self.args = args
        self.kwords = kwords

    def __call__(self, fct, xdata, ydata, *args, **kwords):
        repeats = kwords.pop('repeats', 3000)
        assert args == self.args
        assert kwords == self.kwords
        return (np.repeat(xdata[..., np.newaxis, :], repeats, axis=-2),
                np.zeros(ydata.shape) + np.r_[1:repeats:repeats * 1j][:, np.newaxis])


class TestBoostrap(object):
    @classmethod
    def setUpClass(cls):
        cls.xdata = np.r_[0:1000:129j]
        cls.ydata = np.zeros((129,), dtype=float)
        cls.eval_points = np.r_[10:900:65j]

    def simple(self, rep, fit, shuffle, CI, extra_attrs, other_tests, args, kwords):
        result = bootstrap.bootstrap(fit, self.xdata, self.ydata, CI=CI, extra_attrs=extra_attrs,
                                     eval_points=self.eval_points, repeats=rep,
                                     shuffle_method=shuffle, *args, **kwords)
        np.testing.assert_array_equal(result.y_est, self.xdata)
        np.testing.assert_array_equal(result.y_eval, self.eval_points)
        assert len(result.CIs) == 1 + len(extra_attrs)
        for ci in result.CIs:
            assert ci.shape[:-1] == (len(CI), 2)
        assert result.CIs[0].shape[-1] == len(self.eval_points)

        if result.shuffled_xs is not None:
            assert result.shuffled_xs.shape[-2] in (1, rep)
            assert result.shuffled_ys.shape[-2] in (1, rep)
            assert result.full_results.shape == (rep + 1, len(self.eval_points))
            expected = self.eval_points + np.r_[0:rep:(rep + 1) * 1j][:, np.newaxis]
            np.testing.assert_array_equal(result.full_results,
                                          expected)
        other_tests(result)

    def iter_simple(self, fit, shuffle, CI, extra_attrs=(),
                    other_tests = lambda x: None, *args, **kwords):
        for rep in (20, 129, 500):
            for worker in (1, None, 2, 4):
                kw = dict(kwords)
                kw['nb_workers'] = worker
                yield self.simple, rep, fit, shuffle, CI, extra_attrs, other_tests, args, kw

    def test_simple_sparse(self):
        for arg in self.iter_simple(VoidFitting(1, 2, a=3, b=4),
                                    CountShuffleSparse(10, 12, d=4, e=6), CI=(95,),
                                    shuffle_args = (10, 12), shuffle_kwrds = {'d': 4, 'e': 6},
                                    fit_args=(1, 2), fit_kwrds = {'a': 3, 'b': 4}):
            yield arg

    def test_simple_attrs(self):
        for arg in self.iter_simple(VoidFitting(1, 2, a=3, b=4),
                                    CountShuffleSparse(10, 12, d=4, e=6),
                                    CI=(95,), extra_attrs = ('args',),
                                    shuffle_args = (10, 12),
                                    shuffle_kwrds = {'d': 4, 'e': 6},
                                    fit_args=(1, 2), fit_kwrds = {'a': 3, 'b': 4}):
            yield arg

    def test_simple_full(self):
        for arg in self.iter_simple(VoidFitting(), CountShuffleSparse(),
                                    CI=(95,), full_results=True):
            yield arg
