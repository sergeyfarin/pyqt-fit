from __future__ import division, absolute_import, print_function

from .. import kernels
from .. import _kernels

from scipy import stats
from scipy.fftpack import fft, dct
from scipy.integrate import quad
import numpy as np


class KernelTester(object):
    @classmethod
    def initUpClass(cls, lower=-np.inf, test_width=3):
        cls.lower = float(lower)
        cls.hard_points = ()
        cls.quad_args = dict(limit=100)
        cls.test_xs = np.r_[-test_width / 2:test_width / 2:17j]
        cls.pos_xs = 20 * (np.arange(1024) + 0.5) / 1024
        cls.xs = np.roll(20 * (np.arange(1024) - 512) / 1024, 512)
        cls.pos_freqs = np.pi / 20 * np.arange(1024)
        freqs = 2 * np.pi / 20 * (np.arange(1024) - 512)
        freqs = np.roll(freqs, 512)
        cls.freqs = freqs

    def _cdf(self, upper, tol):
        val = self.kernel.cdf(upper)
        args = dict(self.quad_args)
        if self.hard_points:
            args.update(points=[p for p in self.hard_points if p > self.lower and p < upper])
        val1, err = quad(self.kernel.pdf, self.lower, upper, **args)
        assert abs(val - val1) < tol, "cdf({3:.9g}) -- Expected: {0:g}, computed: {1:g}".format(val1, val, err, upper)

    def _pm1(self, upper, tol):
        val = self.kernel.pm1(upper)

        def fct(x):
            return x * self.kernel.pdf(x)
        args = dict(self.quad_args)
        if self.hard_points:
            args.update(points=[p for p in self.hard_points if p > self.lower and p < upper])
        val1, err = quad(fct, self.lower, upper, **args)
        if err < 1e-8:
            err = 1e-8
        assert abs(val - val1) < tol, "pm1({3:.9g}) -- Expected: {0:g}, computed: {1:g} (tol= {4:g})".format(val1, val, err, upper, tol)

    def _pm2(self, upper, tol):
        val = self.kernel.pm2(upper)

        def fct(x):
            return x * x * self.kernel.pdf(x)
        args = dict(self.quad_args)
        if self.hard_points:
            args.update(points=[p for p in self.hard_points if p > self.lower and p < upper])
        val1, err = quad(fct, self.lower, upper, **args)
        if err < 1e-8:
            err = 1e-8
        assert abs(val - val1) < tol, "pm2({3:.9g}) -- Expected: {0:g}, computed: {1:g}".format(val1, val, err, upper)

    def test_cdf_fct(self, tol=1e-5):
        for x in self.test_xs:
            yield self._cdf, x, tol

    def test_pm1_fct(self, tol=1e-5):
        for x in self.test_xs:
            yield self._pm1, x, tol

    def test_pm2_fct(self, tol=1e-5):
        for x in self.test_xs:
            yield self._pm2, x, tol

    def test_dct_fct(self):
        if hasattr(self.kernel, 'dct'):
            k = self.kernel
            p = k.pdf(self.pos_xs)
            cdt_ref = dct(p * (self.pos_xs[1] - self.pos_xs[0]))
            cdt_tst = k.dct(self.pos_freqs)
            np.testing.assert_array_almost_equal(cdt_ref, cdt_tst, 8)

    def test_fft_fct(self):
        if hasattr(self.kernel, 'fft'):
            k = self.kernel
            p = k.pdf(self.xs)
            fft_ref = fft(p * (self.xs[1] - self.xs[0])).real
            fft_tst = k.fft(self.freqs)
            np.testing.assert_array_almost_equal(fft_ref, fft_tst, 10)


class TestNormal1d(KernelTester):

    @classmethod
    def setUpClass(cls):
        cls.kernel = kernels.normal_kernel1d()
        cls.initUpClass()
        cls.norm_ref = stats.norm(loc=0, scale=1)

    def attr(self, attr):
        n_ref = self.norm_ref
        n_tst = self.kernel
        ref_vals = getattr(n_ref, attr)(self.xs)
        tst_vals = getattr(n_tst, attr)(self.xs)
        np.testing.assert_array_almost_equal(ref_vals, tst_vals, 10)

    def test_pdf(self):
        self.attr('pdf')

    def test_cdf(self):
        self.attr('cdf')

    def test_dct(self):
        n_ref = self.norm_ref
        n_tst = self.kernel
        p = n_ref.pdf(self.pos_xs)
        cdt_ref = dct(p * (self.pos_xs[1] - self.pos_xs[0]))
        cdt_tst = n_tst.dct(self.pos_freqs)
        np.testing.assert_array_almost_equal(cdt_ref, cdt_tst, 8)

    def test_fft(self):
        n_ref = self.norm_ref
        n_tst = self.kernel
        p = n_ref.pdf(self.xs)
        fft_ref = fft(p * (self.xs[1] - self.xs[0])).real
        fft_tst = n_tst.fft(self.freqs)
        np.testing.assert_array_almost_equal(fft_ref, fft_tst, 10)


class TestEpanechnikov(KernelTester):
    @classmethod
    def setUpClass(cls):
        cls.initUpClass()
        cls.kernel = kernels.Epanechnikov()


class TestTricube(KernelTester):
    @classmethod
    def setUpClass(cls):
        cls.initUpClass(lower=-5)
        cls.kernel = kernels.tricube()
        cls.hard_points = (-1 / _kernels.tricube_width, 1 / _kernels.tricube_width)


class TestEpanechnikov_order4(KernelTester):
    @classmethod
    def setUpClass(cls):
        cls.initUpClass(lower=-5)
        cls.kernel = kernels.Epanechnikov_order4()


class Testnormal_order4(KernelTester):
    @classmethod
    def setUpClass(cls):
        cls.initUpClass()
        cls.kernel = kernels.normal_order4()


class ComparePythonCython(object):
    @classmethod
    def initUpClass(cls):
        cls.xs = np.r_[-3:3:4096j]

    def _pdf(self, k):
        kernels.useCython()
        cy = k.pdf(self.xs)
        kernels.usePython()
        py = k.pdf(self.xs)
        np.testing.assert_array_almost_equal(cy, py, 7)

    def testing_pdf(self):
        for cls in kernels.kernels1D:
            yield self._pdf, cls

    def _cdf(self, k):
        kernels.useCython()
        cy = k.cdf(self.xs)
        kernels.usePython()
        py = k.cdf(self.xs)
        np.testing.assert_array_almost_equal(cy, py, 7)

    def testing_cdf(self):
        for cls in kernels.kernels1D:
            yield self._cdf, cls

    def _pm1(self, k):
        kernels.useCython()
        cy = k.pm1(self.xs)
        kernels.usePython()
        py = k.pm1(self.xs)
        np.testing.assert_array_almost_equal(cy, py, 7)

    def testing_pm1(self):
        for cls in kernels.kernels1D:
            yield self._pm1, cls

    def _pm2(self, k):
        kernels.useCython()
        cy = k.pm2(self.xs)
        kernels.usePython()
        py = k.pm2(self.xs)
        np.testing.assert_array_almost_equal(cy, py, 7)

    def testing_pm2(self):
        for cls in kernels.kernels1D:
            yield self._pm2, cls
