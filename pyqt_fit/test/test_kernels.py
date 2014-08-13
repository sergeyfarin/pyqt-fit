from __future__ import division, absolute_import, print_function

from .. import kernels
from .. import _kernels

from scipy import stats
import numpy.testing
import numpy as np
from . import kde_utils

class RefKernel1D(kernels.Kernel1D):
    def __init__(self, kernel):
        self.real_kernel = kernel

    def pdf(self, z, out=None):
        return self.real_kernel.pdf(z, out)

tol = 1e-8

class TestKernels(object):
    @classmethod
    def setUpClass(cls, lower=-np.inf, test_width=3):
        cls.lower = float(lower)
        cls.hard_points = ()
        cls.quad_args = dict(limit=100)
        cls.xs = np.r_[-test_width / 2:test_width / 2:17j]
        bw = 0.5
        R = 10
        N = 1024
        cls.fft_xs = np.roll((np.arange(N) - N / 2) * (2 * np.pi * bw / R), N // 2)
        cls.dct_xs = np.arange(N) * (np.pi * bw / R)
        cls.small = np.array([-5,-1,-0.5,0,0.5,1,5])

    def _cdf(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.cdf(self.xs)
        val = ker.cdf(self.xs)
        acc = kernel.precision_factor * tol
        np.testing.assert_allclose(val, ref, acc, acc)
        tot = ker.cdf([np.inf])
        assert abs(tot-1) < acc, "ker.cdf(inf) = {0}, while it should be close to 1".format(tot)
        short1 = ker.cdf(self.small)
        short2 = [float(ker.cdf(x)) for x in self.small]
        np.testing.assert_allclose(short1, short2, acc, acc)

    def cdf(self, kernel):
        self._cdf(kernel)
        if kernels.HAS_CYTHON:
            try:
                kernels.usePython()
                self._cdf(kernel)
            finally:
                kernels.useCython()

    def _pm1(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.pm1(self.xs)
        val = ker.pm1(self.xs)
        acc = kernel.precision_factor * tol
        np.testing.assert_allclose(val, ref, acc, acc)
        tot = ker.pm1(np.inf)
        assert abs(tot) < acc, "ker.cdf(inf) = {0}, while it should be close to 0".format(tot)
        short1 = ker.pm1(self.small)
        short2 = [float(ker.pm1(x)) for x in self.small]
        np.testing.assert_allclose(short1, short2, acc, acc)

    def pm1(self, kernel):
        self._pm1(kernel)
        if kernels.HAS_CYTHON:
            try:
                kernels.usePython()
                self._pm1(kernel)
            finally:
                kernels.useCython()

    def _pm2(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.pm2(self.xs)
        val = ker.pm2(self.xs)
        acc = kernel.precision_factor * tol
        np.testing.assert_allclose(val, ref, acc, acc)
        tot = ker.pm2(np.inf)
        assert abs(tot - kernel.var) < acc, "ker.cdf(inf) = {0}, while it should be close to {1}".format(tot, kernel.var)
        short1 = ker.pm2(self.small)
        short2 = [float(ker.pm2(x)) for x in self.small]
        np.testing.assert_allclose(short1, short2, acc, acc)

    def pm2(self, kernel):
        self._pm2(kernel)
        if kernels.HAS_CYTHON:
            try:
                kernels.usePython()
                self._pm2(kernel)
            finally:
                kernels.useCython()

    def _fft(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.fft(self.fft_xs)
        val = ker.fft(self.fft_xs)
        acc = kernel.precision_factor * tol
        np.testing.assert_allclose(val, ref, acc, acc)

    def fft(self, kernel):
        self._fft(kernel)
        if kernels.HAS_CYTHON:
            try:
                kernels.usePython()
                self._fft(kernel)
            finally:
                kernels.useCython()

    def _dct(self, kernel):
        ker = kernel.cls()
        ref_ker = RefKernel1D(ker)
        ref = ref_ker.dct(self.dct_xs)
        val = ker.dct(self.dct_xs)
        acc = kernel.precision_factor * tol
        np.testing.assert_allclose(val, ref, acc, acc)

    def dct(self, kernel):
        self._dct(kernel)
        if kernels.HAS_CYTHON:
            try:
                kernels.usePython()
                self._dct(kernel)
            finally:
                kernels.useCython()

    def test_cdf(self):
        for kernel in kde_utils.kernels1d:
            yield self.cdf, kernel

    def test_pm1(self):
        for kernel in kde_utils.kernels1d:
            yield self.pm1, kernel

    def test_pm2(self):
        for kernel in kde_utils.kernels1d:
            yield self.pm2, kernel

    def test_dct(self):
        for kernel in kde_utils.kernels1d:
            yield self.dct, kernel

    def test_fft(self):
        for kernel in kde_utils.kernels1d:
            yield self.fft, kernel


class TestNormal1d(object):
    @classmethod
    def setUpClass(cls, lower=-np.inf, test_width=3):
        cls.kernel = kernels.normal_kernel1d()
        cls.norm_ref = stats.norm(loc=0, scale=1)
        cls.xs = np.r_[-test_width / 2:test_width / 2:17j]

    def attr(self, attr):
        n_ref = self.norm_ref
        n_tst = self.kernel
        ref_vals = getattr(n_ref, attr)(self.xs)
        tst_vals = getattr(n_tst, attr)(self.xs)
        np.testing.assert_allclose(ref_vals, tst_vals, tol, tol)

    def python_attr(self, attr):
        ker = self.kernel
        ref = "_" + attr
        ref_vals = getattr(ker, ref)(self.xs)
        tst_vals = getattr(ker, attr)(self.xs)
        np.testing.assert_allclose(ref_vals, tst_vals, tol, tol)

    def test_pdf(self):
        self.attr('pdf')
        self.python_attr('pdf')

    def test_cdf(self):
        self.attr('cdf')
        self.python_attr('pdf')

    def test_pm1(self):
        self.python_attr('pm1')

    def test_pm2(self):
        self.python_attr('pm2')

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

class SimpleNormal(kernels.Kernel1D):
    def pdf(self, x, out=None):
        return kernels.kernels_imp.norm1d_pdf(x, out)

class TestDefaultKernelOps(object):
    @classmethod
    def setUpClass(cls):
        cls.xs = np.r_[-4:4:1024j]
        bw = 0.5
        R = 10
        N = 1024
        cls.fft_xs = np.roll((np.arange(N) - N / 2) * (2 * np.pi * bw / R), N // 2)
        cls.dct_xs = np.arange(N) * (np.pi * bw / R)
        cls.k_est = SimpleNormal()
        cls.k_ref = kernels.normal_kernel1d()

    def test_cdf(self):
        y_est = self.k_est.cdf(self.xs)
        y_ref = self.k_ref.cdf(self.xs)
        np.testing.assert_allclose(y_est, y_ref, tol, tol)

    def test_pm1(self):
        y_est = self.k_est.pm1(self.xs)
        y_ref = self.k_ref.pm1(self.xs)
        np.testing.assert_allclose(y_est, y_ref, tol, tol)

    def test_pm2(self):
        y_est = self.k_est.pm2(self.xs)
        y_ref = self.k_ref.pm2(self.xs)
        np.testing.assert_allclose(y_est, y_ref, tol, tol)

    def test_fft(self):
        y_est = self.k_est.fft(self.fft_xs)
        y_ref = self.k_ref.fft(self.fft_xs)
        np.testing.assert_allclose(y_est, y_ref, tol, tol)

    def test_dct(self):
        y_est = self.k_est.dct(self.dct_xs)
        y_ref = self.k_ref.dct(self.dct_xs)
        np.testing.assert_allclose(y_est, y_ref, tol, tol)

