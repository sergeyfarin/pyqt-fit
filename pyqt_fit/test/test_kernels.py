from __future__ import division, absolute_import, print_function

import unittest
from .. import kernels

from scipy import stats
from scipy.fftpack import fft, dct
import numpy as np

class TestNormal1d(unittest.TestCase):
    def setUp(self):
        self.pos_xs = 20 * (np.arange(1024) + 0.5)/ 1024
        self.xs = np.roll(20*(np.arange(1024)-512)/1024, 512)
        self.norm_ref = stats.norm(loc=0, scale=1)
        self.norm_tst = kernels.normal_kernel1d()
        self.pos_freqs = np.pi/20*np.arange(1024)
        freqs = 2*np.pi/20*(np.arange(1024)-512)
        freqs = np.roll(freqs, 512)
        self.freqs = freqs

    def attr(self, attr):
        n_ref = self.norm_ref
        n_tst = self.norm_tst
        ref_vals = getattr(n_ref, attr)(self.xs)
        tst_vals = getattr(n_tst, attr)(self.xs)
        np.testing.assert_array_almost_equal(ref_vals, tst_vals, 10)

    def test_pdf(self):
        self.attr('pdf')

    def test_cdf(self):
        self.attr('cdf')

    def test_dct(self):
        n_ref = self.norm_ref
        n_tst = self.norm_tst
        p = n_ref.pdf(self.pos_xs)
        cdt_ref = dct(p * (self.pos_xs[1] - self.pos_xs[0]))
        cdt_tst = n_tst.dct(self.pos_freqs)
        np.testing.assert_array_almost_equal(cdt_ref, cdt_tst, 8)

    def test_fft(self):
        n_ref = self.norm_ref
        n_tst = self.norm_tst
        p = n_ref.pdf(self.xs)
        fft_ref = fft(p*(self.xs[1] - self.xs[0])).real
        fft_tst = n_tst.fft(self.freqs)
        np.testing.assert_array_almost_equal(fft_ref, fft_tst, 10)

