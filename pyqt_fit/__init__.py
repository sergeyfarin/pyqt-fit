"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This package is designed to perform n-D least-square fitting of user-defined
functions. It also provides a GUI that can use pre-defined fitting methods.
"""

__all__ = ['bootstrap', 'plot_fit', 'curve_fitting', 'kernel_smoothing']

import functions
import residuals
from plot_fit import fit, write1d, plot1d
from curve_fitting import curve_fit
from path import path

with (path(__file__).dirname() / 'version.txt').open() as f:
    __version__ = f.read().strip()

