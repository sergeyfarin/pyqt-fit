"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This package is designed to perform n-D least-square fitting of user-defined
functions. It also provides a GUI that can use pre-defined fitting methods.
"""

from __future__ import absolute_import, print_function

__all__ = ['bootstrap', 'plot_fit',
           'curve_fitting', 'nonparam_regression',
           'functions', 'residuals', 'CurveFitting']

from . import functions, residuals
from . import bootstrap, plot_fit, curve_fitting, nonparam_regression
from .curve_fitting import CurveFitting
from path import path

with (path(__file__).dirname() / 'version.txt').open() as f:
    __version__ = f.read().strip()
