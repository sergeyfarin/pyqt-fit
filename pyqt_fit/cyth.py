# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 20:11:03 2012

@author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>
"""

from __future__ import absolute_import, print_function
import os
import numpy

try:
    import pyximport
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

def addFlags(var, flags, sep = ' '):
    if var in os.environ:
        flags = [os.environ[var]] + flags
    os.environ[var] = sep.join(flags)

if HAS_CYTHON:
    USE_MINGW=False
    if os.name == 'nt':
        addFlags('CPATH', [numpy.get_include()], ';')

        mingw_setup_args = dict(options={})

        if USE_MINGW:
            addFlags('PATH', [r'C:\MinGW\bin'], ';')
            mingw_setup_args['options']['build_ext'] = {'compiler': 'mingw32'}

        pyximport.install(setup_args=mingw_setup_args,reload_support=True)

    elif os.name == 'posix':
        extra_flags = ['-I' + numpy.get_include()]
        addFlags('CFLAGS', extra_flags)
        addFlags('CXXFLAGS', extra_flags)

        pyximport.install(reload_support=True)

