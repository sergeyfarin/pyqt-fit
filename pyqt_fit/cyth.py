# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 20:11:03 2012

@author: ips_user
"""

import os
import numpy
import pyximport

if os.name == 'nt':
    if os.environ.has_key('CPATH'):
        os.environ['CPATH'] = os.environ['CPATH'] + numpy.get_include()
    else:
        os.environ['CPATH'] = numpy.get_include()

    # XXX: we're assuming that MinGW is installed in C:\MinGW (default)
    if os.environ.has_key('PATH'):
        os.environ['PATH'] = os.environ['PATH'] + r';C:\MinGW\bin'
    else:
        os.environ['PATH'] = r'C:\MinGW\bin'

    mingw_setup_args = { 'options': { 'build_ext': { 'compiler': 'mingw32' } } }
    pyximport.install(setup_args=mingw_setup_args, reload_support=True)

elif os.name == 'posix':
    extra_flags = '-I' + numpy.get_include()
    os.environ['CFLAGS'] = " ".join([os.environ.get('CFLAGS', ''), extra_flags])
    os.environ['CXXFLAGS'] = " ".join([os.environ.get('CXXFLAGS', ''), extra_flags])

    pyximport.install(reload_support=True)
