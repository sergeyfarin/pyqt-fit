# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 20:11:03 2012

@author: ips_user
"""

import os
import numpy
import pyximport

if os.name == 'nt':
    if 'CPATH' in os.environ:
        os.environ['CPATH'] = os.environ['CPATH'] + numpy.get_include()
    else:
        os.environ['CPATH'] = numpy.get_include()

    # XXX: we're assuming that MinGW is installed in C:\MinGW (default)
    if 'PATH' in os.environ:
        os.environ['PATH'] = os.environ['PATH'] + ';C:\MinGW\bin'
    else:
        os.environ['PATH'] = 'C:\MinGW\bin'

    mingw_setup_args = { 'options': { 'build_ext': { 'compiler': 'mingw32' } } }
    pyximport.install(setup_args=mingw_setup_args, reload_support=True)

elif os.name == 'posix':
    extra_flags = ' -I' + numpy.get_include()
    if 'CFLAGS' in os.environ:
        os.environ['CFLAGS'] = os.environ['CFLAGS'] + extra_flags
    else:
        os.environ['CFLAGS'] = extra_flags

    pyximport.install(reload_support=True)
