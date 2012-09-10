#!/usr/bin/env python

from distutils.core import setup

setup(name='PyQt-Fit',
      version='1.0',
      description='Last-square fitting of user-defined functions',
      author='Pierre Barbier de Reuille',
      author_email='pierre.barbierdereuille@gmail.com',
      packages= [''],
      requires=['scipy', 'numpy', 'cython', 'pylab', 'PyQT4', 'matplotlib'],
     )