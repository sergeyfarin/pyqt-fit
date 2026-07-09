#!/usr/bin/env python

from __future__ import print_function

import os

from setuptools import find_packages, setup


ROOT = os.path.dirname(os.path.abspath(__file__))


def read_file(filename):
    with open(os.path.join(ROOT, filename)) as stream:
        return stream.read()


VERSION = read_file(os.path.join('pyqt_fit', 'version.txt')).strip()

setup(
    name='PyQt-Fit',
    version=VERSION,
    description='Parametric and non-parametric regression, with plotting and testing methods.',
    long_description=read_file('README.txt'),
    long_description_content_type='text/plain',
    author='Pierre Barbier de Reuille',
    author_email='pierre.barbierdereuille@gmail.com',
    url='https://github.com/sergeyfarin/pyqt-fit',
    license='GPL-3.0-or-later',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'pyqt_fit': [
            'qt_fit.ui',
            'version.txt',
            '*.pyx',
            '*.pxd',
        ],
    },
    install_requires=[
        'numpy>=1.5.0; python_version < "3"',
        'numpy>=1.23.0; python_version >= "3"',
        'scipy>=0.10.0; python_version < "3"',
        'scipy>=1.9.0; python_version >= "3"',
        'matplotlib>=1.2.0; python_version < "3"',
        'matplotlib>=3.6.0; python_version >= "3"',
    ],
    extras_require={
        'cython': ['Cython>=0.17'],
        'gui': [
            'PySide6>=6.8; python_version >= "3.10"',
            'PySide2>=5.15.2; python_version < "3.10"',
        ],
        'gui-qt6': ['PySide6>=6.8; python_version >= "3.10"'],
        'gui-qt5': ['PySide2>=5.15.2; python_version < "3.11"'],
        'test': [
            'pytest>=7; python_version >= "3.7"',
            'pytest<5; python_version < "3.7"',
        ],
        'coverage': [
            'pytest>=7; python_version >= "3.7"',
            'pytest-cov>=4; python_version >= "3.7"',
            'pytest<5; python_version < "3.7"',
            'pytest-cov<3; python_version < "3.7"',
        ],
    },
    entry_points={
        'console_scripts': [
            'pyqt_fit1d=pyqt_fit.pyqt_fit1d:run',
        ],
    },
    python_requires='>=2.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: X11 Applications :: Qt',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
)
