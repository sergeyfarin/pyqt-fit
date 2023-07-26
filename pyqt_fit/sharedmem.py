"""
Module containing utility classes and functions.
"""

from __future__ import division, absolute_import
import ctypes
import multiprocessing as mp
import numpy as np
from .compat import irange

CTYPES_CHAR_LIST = [ctypes.c_char,
                    ctypes.c_wchar
                    ]

CTYPES_INT_LIST = [ctypes.c_byte,
                   ctypes.c_short,
                   ctypes.c_int,
                   ctypes.c_long,
                   ctypes.c_longlong
                   ]

CTYPES_UINT_LIST = [ctypes.c_ubyte,
                    ctypes.c_ushort,
                    ctypes.c_uint,
                    ctypes.c_ulong,
                    ctypes.c_ulonglong
                    ]

CTYPES_FLOAT_LIST = [ctypes.c_float,
                     ctypes.c_double,
                     ctypes.c_longdouble
                     ]

CTYPES_TO_NUMPY = {ctypes.c_char: np.dtype(np.character),
                   ctypes.c_wchar: np.dtype(np.unicode_),
                   }


def _get_ctype_size(ct):
    return ctypes.sizeof(ct)

for t in CTYPES_INT_LIST:
    CTYPES_TO_NUMPY[t] = np.dtype("=i{:d}".format(_get_ctype_size(t)))

for t in CTYPES_UINT_LIST:
    CTYPES_TO_NUMPY[t] = np.dtype("=u{:d}".format(_get_ctype_size(t)))

for t in CTYPES_FLOAT_LIST:
    CTYPES_TO_NUMPY[t] = np.dtype("=f{:d}".format(_get_ctype_size(t)))

NUMPY_TO_CTYPES = {CTYPES_TO_NUMPY[t]: t for t in CTYPES_TO_NUMPY}


class _dummy(object):
    pass


def _shmem_as_ndarray(raw_array, shape=None, order='C'):
    address = ctypes.addressof(raw_array)
    length = len(raw_array)
    size = ctypes.sizeof(raw_array)
    item_size = size // length

    if shape is None:
        shape = (length,)
    else:
        assert np.prod(shape) == length
    dtype = CTYPES_TO_NUMPY.get(raw_array._type_, None)
    if dtype is None:
        raise TypeError("Unknown conversion from {} to numpy type".format(raw_array._type_))
    strides = tuple(item_size * np.prod(shape[i + 1:], dtype=int) for i in irange(len(shape)))
    if order != 'C':
        strides = strides[::-1]
    d = _dummy()
    d.__array_interface__ = {'data': (address, False),
                             'typestr': dtype.str,
                             'desc': dtype.descr,
                             'shape': shape,
                             'strides': strides,
                             }
    return np.asarray(d)


def _allocate_raw_array(size, dtype):
    dtype = np.dtype(dtype)
    ct = NUMPY_TO_CTYPES.get(dtype)
    if ct is None:
        raise TypeError("Error, cannot convert numpy type {} into ctypes".format(dtype))
    return mp.RawArray(ct, int(size))


class SharedArray(object):
    def __init__(self, ra, shape=None, order='C'):
        self._ra = ra
        if ra is not None:
            self._np = _shmem_as_ndarray(self._ra, shape, order)
            self._shape = shape
        else:
            self._np = None
            self._shape = None

    def _get_np(self):
        return self._np

    def _set_np(self, content):
        self.np[:] = content

    np = property(_get_np, _set_np)

    def _get_ra(self):
        return self._ra

    ra = property(_get_ra)

    def __getinitargs__(self):
        return (self._ra, self._shape)


def array(content, dtype=None, order=None, ndmin=0):
    content = np.asarray(content)
    if dtype is None:
        dtype = content.dtype
    ra = _allocate_raw_array(np.prod(content.shape), dtype)
    shape = tuple(content.shape)
    if ndmin > len(shape):
        shape = (1,) * (ndmin - len(shape)) + shape
    sa = SharedArray(ra, shape)
    sa.np = content
    return sa


def ones(shape, dtype=float, order=None):
    ra = _allocate_raw_array(np.prod(shape), dtype)
    sa = SharedArray(ra, shape)
    sa.np = 1
    return sa


def zeros(shape, dtype=float, order=None):
    ra = _allocate_raw_array(np.prod(shape), dtype)
    sa = SharedArray(ra, shape)
    return sa
