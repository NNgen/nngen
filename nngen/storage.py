from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.basic_types as bt


class placeholder(bt._Storage):

    def __init__(self, dtype, shape=None, name=None):
        bt._Storage.__init__(self,
                             dtype=dtype, shape=shape, name=name,
                             is_input=True)


class variable(bt._Storage):

    def __init__(self, dtype, shape=None, name=None):
        bt._Storage.__init__(self,
                             dtype=dtype, shape=shape, name=name,
                             is_input=False)


class constant(bt._Constant):

    def __init__(self, value, dtype=None, shape=None, name=None):
        bt._Constant.__init__(self, value,
                              dtype=dtype, shape=shape, name=name)


class zeros(constant):

    def __init__(self, shape, dtype=None, name=None):
        value = np.zeros(shape, dtype=np.int64)
        constant.__init__(self, value, dtype, shape, name)


def zeros_like(x, dtype=None, name=None):
    shape = x.shape
    return zeros(shape, dtype, name)


class ones(constant):

    def __init__(self, shape, dtype=None, name=None):
        value = np.ones(shape, dtype=np.int64)
        if dtype is not None and dtype.point > 0:
            value = value << [dtype.point]
        if dtype is not None and dtype.point < 0:
            value = value >> [-dtype.point]
        constant.__init__(self, value, dtype, shape, name)


def ones_like(x, dtype=None, name=None):
    shape = x.shape
    return ones(shape, dtype, name)


class full(constant):

    def __sub_str__(self):
        return ' fill_value:%d' % self.fill_value

    def __init__(self, shape, fill_value, dtype=None, name=None):
        value = np.full(shape, fill_value, dtype=np.int64)
        self.fill_value = fill_value
        if dtype is not None and dtype.point > 0:
            value = value << [dtype.point]
        if dtype is not None and dtype.point < 0:
            value = value >> [-dtype.point]
        constant.__init__(self, value, dtype, shape, name)


def full_like(x, fill_value, dtype=None, name=None):
    shape = x.shape
    return full(shape, fill_value, dtype, name)
