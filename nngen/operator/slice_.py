from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.basic_types as bt
import nngen.util as util


class slice_(bt._Operator):
    """
    Create a sliced tensor with a similar API to the numpy slice.
    """

    def __init__(self, value, begins, ends, strides,
                 dtype=None, name=None, par=1):

        if not isinstance(begins, (tuple, list)):
            raise TypeError('begins must be tuple or list.')

        if not isinstance(ends, (tuple, list)):
            raise TypeError('ends must be tuple or list.')

        if not isinstance(strides, (tuple, list)):
            raise TypeError('strides must be tuple or list.')

        if len(value.shape) != len(begins):
            raise ValueError('length mismatch between value.shape and begins: %d != %d' %
                             (len(value.shape), len(begins)))

        if len(value.shape) != len(ends):
            raise ValueError('length mismatch between value.shape and ends: %d != %d' %
                             (len(value.shape), len(ends)))

        if len(value.shape) != len(strides):
            raise ValueError('length mismatch between value.shape and strides: %d != %d' %
                             (len(value.shape), len(strides)))

        for begin in begins:
            begin = int(begin)
            if not isinstance(begin, int):
                raise TypeError('values of begins must be int, not %s' % str(type(begin)))

        for end in ends:
            end = int(end)
            if not isinstance(end, int):
                raise TypeError('values of ends must be int, not %s' % str(type(end)))

        for stride in strides:
            stride = int(stride)
            if not isinstance(stride, int):
                raise TypeError('values of strides must be int, not %s' % str(type(stride)))

        if par != 1:
            raise ValueError("par must be 1 in the current implementation.")

        slices = to_slices(begins, ends, strides)
        shape = np.zeros(value.shape)[slices].shape

        bt._Operator.__init__(self, value,
                              dtype=dtype, shape=shape, name=name, par=par)

        self.begins = tuple(begins)
        self.ends = tuple(ends)
        self.strides = tuple(strides)

        raise NotImplementedError()


def to_slices(begins, ends, strides):
    slices = []

    for begin, end, stride in zip(begins, ends, strides):
        slices.append(slice(begin, end, stride))

    return tuple(slices)
