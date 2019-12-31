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
                 dtype=None, name=None, par=1,
                 value_ram_size=None, out_ram_size=None):

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

        if value_ram_size is not None and value_ram_size < 1:
            raise ValueError('value_ram_size must be greater than 0')

        if out_ram_size is not None and out_ram_size < 1:
            raise ValueError('out_ram_size must be greater than 0')

        # deligate a shape calculation to numpy
        slices = to_slices(begins, ends, strides)
        shape = np.zeros(value.shape)[slices].shape

        bt._Operator.__init__(self, value,
                              dtype=dtype, shape=shape, name=name, par=par)

        self.begins = tuple(begins)
        self.ends = tuple(ends)
        self.strides = tuple(strides)

        # attribute
        self.value_ram_size = value_ram_size
        self.out_ram_size = out_ram_size
        slice_.attribute(self, par, value_ram_size, out_ram_size)

    def eval(self, memo, input_dict, **kwargs):
        if id(self) in memo:
            return memo[id(self)]

        import nngen.verify as verify

        name = self.__class__.__name__
        method = getattr(verify, name, None)

        args = [arg.eval(memo, input_dict)
                for arg in self.args]

        value = args[0]

        kwargs['begins'] = self.begins
        kwargs['ends'] = self.ends
        kwargs['strides'] = self.strides
        kwargs['dtype'] = self.dtype
        kwargs['name'] = self.name
        kwargs['par'] = self.par

        ret = method(value, **kwargs)
        memo[id(self)] = ret

        return ret

    def attribute(self, par=None, value_ram_size=None, out_ram_size=None):
        if par is not None:
            if (par - 1) & par != 0:
                raise ValueError('par must be power of 2.')

            self.par = par

            for arg in self.args:
                arg.add_alignment_request(self.par)

            self.add_alignment_request(self.par)

        if value_ram_size is not None:
            if value_ram_size < 1:
                raise ValueError('value_ram_size must be greater than 0')

            self.value_ram_size = value_ram_size

        if out_ram_size is not None:
            if out_ram_size < 1:
                raise ValueError('out_ram_size must be greater than 0')

            self.out_ram_size = out_ram_size


def to_slices(begins, ends, strides):
    slices = []

    for begin, end, stride in zip(begins, ends, strides):
        slices.append(slice(begin, end, stride))

    return tuple(slices)
