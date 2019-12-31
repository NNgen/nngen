from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import nngen.basic_types as bt
import nngen.util as util


class slice(bt._Operator):
    """
    Create a sliced tensor with a similar API to the numpy slice.
    """

    def __init__(self, value, starts, ends, steps,
                 dtype=None, name=None):

        if not isinstance(starts, (tuple, list)):
            raise TypeError('starts must be tuple or list.')

        if not isinstance(ends, (tuple, list)):
            raise TypeError('ends must be tuple or list.')

        if not isinstance(steps, (tuple, list)):
            raise TypeError('steps must be tuple or list.')

        if len(value.shape) != len(starts):
            raise ValueError('length mismatch between value.shape and starts: %d != %d' %
                             (len(value.shape), len(starts)))

        if len(value.shape) != len(ends):
            raise ValueError('length mismatch between value.shape and ends: %d != %d' %
                             (len(value.shape), len(ends)))

        if len(value.shape) != len(steps):
            raise ValueError('length mismatch between value.shape and steps: %d != %d' %
                             (len(value.shape), len(steps)))

        raise NotImplementedError()
