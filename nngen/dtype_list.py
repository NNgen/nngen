from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math

base_order = ['int', 'fixed']


class dtype_info(object):

    def __repr__(self):
        return '<dtype %s%s%d%s>' % ('' if self.signed else 'u',
                                     self.base, self.width,
                                     '' if self.base == 'int' else '_%d' % self.point)

    def to_str(self):
        return '%s%s%d%s' % ('' if self.signed else 'u',
                             self.base, self.width,
                             '' if self.base == 'int' else '_%d' % self.point)

    def __hash__(self):
        return hash((self.base, self.width, self.point, self.signed))

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return ((self.base, self.width, self.point, self.signed) ==
                (other.base, other.width, other.point, other.signed))

    def __init__(self, base, width, point=0, signed=True):
        if base != 'fixed' and point != 0:
            raise ValueError("point must be 0 for dtype '%s'" % base)

        self.base = base
        self.width = width
        self.point = point
        self.signed = signed

    @property
    def wordsize(self):
        return int(math.ceil(self.width / 8))


def dtype_int(width=None, signed=True):
    base = 'int'
    if width is None:
        width = 32
    point = 0
    return dtype_info(base, width, point, signed)


def dtype_fixed(width=None, point=None, signed=True):
    base = 'fixed'
    if width is None:
        width = 32
    if point is None:
        point = 0
    return dtype_info(base, width, point, signed)


def get_max_dtype(*args):
    dtype = None
    for arg in args:
        if arg.dtype is None:
            continue

        if dtype is None:
            dtype = arg.dtype
            continue

        base = base_order[max(base_order.index(arg.dtype.base),
                              base_order.index(dtype.base))]
        width = max(arg.dtype.width, dtype.width)
        point = max(arg.dtype.point, dtype.point)
        signed = arg.dtype.signed or dtype.signed
        dtype = dtype_info(base, width, point, signed)

    return dtype


int64 = dtype_int(64, True)
int32 = dtype_int(32, True)
int16 = dtype_int(16, True)
int8 = dtype_int(8, True)
int4 = dtype_int(4, True)
int2 = dtype_int(2, True)
uint64 = dtype_int(64, False)
uint32 = dtype_int(32, False)
uint16 = dtype_int(16, False)
uint8 = dtype_int(8, False)
uint4 = dtype_int(4, False)
uint2 = dtype_int(2, False)
uint1 = dtype_int(1, False)

fixed32_16 = dtype_fixed(32, 16, True)
fixed16_8 = dtype_fixed(16, 8, True)
fixed8_4 = dtype_fixed(8, 4, True)
