from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from . import pool


def avg_pool_serial(value, ksize, stride, padding='SAME',
                    dtype=None, sum_dtype=None, name=None, par=1,
                    force_div=False,
                    value_ram_size=None, out_ram_size=None,
                    value_dtype=None):

    return pool.avg_pool(value, ksize, stride, padding,
                         dtype, sum_dtype, name, par,
                         force_div,
                         value_ram_size, out_ram_size,
                         value_dtype)


def max_pool_serial(value, ksize, stride, padding='SAME',
                    dtype=None, name=None, par=1,
                    value_ram_size=None, out_ram_size=None,
                    value_dtype=None):

    return pool.max_pool(value, ksize, stride, padding,
                         dtype, name, par,
                         value_ram_size, out_ram_size,
                         value_dtype)
