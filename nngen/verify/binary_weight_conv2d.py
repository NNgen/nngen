from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from .conv2d import conv2d


def binary_weight_conv2d(input, filter, strides,
                         bias=None, scale=None,
                         rshift_mul=None, rshift_sum=None, rshift_out=None,
                         act_func=None, padding='SAME', asymmetric_clip=False,
                         dtype=None, mul_dtype=None, sum_dtype=None,
                         name=None,
                         par_ich=1, par_och=1, par_col=1, par_row=1,
                         concur_och=None, stationary='filter',
                         input_ram_size=None, filter_ram_size=None,
                         bias_ram_size=None, scale_ram_size=None,
                         vshamt_mul_ram_size=None,
                         vshamt_sum_ram_size=None,
                         vshamt_out_ram_size=None,
                         out_ram_size=None,
                         disable_keep_input=False,
                         input_dtype=None, filter_dtype=None,
                         bias_dtype=None, scale_dtype=None):

    bin_filter = np.where(filter == 0, -1, 1)

    return conv2d(input, bin_filter, strides,
                  bias, scale,
                  rshift_mul, rshift_sum, rshift_out,
                  act_func, padding, asymmetric_clip,
                  dtype, mul_dtype, sum_dtype,
                  name,
                  par_ich, par_och, par_col, par_row,
                  concur_och, stationary,
                  input_ram_size, filter_ram_size,
                  bias_ram_size, scale_ram_size,
                  vshamt_mul_ram_size,
                  vshamt_sum_ram_size,
                  vshamt_out_ram_size,
                  out_ram_size,
                  disable_keep_input,
                  input_dtype, filter_dtype,
                  bias_dtype, scale_dtype)
