from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def pad(value, padding,
        dtype=None, name=None, par=1,
        value_ram_size=None, out_ram_size=None,
        value_dtype=None):

    ksize_row = 1
    ksize_col = 1
    stride_row = 1
    stride_col = 1

    if isinstance(padding, str) and padding == 'SAME':
        pad_col, pad_col_left, pad_col_right = util.pad_size_split(
            value.shape[2], ksize_col, stride_col)
        pad_row, pad_row_top, pad_row_bottom = util.pad_size_split(
            value.shape[1], ksize_row, stride_row)

    elif isinstance(padding, str) and padding == 'VALID':
        pad_col, pad_col_left, pad_col_right = 0, 0, 0
        pad_row, pad_row_top, pad_row_bottom = 0, 0, 0

    elif isinstance(padding, int):
        pad_col, pad_col_left, pad_col_right = padding * 2, padding, padding
        pad_row, pad_row_top, pad_row_bottom = padding * 2, padding, padding

    elif isinstance(padding, (tuple, list)):
        pad_col, pad_col_left, pad_col_right = padding[2] + padding[3], padding[2], padding[3]
        pad_row, pad_row_top, pad_row_bottom = padding[0] + padding[1], padding[0], padding[1]

    else:
        raise ValueError("padding options must be 'SAME', 'VALID', int, tuple, or list.")

    out = np.pad(value, [(0, 0),
                         (pad_row_top, pad_row_bottom),
                         (pad_col_left, pad_col_right),
                         (0, 0)], 'constant')

    return out
