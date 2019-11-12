from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.util as util


def avg_pool(value, ksize, stride, padding='SAME',
             dtype=None, sum_dtype=None, name=None, par=1,
             force_div=False,
             value_ram_size=None, out_ram_size=None,
             value_dtype=None):

    ksize_row = ksize[1]
    ksize_col = ksize[2]
    stride_row = stride[2]
    stride_col = stride[2]

    if isinstance(padding, str) and padding == 'SAME':
        pad_col, pad_col_left, pad_col_right = util.pad_size_split(
            value.shape[2], ksize_col, stride_col)
        pad_row, pad_row_top, pad_row_bottom = util.pad_size_split(
            value.shape[1], ksize_row, stride_row)

        out_shape = (value.shape[0],
                     util.pix_size(value.shape[1], ksize_row, stride_row, padding),
                     util.pix_size(value.shape[2], ksize_col, stride_col, padding),
                     value.shape[3])

    elif isinstance(padding, str) and padding == 'VALID':
        pad_col, pad_col_left, pad_col_right = 0, 0, 0
        pad_row, pad_row_top, pad_row_bottom = 0, 0, 0

        out_shape = (value.shape[0],
                     util.pix_size(value.shape[1], ksize_row, stride_row, 'VALID'),
                     util.pix_size(value.shape[2], ksize_col, stride_col, 'VALID'),
                     value.shape[3])

    elif isinstance(padding, int):
        pad_col, pad_col_left, pad_col_right = padding * 2, padding, padding
        pad_row, pad_row_top, pad_row_bottom = padding * 2, padding, padding

        out_shape = (value.shape[0],
                     util.pix_size(value.shape[1] + padding * 2,
                                   ksize_row, stride_row, 'VALID'),
                     util.pix_size(value.shape[2] + padding * 2,
                                   ksize_col, stride_col, 'VALID'),
                     value.shape[3])

    elif isinstance(padding, (tuple, list)):
        pad_col, pad_col_left, pad_col_right = padding[2] + padding[3], padding[2], padding[3]
        pad_row, pad_row_top, pad_row_bottom = padding[0] + padding[1], padding[0], padding[1]

        out_shape = (value.shape[0],
                     util.pix_size(value.shape[1] + padding[0] + padding[1],
                                   ksize_row, stride_row, 'VALID'),
                     util.pix_size(value.shape[2] + padding[2] + padding[3],
                                   ksize_col, stride_col, 'VALID'),
                     value.shape[3])

    else:
        raise ValueError("padding options must be 'SAME', 'VALID', int, tuple, or list.")

    out = np.zeros(out_shape, dtype=np.int64)

    value = np.pad(value, [(0, 0),
                           (pad_row_top, pad_row_bottom),
                           (pad_col_left, pad_col_right),
                           (0, 0)], 'constant')

    value_point = 0 if value_dtype is None else value_dtype.point
    out_point = value_point if dtype is None else dtype.point
    div_shift = out_point - value_point

    div_op = (lambda x: x << div_shift if div_shift >= 0 else
              lambda x: x >> -div_shift)

    num_vars = ksize_col * ksize_row
    if force_div or num_vars & (num_vars - 1) != 0:
        def divider(x): return (x / num_vars).astype(np.int64)
    else:
        def divider(x): return x // num_vars

    for bat in range(value.shape[0]):

        oy = 0
        for py in range(-pad_row_top, value.shape[1] + pad_row_bottom, stride_row):

            ox = 0
            for px in range(-pad_col_left, value.shape[2] + pad_col_right, stride_col):

                ys = py + pad_row_top
                ye = ys + ksize_row
                xs = px + pad_col_left
                xe = xs + ksize_col
                a = value[bat, ys: ye, xs: xe]
                a = np.add.reduce(a, axis=0)
                sum = np.add.reduce(a, axis=0)

                sum += (num_vars // 2)
                div = divider(sum)

                out[bat][oy][ox][:] = div_op(div)

                ox += 1
                if ox >= out.shape[2]:
                    break

            oy += 1
            if oy >= out.shape[1]:
                break

    return out


def max_pool(value, ksize, stride, padding='SAME',
             dtype=None, name=None, par=1,
             value_ram_size=None, out_ram_size=None,
             value_dtype=None):

    ksize_row = ksize[1]
    ksize_col = ksize[2]
    stride_row = stride[2]
    stride_col = stride[2]

    if isinstance(padding, str) and padding == 'SAME':
        pad_col, pad_col_left, pad_col_right = util.pad_size_split(
            value.shape[2], ksize_col, stride_col)
        pad_row, pad_row_top, pad_row_bottom = util.pad_size_split(
            value.shape[1], ksize_row, stride_row)

        out_shape = (value.shape[0],
                     util.pix_size(value.shape[1], ksize_row, stride_row, padding),
                     util.pix_size(value.shape[2], ksize_col, stride_col, padding),
                     value.shape[3])

    elif isinstance(padding, str) and padding == 'VALID':
        pad_col, pad_col_left, pad_col_right = 0, 0, 0
        pad_row, pad_row_top, pad_row_bottom = 0, 0, 0

        out_shape = (value.shape[0],
                     util.pix_size(value.shape[1], ksize_row, stride_row, 'VALID'),
                     util.pix_size(value.shape[2], ksize_col, stride_col, 'VALID'),
                     value.shape[3])

    elif isinstance(padding, int):
        pad_col, pad_col_left, pad_col_right = padding * 2, padding, padding
        pad_row, pad_row_top, pad_row_bottom = padding * 2, padding, padding

        out_shape = (value.shape[0],
                     util.pix_size(value.shape[1] + padding * 2,
                                   ksize_row, stride_row, 'VALID'),
                     util.pix_size(value.shape[2] + padding * 2,
                                   ksize_col, stride_col, 'VALID'),
                     value.shape[3])

    elif isinstance(padding, (tuple, list)):
        pad_col, pad_col_left, pad_col_right = padding[2] + padding[3], padding[2], padding[3]
        pad_row, pad_row_top, pad_row_bottom = padding[0] + padding[1], padding[0], padding[1]

        out_shape = (value.shape[0],
                     util.pix_size(value.shape[1] + padding[0] + padding[1],
                                   ksize_row, stride_row, 'VALID'),
                     util.pix_size(value.shape[2] + padding[2] + padding[3],
                                   ksize_col, stride_col, 'VALID'),
                     value.shape[3])

    else:
        raise ValueError("padding options must be 'SAME', 'VALID', int, tuple, or list.")

    out = np.zeros(out_shape, dtype=np.int64)

    if value_dtype is not None:
        pad_value_shift = value_dtype.width
    elif dtype is not None:
        pad_value_shift = dtype.width
    else:
        pad_value_shift = 32
    pad_value = (-1) * (1 << (pad_value_shift - 1))

    value = np.pad(value, [(0, 0),
                           (pad_row_top, pad_row_bottom),
                           (pad_col_left, pad_col_right),
                           (0, 0)], 'constant',
                   constant_values=pad_value)

    value_point = 0 if value_dtype is None else value_dtype.point
    out_point = value_point if dtype is None else dtype.point
    max_shift = out_point - value_point

    max_op = ((lambda x: x << max_shift) if max_shift >= 0 else
              (lambda x: x >> -max_shift))

    for bat in range(value.shape[0]):

        oy = 0
        for py in range(-pad_row_top, value.shape[1] + pad_row_bottom, stride_row):

            ox = 0
            for px in range(-pad_col_left, value.shape[2] + pad_col_right, stride_col):

                ys = py + pad_row_top
                ye = ys + ksize_row
                xs = px + pad_col_left
                xe = xs + ksize_col
                a = value[bat, ys: ye, xs: xe]
                a = np.max(a, axis=0)
                max_val = np.max(a, axis=0)

                out[bat][oy][ox][:] = max_val

                ox += 1
                if ox >= out.shape[2]:
                    break

            oy += 1
            if oy >= out.shape[1]:
                break

    return out
