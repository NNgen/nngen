from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import math

import nngen.util as util


def conv2d(input, filter, strides,
           bias=None, scale=None,
           rshift_mul=None, rshift_sum=None, rshift_out=None,
           act_func=None, padding='SAME',
           asymmetric_clip=False,
           dtype=None, mul_dtype=None, sum_dtype=None,
           name=None,
           par_ich=1, par_och=1, par_col=1, par_row=1,
           concur_och=None, stationary='filter',
           input_ram_size=None, filter_ram_size=None,
           bias_ram_size=None, scale_ram_size=None,
           vshamt_mul_ram_size=None, vshamt_sum_ram_size=None, vshamt_out_ram_size=None,
           out_ram_size=None,
           disable_keep_input=False,
           input_shape=None, filter_shape=None, out_shape=None,
           input_dtype=None, filter_dtype=None,
           bias_dtype=None, scale_dtype=None,
           vshamt_mul_dtype=None, vshamt_sum_dtype=None, vshamt_out_dtype=None):

    # opposite order to pool
    if isinstance(padding, str) and padding == 'SAME':
        pad_col, pad_col_right, pad_col_left = util.pad_size_split(
            input.shape[2], filter.shape[2], strides[2])
        pad_row, pad_row_bottom, pad_row_top = util.pad_size_split(
            input.shape[1], filter.shape[1], strides[1])

        shape = (int(math.ceil(input.shape[0] / strides[0])),
                 util.pix_size(input.shape[1],
                               filter.shape[1], strides[1], 'SAME'),
                 util.pix_size(input.shape[2],
                               filter.shape[2], strides[2], 'SAME'),
                 int(math.ceil(filter.shape[0] / strides[3])))

    elif isinstance(padding, str) and padding == 'VALID':
        pad_col, pad_col_right, pad_col_left = 0, 0, 0
        pad_row, pad_row_bottom, pad_row_top = 0, 0, 0

        shape = (int(math.ceil(input.shape[0] / strides[0])),
                 util.pix_size(input.shape[1],
                               filter.shape[1], strides[1], 'VALID'),
                 util.pix_size(input.shape[2],
                               filter.shape[2], strides[2], 'VALID'),
                 int(math.ceil(filter.shape[0] / strides[3])))

    elif isinstance(padding, int):
        pad_col, pad_col_right, pad_col_left = padding * 2, padding, padding
        pad_row, pad_row_bottom, pad_row_top = padding * 2, padding, padding

        shape = (int(math.ceil(input.shape[0] / strides[0])),
                 util.pix_size(input.shape[1] + padding * 2,
                               filter.shape[1], strides[1], 'VALID'),
                 util.pix_size(input.shape[2] + padding * 2,
                               filter.shape[2], strides[2], 'VALID'),
                 int(math.ceil(filter.shape[0] / strides[3])))

    elif isinstance(padding, (tuple, list)):
        pad_col, pad_col_right, pad_col_left = padding[2] + padding[3], padding[3], padding[2]
        pad_row, pad_row_bottom, pad_row_top = padding[0] + padding[1], padding[1], padding[0]

        shape = (int(math.ceil(input.shape[0] / strides[0])),
                 util.pix_size(input.shape[1] + padding[0] + padding[1],
                               filter.shape[1], strides[1], 'VALID'),
                 util.pix_size(input.shape[2] + padding[2] + padding[3],
                               filter.shape[2], strides[2], 'VALID'),
                 int(math.ceil(filter.shape[0] / strides[3])))
    else:
        raise ValueError("padding options must be 'SAME', 'VALID', int, tuple, or list.")

    out = np.zeros(shape, dtype=np.int64)

    input = np.pad(input, [(0, 0),
                           (pad_row_top, pad_row_bottom),
                           (pad_col_left, pad_col_right),
                           (0, 0)], 'constant')

    if bias is None:
        bias = np.zeros([shape[-1]], dtype=np.int64)
    elif not isinstance(bias, np.ndarray):
        new_bias = np.zeros([shape[-1]], dtype=np.int64)
        for i in range(new_bias.shape[-1]):
            new_bias[i] = bias
        bias = new_bias
    elif len(bias.shape) == 1 and bias.shape[0] == 1:
        new_bias = np.zeros([shape[-1]], dtype=np.int64)
        for i in range(new_bias.shape[-1]):
            new_bias[i] = bias[0]
        bias = new_bias

    if scale is None:
        scale = np.ones([shape[-1]], dtype=np.int64)
    elif not isinstance(scale, np.ndarray):
        new_scale = np.zeros([shape[-1]], dtype=np.int64)
        for i in range(new_scale.shape[-1]):
            new_scale[i] = scale
        scale = new_scale
    elif len(scale.shape) == 1 and scale.shape[0] == 1:
        new_scale = np.zeros([shape[-1]], dtype=np.int64)
        for i in range(new_scale.shape[-1]):
            new_scale[i] = scale[0]
        scale = new_scale

    if rshift_mul is None:
        rshift_mul = np.zeros([shape[-1]], dtype=np.int64)
    elif not isinstance(rshift_mul, np.ndarray):
        new_rshift_mul = np.zeros([shape[-1]], dtype=np.int64)
        for i in range(new_rshift_mul.shape[-1]):
            new_rshift_mul[i] = rshift_mul
        rshift_mul = new_rshift_mul
    elif len(rshift_mul.shape) == 1 and rshift_mul.shape[0] == 1:
        new_rshift_mul = np.zeros([shape[-1]], dtype=np.int64)
        for i in range(new_rshift_mul.shape[-1]):
            new_rshift_mul[i] = rshift_mul[0]
        rshift_mul = new_rshift_mul

    if rshift_sum is None:
        rshift_sum = np.zeros([shape[-1]], dtype=np.int64)
    elif not isinstance(rshift_sum, np.ndarray):
        new_rshift_sum = np.zeros([shape[-1]], dtype=np.int64)
        for i in range(new_rshift_sum.shape[-1]):
            new_rshift_sum[i] = rshift_sum
        rshift_sum = new_rshift_sum
    elif len(rshift_sum.shape) == 1 and rshift_sum.shape[0] == 1:
        new_rshift_sum = np.zeros([shape[-1]], dtype=np.int64)
        for i in range(new_rshift_sum.shape[-1]):
            new_rshift_sum[i] = rshift_sum[0]
        rshift_sum = new_rshift_sum

    if rshift_out is None:
        rshift_out = np.zeros([shape[-1]], dtype=np.int64)
    elif not isinstance(rshift_out, np.ndarray):
        new_rshift_out = np.zeros([shape[-1]], dtype=np.int64)
        for i in range(new_rshift_out.shape[-1]):
            new_rshift_out[i] = rshift_out
        rshift_out = new_rshift_out
    elif len(rshift_out.shape) == 1 and rshift_out.shape[0] == 1:
        new_rshift_out = np.zeros([shape[-1]], dtype=np.int64)
        for i in range(new_rshift_out.shape[-1]):
            new_rshift_out[i] = rshift_out[0]
        rshift_out = new_rshift_out

    rshift_mul_pow = np.where(rshift_mul > np.zeros_like(rshift_mul, dtype=np.int64),
                              rshift_mul - 1,
                              np.zeros_like(rshift_mul))
    rshift_mul_round = np.where(rshift_mul > np.zeros_like(rshift_mul, dtype=np.int64),
                                np.power(np.ones_like(rshift_mul, dtype=np.int64) * 2,
                                         rshift_mul_pow),
                                np.zeros_like(rshift_mul, dtype=np.int64))

    rshift_sum_pow = np.where(rshift_sum > np.zeros_like(rshift_sum, dtype=np.int64),
                              rshift_sum - 1,
                              np.zeros_like(rshift_sum))
    rshift_sum_round = np.where(rshift_sum > np.zeros_like(rshift_sum, dtype=np.int64),
                                np.power(np.ones_like(rshift_sum, dtype=np.int64) * 2,
                                         rshift_sum_pow),
                                np.zeros_like(rshift_sum, dtype=np.int64))

    rshift_out_pow = np.where(rshift_out > np.zeros_like(rshift_out, dtype=np.int64),
                              rshift_out - 1,
                              np.zeros_like(rshift_out))
    rshift_out_round = np.where(rshift_out > np.zeros_like(rshift_out, dtype=np.int64),
                                np.power(np.ones_like(rshift_out, dtype=np.int64) * 2,
                                         rshift_out_pow),
                                np.zeros_like(rshift_out, dtype=np.int64))

    input_point = 0 if input_dtype is None else input_dtype.point
    filter_point = 0 if filter_dtype is None else filter_dtype.point
    bias_point = 0 if bias_dtype is None else bias_dtype.point
    scale_point = 0 if scale_dtype is None else scale_dtype.point
    out_point = (max(input_point, filter_point)
                 if dtype is None else dtype.point)
    out_width = 32 if dtype is None else dtype.width

    mul_point = max(input_point, filter_point)
    mul_shift = min(input_point, filter_point)
    sum_point = mul_point
    add_point = max(sum_point, bias_point)

    sum_shift = add_point - sum_point
    bias_shift = add_point - bias_point
    shifted_bias = np.left_shift(bias, bias_shift)

    scl_point = max(sum_point, scale_point)
    scl_shift = min(sum_point, scl_point)
    shifted_scale = np.right_shift(scale, scl_shift)

    p_th = (1 << (out_width - 1)) - 1
    if asymmetric_clip:
        n_th = -1 * p_th - 1
    else:
        n_th = -1 * p_th

    p_th = p_th >> out_point
    n_th = n_th >> out_point

    def my_matmul_by_matmul(a, w):
        return np.matmul(a, w.T)

    def my_matmul_by_multiply(a, w):
        mul = np.multiply(a, w)
        mul = np.add(mul, rshift_mul_round.reshape([rshift_mul_round.shape[-1], 1]))
        mul = np.right_shift(mul, mul_shift)
        mul = np.right_shift(mul, rshift_mul.reshape([rshift_mul.shape[-1], 1]))
        return np.add.reduce(mul, axis=1)

    if mul_shift == 0 and rshift_mul_round.all() == 0 and rshift_mul.all() == 0:
        my_matmul = my_matmul_by_matmul
    else:
        my_matmul = my_matmul_by_multiply

    if act_func is None:
        def act_op(x): return x
    else:
        act_op = act_func.get_act_func()

    for bat in range(shape[0]):
        w = filter.reshape([shape[3], -1])

        oy = 0
        for py in range(-pad_row_top, input.shape[1] + pad_row_bottom, strides[1]):

            ox = 0
            for px in range(-pad_col_left, input.shape[2] + pad_col_right, strides[2]):

                ys = py + pad_row_top
                ye = ys + filter.shape[1]
                xs = px + pad_col_left
                xe = xs + filter.shape[2]
                a = input[bat, ys: ye, xs: xe].reshape([-1])

                sum = my_matmul(a, w)

                sum = np.left_shift(sum, sum_shift)
                sum = np.add(sum, rshift_sum_round)
                sum = np.right_shift(sum, rshift_sum)
                sum = np.add(sum, shifted_bias)
                sum = np.multiply(sum, shifted_scale)
                frac = np.where(rshift_out != 0, np.where(sum >= 0, rshift_out_round, rshift_out_round - 1),
                                np.zeros_like(rshift_out, dtype=np.int64))
                sum = np.add(sum, frac)
                sum = np.right_shift(sum, rshift_out)
                sum = np.where(sum > p_th, p_th, np.where(sum < n_th, n_th, sum))

                out[bat][oy][ox][:] = act_op(sum)

                ox += 1
                if ox >= out.shape[2]:
                    break

            oy += 1
            if oy >= out.shape[1]:
                break

    return out
