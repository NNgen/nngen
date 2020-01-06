from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.util as util


def matmul(a, b,
           bias=None, scale=None,
           transposed_a=False, transposed_b=True,
           rshift_mul=None, rshift_sum=None, rshift_out=None,
           act_func=None,
           dtype=None, mul_dtype=None, sum_dtype=None,
           name=None,
           par_left_col=1, par_left_row=1, par_out_col=1,
           concur_out_col=None, stationary='right',
           left_ram_size=None, right_ram_size=None,
           bias_ram_size=None, scale_ram_size=None,
           vshamt_mul_ram_size=None, vshamt_sum_ram_size=None, vshamt_out_ram_size=None,
           out_ram_size=None,
           disable_keep_left=False,
           a_dtype=None, b_dtype=None,
           bias_dtype=None, scale_dtype=None):

    if transposed_a:
        a = a.transpose()

    if not transposed_b:
        b = b.transpose()

    if a.shape[1] != b.shape[1]:
        raise ValueError("shape mismatch: %s != %s" %
                         str(a.shape), str(b.shape))

    c_shape = (a.shape[0], b.shape[0])

    if bias is None:
        bias = np.zeros([c_shape[-1]], dtype=np.int64)
    elif not isinstance(bias, np.ndarray):
        new_bias = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_bias.shape[-1]):
            new_bias[i] = bias
        bias = new_bias
    elif len(bias.shape) == 1 and bias.shape[0] == 1:
        new_bias = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_bias.shape[-1]):
            new_bias[i] = bias[0]
        bias = new_bias

    if scale is None:
        scale = np.ones([c_shape[-1]], dtype=np.int64)
    elif not isinstance(scale, np.ndarray):
        new_scale = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_scale.shape[-1]):
            new_scale[i] = scale
        scale = new_scale
    elif len(scale.shape) == 1 and scale.shape[0] == 1:
        new_scale = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_scale.shape[-1]):
            new_scale[i] = scale[0]
        scale = new_scale

    if rshift_mul is None:
        rshift_mul = np.zeros([c_shape[-1]], dtype=np.int64)
    elif not isinstance(rshift_mul, np.ndarray):
        new_rshift_mul = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_rshift_mul.shape[-1]):
            new_rshift_mul[i] = rshift_mul
        rshift_mul = new_rshift_mul
    elif len(rshift_mul.shape) == 1 and rshift_mul.shape[0] == 1:
        new_rshift_mul = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_rshift_mul.shape[-1]):
            new_rshift_mul[i] = rshift_mul[0]
        rshift_mul = new_rshift_mul

    if rshift_sum is None:
        rshift_sum = np.zeros([c_shape[-1]], dtype=np.int64)
    elif not isinstance(rshift_sum, np.ndarray):
        new_rshift_sum = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_rshift_sum.shape[-1]):
            new_rshift_sum[i] = rshift_sum
        rshift_sum = new_rshift_sum
    elif len(rshift_sum.shape) == 1 and rshift_sum.shape[0] == 1:
        new_rshift_sum = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_rshift_sum.shape[-1]):
            new_rshift_sum[i] = rshift_sum[0]
        rshift_sum = new_rshift_sum

    if rshift_out is None:
        rshift_out = np.zeros([c_shape[-1]], dtype=np.int64)
    elif not isinstance(rshift_out, np.ndarray):
        new_rshift_out = np.zeros([c_shape[-1]], dtype=np.int64)
        for i in range(new_rshift_out.shape[-1]):
            new_rshift_out[i] = rshift_out
        rshift_out = new_rshift_out
    elif len(rshift_out.shape) == 1 and rshift_out.shape[0] == 1:
        new_rshift_out = np.zeros([c_shape[-1]], dtype=np.int64)
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

    a_point = 0 if a_dtype is None else a_dtype.point
    b_point = 0 if b_dtype is None else b_dtype.point
    bias_point = 0 if bias_dtype is None else bias_dtype.point
    scale_point = 0 if scale_dtype is None else scale_dtype.point
    c_point = max(a_point, b_point) if dtype is None else dtype.point
    c_width = 32 if dtype is None else dtype.width

    mul_point = max(a_point, b_point)
    mul_shift = min(a_point, b_point)
    sum_point = mul_point
    add_point = max(sum_point, bias_point)

    sum_shift = add_point - sum_point
    bias_shift = add_point - bias_point
    shifted_bias = np.left_shift(bias, bias_shift)

    scl_point = max(sum_point, scale_point)
    scl_shift = min(sum_point, scl_point)
    shifted_scale = np.right_shift(scale, scl_shift)

    p_th = (1 << (c_width - 1)) - 1
    n_th = -1 * p_th
    p_th = p_th >> c_point
    n_th = n_th >> c_point

    def my_matmul_by_matmul(a, w):
        return np.matmul(a, w.T)

    def my_matmul_by_multiply(a, w):
        v = a.reshape([a.shape[0], 1, a.shape[1]])
        mul = np.multiply(v, w)
        mul = np.right_shift(mul, mul_shift)
        mul = np.add(mul, rshift_mul_round.reshape([rshift_mul_round.shape[-1], 1]))
        mul = np.right_shift(mul, rshift_mul.reshape([rshift_mul.shape[-1], 1]))
        return np.add.reduce(mul, axis=2)

    if mul_shift == 0 and rshift_mul_round.all() == 0 and rshift_mul.all() == 0:
        my_matmul = my_matmul_by_matmul
    else:
        my_matmul = my_matmul_by_multiply

    if act_func is None:
        def act_op(x): return x
    else:
        act_op = act_func.get_act_func()

    sum = my_matmul(a, b)

    sum = np.left_shift(sum, sum_shift)
    sum = np.add(sum, rshift_sum_round)
    sum = np.right_shift(sum, rshift_sum)
    sum = np.add(sum, shifted_bias)
    sum = np.multiply(sum, shifted_scale)
    sum = np.right_shift(sum, rshift_out)
    sum = np.where(sum > p_th, p_th, np.where(sum < n_th, n_th, sum))

    c = act_op(sum)

    return c
