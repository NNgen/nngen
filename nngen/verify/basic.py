from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.util as util


def add(x, y, dtype=None, name=None, par=1,
        x_dtype=None, y_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    y_point = 0 if y_dtype is None else y_dtype.point
    xy_point = max(x_point, y_point)
    out_point = xy_point if dtype is None else dtype.point
    out_shift = out_point - xy_point

    x = x << (out_point - x_point)
    y = y << (out_point - y_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(x + y)

    return ret


def sub(x, y, dtype=None, name=None, par=1,
        x_dtype=None, y_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    y_point = 0 if y_dtype is None else y_dtype.point
    xy_point = max(x_point, y_point)
    out_point = xy_point if dtype is None else dtype.point
    out_shift = out_point - xy_point

    x = x << (out_point - x_point)
    y = y << (out_point - y_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(x - y)

    return ret


def neg(x, dtype=None, name=None, par=1,
        x_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    out_point = x_point if dtype is None else dtype.point
    out_shift = out_point - x_point

    x = x << (out_point - x_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(np.negative(x))

    return ret


def abs(x, dtype=None, name=None, par=1,
        x_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    out_point = x_point if dtype is None else dtype.point
    out_shift = out_point - x_point

    x = x << (out_point - x_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(np.abs(x))

    return ret


def zeros_imm(shape, dtype=None, name=None, par=1):

    out_point = x_point if dtype is None else dtype.point
    out_shift = out_point

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(np.zeros(shape, dtype=np.int64))

    return ret


def zeros_imm_like(x, dtype=None, name=None, par=1):
    shape = x.shape
    return zeros_imm(shape, dtype, name, par)


def ones_imm(shape, dtype=None, name=None, par=1):

    out_point = x_point if dtype is None else dtype.point
    out_shift = out_point

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(np.ones(shape, dtype=np.int64))

    return ret


def ones_imm_like(x, dtype=None, name=None, par=1):
    shape = x.shape
    return ones_imm(shape, dtype, name, par)


def full_imm(shape, fill_value, dtype=None, name=None, par=1):

    out_point = x_point if dtype is None else dtype.point
    out_shift = out_point

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(np.full(shape, fill_value, dtype=np.int64))

    return ret


def full_imm_like(x, fill_value, dtype=None, name=None, par=1):
    shape = x.shape
    return full_imm(shape, fill_value, dtype, name, par)


def equal(x, y, dtype=None, name=None, par=1,
          x_dtype=None, y_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    y_point = 0 if y_dtype is None else y_dtype.point
    xy_point = max(x_point, y_point)
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - xy_point

    x = x << (out_point - x_point)
    y = y << (out_point - y_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(x == y)

    return ret


def not_equal(x, y, dtype=None, name=None, par=1,
              x_dtype=None, y_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    y_point = 0 if y_dtype is None else y_dtype.point
    xy_point = max(x_point, y_point)
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - xy_point

    x = x << (out_point - x_point)
    y = y << (out_point - y_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(x != y)

    return ret


def less(x, y, dtype=None, name=None, par=1,
         x_dtype=None, y_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    y_point = 0 if y_dtype is None else y_dtype.point
    xy_point = max(x_point, y_point)
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - xy_point

    x = x << (out_point - x_point)
    y = y << (out_point - y_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(x < y)

    return ret


def less_equal(x, y, dtype=None, name=None, par=1,
               x_dtype=None, y_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    y_point = 0 if y_dtype is None else y_dtype.point
    xy_point = max(x_point, y_point)
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - xy_point

    x = x << (out_point - x_point)
    y = y << (out_point - y_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(x <= y)

    return ret


def greater(x, y, dtype=None, name=None, par=1,
            x_dtype=None, y_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    y_point = 0 if y_dtype is None else y_dtype.point
    xy_point = max(x_point, y_point)
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - xy_point

    x = x << (out_point - x_point)
    y = y << (out_point - y_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(x > y)

    return ret


def greater_equal(x, y, dtype=None, name=None, par=1,
                  x_dtype=None, y_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    y_point = 0 if y_dtype is None else y_dtype.point
    xy_point = max(x_point, y_point)
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - xy_point

    x = x << (out_point - x_point)
    y = y << (out_point - y_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(x >= y)

    return ret


def sign_binary(x, dtype=None, name=None, par=1,
                x_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    out_point = x_point if dtype is None else dtype.point
    out_shift = out_point - x_point

    x = x << (out_point - x_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(np.where(x > np.zeros_like(x, dtype=np.int64),
                          np.ones_like(x, dtype=np.int64),
                          np.negative(np.ones_like(x, dtype=np.int64))))

    return ret


def sign_ternary(x, dtype=None, name=None, par=1,
                 x_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    out_point = x_point if dtype is None else dtype.point
    out_shift = out_point - x_point

    x = x << (out_point - x_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(np.where(x > np.zeros_like(x, dtype=np.int64),
                          np.ones_like(x, dtype=np.int64),
                          np.where(x == np.zeros_like(x, dtype=np.int64),
                                   np.zeros_like(x, dtype=np.int64),
                                   np.negative(np.ones_like(x, dtype=np.int64)))))

    return ret


def where(condition, x, y, dtype=None, name=None, par=1,
          condition_dtype=None, x_dtype=None, y_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    y_point = 0 if y_dtype is None else y_dtype.point
    xy_point = max(x_point, y_point)
    out_point = xy_point if dtype is None else dtype.point
    out_shift = out_point - xy_point

    x = x << (out_point - x_point)
    y = y << (out_point - y_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(np.where(condition, x, y))

    return ret


def add_n(arg, dtype=None, name=None, par=1,
          arg_dtypes=None):

    if not isinstance(arg, (tuple, list)):
        raise TypeError('expected tuple or list')

    if arg_dtypes is None:
        arg_dtypes = [None for _ in arg]

    if not isinstance(arg_dtypes, (tuple, list)):
        raise TypeError('expected tuple or list')

    if len(arg) != len(arg_dtypes):
        raise ValueError('length mismatch: %d != %d' % (len(arg), len(arg_dtypes)))

    arg_points = [0 if a_dtype is None else a_dtype.point
                  for a_dtype in arg_dtypes]
    max_arg_point = max(*arg_points)

    out_point = max_arg_point if dtype is None else dtype.point
    out_shift = out_point - max_arg_point

    values = [a << (out_point - a_point)
              for a, a_point in zip(arg, arg_points)]

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = values[0]
    for value in values[1:]:
        ret += value

    ret = out_op(ret)

    return ret


def lshift(x, y, dtype=None, name=None, par=1,
           x_dtype=None, y_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    y_point = 0 if y_dtype is None else y_dtype.point
    xy_point = max(x_point, y_point)
    xy_shift = min(x_point, y_point)
    out_point = xy_point if dtype is None else dtype.point
    out_shift = -xy_shift if dtype is None else dtype.point - xy_point - xy_shift

    x = x << (out_point - x_point)
    y = y << (out_point - y_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(x << y)

    return ret


def rshift(x, y, dtype=None, name=None, par=1,
           x_dtype=None, y_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    y_point = 0 if y_dtype is None else y_dtype.point
    xy_point = max(x_point, y_point)
    xy_shift = min(x_point, y_point)
    out_point = xy_point if dtype is None else dtype.point
    out_shift = -xy_shift if dtype is None else dtype.point - xy_point - xy_shift

    x = x << (out_point - x_point)
    y = y << (out_point - y_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(x >> y)

    return ret


def rshift_round(x, y, dtype=None, name=None, par=1,
                 x_dtype=None, y_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    y_point = 0 if y_dtype is None else y_dtype.point
    xy_point = max(x_point, y_point)
    xy_shift = min(x_point, y_point)
    out_point = xy_point if dtype is None else dtype.point
    out_shift = -xy_shift if dtype is None else dtype.point - xy_point - xy_shift

    x = x << (out_point - x_point)
    y = y << (out_point - y_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    shifted = x >> y
    last_bit = (x >> (y - 1)) & 0x1

    ret = out_op(np.where(y == 0, x, shifted + last_bit))

    return ret


def clip(x, asymmetric_clip=False,
         dtype=None, name=None, par=1,
         x_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - x_point

    width = 32 if dtype is None else dtype.width

    if dtype.signed:
        p_th = (1 << (width - 1)) - 1
        if asymmetric_clip:
            n_th = -1 * p_th - 1
        else:
            n_th = -1 * p_th
    else:
        p_th = (1 << width) - 1
        n_th = 0

    p_th = np.ones_like(x, dtype=np.int64) * [p_th]
    n_th = np.ones_like(x, dtype=np.int64) * [n_th]

    p = np.where(x > p_th, p_th, x)
    n = np.where(x < n_th, n_th, x)
    x = np.where(x >= 0, p, n)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(x)

    return ret


def multiply(x, y, dtype=None, name=None, par=1,
             x_dtype=None, y_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    y_point = 0 if y_dtype is None else y_dtype.point
    xy_point = max(x_point, y_point)
    xy_shift = min(x_point, y_point)
    out_point = xy_point if dtype is None else dtype.point
    out_shift = -xy_shift if dtype is None else dtype.point - xy_point - xy_shift

    x = x << (out_point - x_point)
    y = y << (out_point - y_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(x * y)

    return ret


def multiply_shared(x, y, dtype=None, name=None, par=1,
                    x_dtype=None, y_dtype=None):

    return multiply(x, y, dtype, name, par, x_dtype, y_dtype)


def div(x, y, dtype=None, name=None, par=1,
        x_dtype=None, y_dtype=None):

    x_point = 0 if x_dtype is None else x_dtype.point
    y_point = 0 if y_dtype is None else y_dtype.point
    xy_point = max(x_point, y_point)
    xy_shift = min(x_point, y_point)
    out_point = xy_point if dtype is None else dtype.point
    out_shift = -xy_shift if dtype is None else dtype.point - xy_point - xy_shift

    x = x << (out_point - x_point)
    y = y << (out_point - y_point)

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(x // y)

    return ret


def multiply_add_rshift_clip(x, y, z, shamt, asymmetric_clip=False,
                             dtype=None, sum_dtype=None, name=None, par=1,
                             x_dtype=None, y_dtype=None, z_dtype=None, shamt_dtype=None):

    v = multiply(x, y, dtype=sum_dtype, par=par,
                 x_dtype=x_dtype, y_dtype=y_dtype)
    v = add(v, z, dtype=sum_dtype, par=par,
            x_dtype=sum_dtype, y_dtype=z_dtype)
    v = rshift(v, shamt, dtype=sum_dtype, par=par,
               x_dtype=sum_dtype, y_dtype=shamt_dtype)
    return clip(v, asymmetric_clip=asymmetric_clip,
                dtype=dtype, par=par,
                x_dtype=sum_dtype)


def reduce_sum(input_tensor,
               axis=None, keep_dims=False, dtype=None, name=None, par=1,
               input_tensor_dtype=None):

    input_tensor_point = 0 if input_tensor_dtype is None else input_tensor_dtype.point
    out_point = input_tensor_point if dtype is None else dtype.point
    out_shift = out_point - input_tensor_point

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    if axis is None:
        input_tensor = input_tensor.reshape([-1])

    ret = np.sum(input_tensor, axis)

    if not isinstance(ret, np.ndarray):
        ret = np.array([ret])

    ret = out_op(ret)

    return ret


def reduce_max(input_tensor,
               axis=None, keep_dims=False, dtype=None, name=None, par=1,
               input_tensor_dtype=None):

    input_tensor_point = 0 if input_tensor_dtype is None else input_tensor_dtype.point
    out_point = input_tensor_point if dtype is None else dtype.point
    out_shift = out_point - input_tensor_point

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    if axis is None:
        input_tensor = input_tensor.reshape([-1])

    ret = np.max(input_tensor, axis=axis)

    if not isinstance(ret, np.ndarray):
        ret = np.array([ret])

    ret = out_op(ret)

    return ret


def reduce_min(input_tensor,
               axis=None, keep_dims=False, dtype=None, name=None, par=1,
               input_tensor_dtype=None):

    input_tensor_point = 0 if input_tensor_dtype is None else input_tensor_dtype.point
    out_point = input_tensor_point if dtype is None else dtype.point
    out_shift = out_point - input_tensor_point

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    if axis is None:
        input_tensor = input_tensor.reshape([-1])

    ret = np.min(input_tensor, axis=axis)

    if not isinstance(ret, np.ndarray):
        ret = np.array([ret])

    ret = out_op(ret)

    return ret


def argmax(input_tensor,
           axis=None, keep_dims=False, dtype=None, name=None, par=1,
           input_tensor_dtype=None):

    input_tensor_point = 0 if input_tensor_dtype is None else input_tensor_dtype.point
    out_point = input_tensor_point if dtype is None else dtype.point
    out_shift = out_point - input_tensor_point

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    if isinstance(axis, (tuple, list, np.ndarray)) and len(axis) > 1:
        raise ValueError('Size of axis must be 1.')

    if isinstance(axis, (tuple, list, np.ndarray)):
        axis = int(axis[0])

    if axis is None:
        input_tensor = input_tensor.reshape([-1])

    ret = np.argmax(input_tensor, axis=axis)

    if not isinstance(ret, np.ndarray):
        ret = np.array([ret])

    ret = out_op(ret)

    return ret


def argmin(input_tensor,
           axis=None, keep_dims=False, dtype=None, name=None, par=1,
           input_tensor_dtype=None):

    input_tensor_point = 0 if input_tensor_dtype is None else input_tensor_dtype.point
    out_point = input_tensor_point if dtype is None else dtype.point
    out_shift = out_point - input_tensor_point

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    if isinstance(axis, (tuple, list, np.ndarray)) and len(axis) > 1:
        raise ValueError('Size of axis must be 1.')

    if isinstance(axis, (tuple, list, np.ndarray)):
        axis = int(axis[0])

    if axis is None:
        input_tensor = input_tensor.reshape([-1])

    ret = np.argmin(input_tensor, axis=axis)

    if not isinstance(ret, np.ndarray):
        ret = np.array([ret])

    ret = out_op(ret)

    return ret


def _reshape(tensor, shape, dtype=None, name=None, tensor_dtype=None):
    return np.reshape(tensor, shape)


def _lazy_reshape(tensor, shape, dtype=None, name=None, tensor_dtype=None):
    return np.reshape(tensor, shape)


def _View(tensor, shape, dtype=None, name=None):
    return np.reshape(tensor, shape)


def reshape(tensor, shape, dtype=None, name=None, tensor_dtype=None):
    return np.reshape(tensor, shape)


def cast(x, dtype=None, name=None):
    return x


def expand_dims(input, axis, name=None):
    shape = input.shape
    rank = len(shape)
    axis = util.to_axis(axis, rank)[0]
    new_shape = list(shape[:axis]) + [1] + list(shape[axis:])

    return input.reshape(new_shape)


def transpose(a, perm=None, dtype=None, name=None, a_dtype=None):
    return np.transpose(a, perm)
