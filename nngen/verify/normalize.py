from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from . import basic
from . import concat


def normalize(x, y, z, shamt,
              dtype=None, sum_dtype=None, name=None, par=1,
              x_dtype=None, y_dtype=None, z_dtype=None, shamt_dtype=None):

    return basic.multiply_add_rshift_clip(x, y, z, shamt,
                                          dtype, sum_dtype, name, par,
                                          x_dtype, y_dtype, z_dtype, shamt_dtype)


def scaled_add(a, b, a_scale, b_scale, shamt,
               dtype=None, sum_dtype=None, name=None, par=1,
               a_dtype=None, b_dtype=None):

    v0 = basic.multiply(a, a_scale, dtype=sum_dtype, par=par,
                        x_dtype=a_dtype)
    v1 = basic.multiply(b, b_scale, dtype=sum_dtype, par=par,
                        x_dtype=b_dtype)
    v = basic.add(v0, v1, dtype=sum_dtype, par=par,
                  x_dtype=sum_dtype, y_dtype=sum_dtype)
    v = basic.rshift(v, shamt, dtype=sum_dtype, par=par,
                     x_dtype=sum_dtype)
    return basic.clip(v, dtype=dtype, par=par,
                      x_dtype=sum_dtype)


def scaled_concat(values, scales, shamt, axis,
                  dtype=None, mul_dtype=None, name=None):

    scaled_values = []
    for value, scale in zip(values, scales):
        v = basic.multiply(value, scale, dtype=mul_dtype)
        v = basic.rshift(v, shamt, dtype=mul_dtype,
                         x_dtype=mul_dtype)
        v = basic.clip(v, dtype=dtype,
                       x_dtype=mul_dtype)
        scaled_values.append(v)

    return concat(scaled_values, axis, dtype, name)


def scaled_multiply(a, b, shamt,
                    dtype=None, mul_dtype=None, name=None, par=1,
                    a_dtype=None, b_dtype=None):

    v = basic.multiply(a, b, dtype=mul_dtype, par=par,
                       x_dtype=a_dtype, y_dtype=b_dtype)
    v = basic.rshift(v, shamt, dtype=mul_dtype, par=par,
                     x_dtype=mul_dtype)
    return basic.clip(v, dtype=dtype, par=par,
                      x_dtype=mul_dtype)


def scaled_div(a, b, shamt,
               dtype=None, div_dtype=None, name=None, par=1,
               a_dtype=None, b_dtype=None):

    a = basic.lshift(a, shamt, dtype=div_dtype, par=par,
                     x_dtype=a_dtype)
    v = basic.div(a, b, dtype=div_dtype, par=par,
                  x_dtype=a_dtype, y_dtype=b_dtype)
    return basic.clip(v, dtype=dtype, par=par,
                      x_dtype=div_dtype)
