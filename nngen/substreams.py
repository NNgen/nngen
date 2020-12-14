from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math

import veriloggen as vg
import veriloggen.thread as vthread
import veriloggen.stream as vstream


_tmp_counter = 0


def _tmp_name(prefix=None):
    global _tmp_counter
    if prefix is None:
        prefix = 'tmp'
    name = '%s_%d' % (prefix, _tmp_counter)
    _tmp_counter += 1
    return name


def mul(m, clk, rst,
        x_datawidth, x_point, x_signed,
        y_datawidth, y_point, y_signed,
        mul_width=None, mul_point=None, mul_signed=None):

    name = _tmp_name('mul')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)

    z = x * y
    z.latency = 4
    if mul_width is not None:
        z.width = mul_width
    if mul_signed is not None:
        z.signed = mul_signed
    if mul_point is not None and point != mul_point:
        z = stream.Cast(z, point=mul_point)

    stream.sink(z, 'z')
    return stream


def mul_rshift(m, clk, rst,
               x_datawidth, x_point, x_signed,
               y_datawidth, y_point, y_signed,
               mul_width=None, mul_point=None, mul_signed=None):

    name = _tmp_name('mul')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1

    z = x * y
    z.latency = 4
    if mul_width is not None:
        z.width = mul_width
    if mul_signed is not None:
        z.signed = mul_signed
    if mul_point is not None and point != mul_point:
        z = stream.Cast(z, point=mul_point)
    z = stream.Sra(z, rshift)

    stream.sink(z, 'z')
    return stream


def mul_rshift_clip(m, clk, rst,
                    x_datawidth, x_point, x_signed,
                    y_datawidth, y_point, y_signed,
                    mul_width=None, mul_point=None, mul_signed=None,
                    out_width=None, out_point=None, out_signed=None):

    name = _tmp_name('mul_rshift_clip')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1

    z = x * y
    z.latency = 4
    if mul_width is not None:
        z.width = mul_width
    if mul_signed is not None:
        z.signed = mul_signed
    if mul_point is not None and point != mul_point:
        z = stream.Cast(z, point=mul_point)
    z = stream.Sra(z, rshift)

    p_th = (1 << (out_width - 1)) - 1
    n_th = -1 * p_th
    p_th = p_th >> out_point
    n_th = n_th >> out_point

    p = stream.Mux(z > p_th, p_th, z)
    n = stream.Mux(z < n_th, n_th, z)
    z = stream.Mux(z >= 0, p, n)

    if out_width is not None:
        z.width = out_width
    if out_signed is not None:
        z.signed = out_signed
    if out_point is not None and z.point != out_point:
        z = stream.Cast(z, point=out_point)

    stream.sink(z, 'z')
    return stream


def mul_rshift_round(m, clk, rst,
                     x_datawidth, x_point, x_signed,
                     y_datawidth, y_point, y_signed,
                     mul_width=None, mul_point=None, mul_signed=None):

    name = _tmp_name('mul')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1

    z = x * y
    z.latency = 4
    if mul_width is not None:
        z.width = mul_width
    if mul_signed is not None:
        z.signed = mul_signed
    if mul_point is not None and point != mul_point:
        z = stream.Cast(z, point=mul_point)
    z = stream.SraRound(z, rshift)

    stream.sink(z, 'z')
    return stream


def mul_rshift_round_clip(m, clk, rst,
                          x_datawidth, x_point, x_signed,
                          y_datawidth, y_point, y_signed,
                          mul_width=None, mul_point=None, mul_signed=None,
                          out_width=None, out_point=None, out_signed=None):

    name = _tmp_name('mul_rshift_round_clip')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1

    z = x * y
    z.latency = 4
    if mul_width is not None:
        z.width = mul_width
    if mul_signed is not None:
        z.signed = mul_signed
    if mul_point is not None and point != mul_point:
        z = stream.Cast(z, point=mul_point)
    z = stream.SraRound(z, rshift)

    p_th = (1 << (out_width - 1)) - 1
    n_th = -1 * p_th
    p_th = p_th >> out_point
    n_th = n_th >> out_point

    p = stream.Mux(z > p_th, p_th, z)
    n = stream.Mux(z < n_th, n_th, z)
    z = stream.Mux(z >= 0, p, n)

    if out_width is not None:
        z.width = out_width
    if out_signed is not None:
        z.signed = out_signed
    if out_point is not None and z.point != out_point:
        z = stream.Cast(z, point=out_point)

    stream.sink(z, 'z')
    return stream


def mul_rshift_round_madd(m, clk, rst,
                          x_datawidth, x_point, x_signed,
                          y_datawidth, y_point, y_signed,
                          mul_width=None, mul_point=None, mul_signed=None):

    name = _tmp_name('mul')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1

    frac = stream.Mux(rshift > 0, stream.Sll(1, rshift - 1), 0)
    frac.width = mul_width
    neg_frac = stream.Uminus(frac)
    neg_frac.width = datawidth
    neg_frac.latency = 0
    frac = stream.Mux(x >= 0, frac, neg_frac)
    frac.latency = 0
    frac.width = datawidth

    z = stream.Madd(x, y, frac)
    z.latency = 4
    if mul_width is not None:
        z.width = mul_width
    if mul_signed is not None:
        z.signed = mul_signed
    if mul_point is not None and point != mul_point:
        z = stream.Cast(z, point=mul_point)
    z = stream.Sra(z, rshift)

    stream.sink(z, 'z')
    return stream


def madd(m, clk, rst,
         x_datawidth, x_point, x_signed,
         y_datawidth, y_point, y_signed,
         z_datawidth, z_point, z_signed,
         sum_width=None, sum_point=None, sum_signed=None):

    name = _tmp_name('madd')
    datawidth = max(x_datawidth, y_datawidth, z_datawidth)
    point = max(x_point, y_point, z_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)
    z = stream.source('z', z_datawidth, z_point, z_signed)

    sum = stream.Madd(x, y, z)
    sum.latency = 4
    if mul_width is not None:
        sum.width = mul_width
    if mul_signed is not None:
        sum.signed = mul_signed
    if mul_point is not None and point != mul_point:
        sum = stream.Cast(sum, point=mul_point)

    stream.sink(sum, 'sum')
    return stream


def madd_rshift(m, clk, rst,
                x_datawidth, x_point, x_signed,
                y_datawidth, y_point, y_signed,
                z_datawidth, z_point, z_signed,
                sum_width=None, sum_point=None, sum_signed=None):

    name = _tmp_name('madd')
    datawidth = max(x_datawidth, y_datawidth, z_datawidth)
    point = max(x_point, y_point, z_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)
    z = stream.source('z', z_datawidth, z_point, z_signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1

    sum = stream.Madd(x, y, z)
    sum.latency = 4
    if mul_width is not None:
        sum.width = mul_width
    if mul_signed is not None:
        sum.signed = mul_signed
    if mul_point is not None and point != mul_point:
        sum = stream.Cast(sum, point=mul_point)
    sum = stream.Sra(sum, rshift)

    stream.sink(sum, 'sum')
    return stream


def madd_rshift_clip(m, clk, rst,
                     x_datawidth, x_point, x_signed,
                     y_datawidth, y_point, y_signed,
                     z_datawidth, z_point, z_signed,
                     sum_width=None, sum_point=None, sum_signed=None,
                     out_width=None, out_point=None, out_signed=None):

    name = _tmp_name('madd')
    datawidth = max(x_datawidth, y_datawidth, z_datawidth)
    point = max(x_point, y_point, z_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)
    z = stream.source('z', z_datawidth, z_point, z_signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1

    sum = stream.Madd(x, y, z)
    sum.latency = 4
    if mul_width is not None:
        sum.width = mul_width
    if mul_signed is not None:
        sum.signed = mul_signed
    if mul_point is not None and point != mul_point:
        sum = stream.Cast(sum, point=mul_point)
    sum = stream.Sra(sum, rshift)

    p_th = (1 << (out_width - 1)) - 1
    n_th = -1 * p_th
    p_th = p_th >> out_point
    n_th = n_th >> out_point

    p = stream.Mux(sum > p_th, p_th, sum)
    n = stream.Mux(sum < n_th, n_th, sum)
    sum = stream.Mux(sum >= 0, p, n)

    if out_width is not None:
        sum.width = out_width
    if out_signed is not None:
        sum.signed = out_signed
    if out_point is not None and sum.point != out_point:
        sum = stream.Cast(sum, point=out_point)

    stream.sink(sum, 'sum')
    return stream


def mac(m, clk, rst,
        x_datawidth, x_point, x_signed,
        y_datawidth, y_point, y_signed,
        mul_width=None, mul_point=None, mul_signed=None,
        sum_width=None, sum_point=None, sum_signed=None):

    name = _tmp_name('mac')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)
    size = stream.constant('size', signed=False)

    z = x * y
    z.latency = 4
    if mul_width is not None:
        z.width = mul_width
    if mul_signed is not None:
        z.signed = mul_signed
    if mul_point is not None and point != mul_point:
        z = stream.Cast(z, point=mul_point)
    sum, v = stream.ReduceAddValid(z, size, width=sum_width, signed=sum_signed)
    if sum_point is not None and point != sum_point:
        sum = stream.Cast(sum, point=sum_point)

    stream.sink(sum, 'sum')
    stream.sink(v, 'valid')
    return stream


def mac_rshift(m, clk, rst,
               x_datawidth, x_point, x_signed,
               y_datawidth, y_point, y_signed,
               mul_width=None, mul_point=None, mul_signed=None,
               sum_width=None, sum_point=None, sum_signed=None):

    name = _tmp_name('mac')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1
    size = stream.constant('size', signed=False)

    z = x * y
    z.latency = 4
    if mul_width is not None:
        z.width = mul_width
    if mul_signed is not None:
        z.signed = mul_signed
    if mul_point is not None and point != mul_point:
        z = stream.Cast(z, point=mul_point)
    z = stream.Sra(z, rshift)
    sum, v = stream.ReduceAddValid(z, size, width=sum_width, signed=sum_signed)
    if sum_point is not None and point != sum_point:
        sum = stream.Cast(sum, point=sum_point)

    stream.sink(sum, 'sum')
    stream.sink(v, 'valid')
    return stream


def mac_rshift_round(m, clk, rst,
                     x_datawidth, x_point, x_signed,
                     y_datawidth, y_point, y_signed,
                     mul_width=None, mul_point=None, mul_signed=None,
                     sum_width=None, sum_point=None, sum_signed=None):

    name = _tmp_name('mac')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1
    size = stream.constant('size', signed=False)

    z = x * y
    z.latency = 4
    if mul_width is not None:
        z.width = mul_width
    if mul_signed is not None:
        z.signed = mul_signed
    if mul_point is not None and point != mul_point:
        z = stream.Cast(z, point=mul_point)
    z = stream.SraRound(z, rshift)
    sum, v = stream.ReduceAddValid(z, size, width=sum_width, signed=sum_signed)
    if sum_point is not None and point != sum_point:
        sum = stream.Cast(sum, point=sum_point)

    stream.sink(sum, 'sum')
    stream.sink(v, 'valid')
    return stream


def mac_rshift_round_madd(m, clk, rst,
                          x_datawidth, x_point, x_signed,
                          y_datawidth, y_point, y_signed,
                          mul_width=None, mul_point=None, mul_signed=None,
                          sum_width=None, sum_point=None, sum_signed=None):

    name = _tmp_name('mac')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1
    size = stream.constant('size', signed=False)

    frac = stream.Mux(rshift > 0, stream.Sll(1, rshift - 1), 0)
    frac.width = mul_width

    z = stream.Madd(x, y, frac)
    z.latency = 4
    if mul_width is not None:
        z.width = mul_width
    if mul_signed is not None:
        z.signed = mul_signed
    if mul_point is not None and point != mul_point:
        z = stream.Cast(z, point=mul_point)
    z = stream.Sra(z, rshift)
    sum, v = stream.ReduceAddValid(z, size, width=sum_width, signed=sum_signed)
    if sum_point is not None and point != sum_point:
        sum = stream.Cast(sum, point=sum_point)

    stream.sink(sum, 'sum')
    stream.sink(v, 'valid')
    return stream


def acc(m, clk, rst,
        datawidth, point, signed,
        sum_width=None, sum_point=None, sum_signed=None):

    name = _tmp_name('acc')

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', datawidth, point, signed)
    size = stream.constant('size', signed=False)

    sum, v = stream.ReduceAddValid(x, size, width=sum_width, signed=sum_signed)
    if sum_point is not None and point != sum_point:
        sum = stream.Cast(sum, point=sum_point)

    stream.sink(sum, 'sum')
    stream.sink(v, 'valid')
    return stream


def acc_rshift(m, clk, rst,
               datawidth, point, signed,
               sum_width=None, sum_point=None, sum_signed=None):

    name = _tmp_name('acc')

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', datawidth, point, signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1
    size = stream.constant('size', signed=False)

    sum, v = stream.ReduceAddValid(x, size, width=sum_width, signed=sum_signed)
    if sum_point is not None and point != sum_point:
        sum = stream.Cast(sum, point=sum_point)

    sum = stream.Sra(sum, rshift)

    stream.sink(sum, 'sum')
    stream.sink(v, 'valid')
    return stream


def acc_rshift_round(m, clk, rst,
                     datawidth, point, signed,
                     sum_width=None, sum_point=None, sum_signed=None):

    name = _tmp_name('acc')

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', datawidth, point, signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1
    size = stream.constant('size', signed=False)

    sum, v = stream.ReduceAddValid(x, size, width=sum_width, signed=sum_signed)
    if sum_point is not None and point != sum_point:
        sum = stream.Cast(sum, point=sum_point)

    sum = stream.SraRound(sum, rshift)

    stream.sink(sum, 'sum')
    stream.sink(v, 'valid')
    return stream


def acc_rshift_round_frac(m, clk, rst,
                          datawidth, point, signed,
                          sum_width=None, sum_point=None, sum_signed=None):

    name = _tmp_name('acc')

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', datawidth, point, signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1
    size = stream.constant('size', signed=False)

    frac = stream.Mux(rshift > 0, stream.Sll(1, rshift - 1), 0)
    frac.width = sum_width

    sum, v = stream.ReduceAddValid(x, size, width=sum_width, signed=sum_signed)
    if sum_point is not None and point != sum_point:
        sum = stream.Cast(sum, point=sum_point)

    sum = sum + frac
    sum = stream.Sra(sum, rshift)

    stream.sink(sum, 'sum')
    stream.sink(v, 'valid')
    return stream


def add_tree(m, clk, rst,
             datawidth, point, signed, num_vars):

    name = _tmp_name('add_tree')

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    vars = [stream.source('var%d' % i, datawidth, point, signed)
            for i in range(num_vars)]

    if len(vars) == 1:
        sum = stream.Cast(vars[0])
    else:
        sum = Add3Tree(stream, *vars)
    stream.sink(sum, 'sum')
    return stream


def add_tree_rshift(m, clk, rst,
                    datawidth, point, signed, num_vars):

    name = _tmp_name('add_tree')

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    vars = [stream.source('var%d' % i, datawidth, point, signed)
            for i in range(num_vars)]
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1

    sum = Add3Tree(stream, *vars)
    sum = stream.Sra(sum, rshift)
    stream.sink(sum, 'sum')
    return stream


def add_tree_rshift_round(m, clk, rst,
                          datawidth, point, signed, num_vars):

    name = _tmp_name('add_tree')

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    vars = [stream.source('var%d' % i, datawidth, point, signed)
            for i in range(num_vars)]
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1

    sum = Add3Tree(stream, *vars)
    sum = stream.SraRound(sum, rshift)
    stream.sink(sum, 'sum')
    return stream


def add_tree_rshift_round_frac(m, clk, rst,
                               datawidth, point, signed, num_vars):

    name = _tmp_name('add_tree')

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    vars = [stream.source('var%d' % i, datawidth, point, signed)
            for i in range(num_vars)]
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1

    frac = stream.Mux(rshift > 0, stream.Sll(1, rshift - 1), 0)
    frac.width = datawidth

    sum = Add3Tree(stream, *(vars + [frac]))
    sum = stream.Sra(sum, rshift)
    stream.sink(sum, 'sum')
    return stream


def _max(m, clk, rst,
         datawidth, point, signed, num_vars):

    name = _tmp_name('_max')

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    vars = [stream.source('var%d' % i, datawidth, point, signed)
            for i in range(num_vars)]

    val = stream.Max(*vars)
    stream.sink(val, 'val')
    return stream


def average(m, clk, rst,
            datawidth, point, signed, num_vars):

    name = _tmp_name('average')

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    vars = [stream.source('var%d' % i, datawidth, point, signed)
            for i in range(num_vars)]

    val = stream.Average(*vars)
    stream.sink(val, 'val')
    return stream


def div(m, clk, rst,
        x_datawidth, x_point, x_signed,
        y_datawidth, y_point, y_signed,
        div_width=None, div_point=None, div_signed=None):

    name = _tmp_name('div')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)

    z = stream.Div(x, y)
    if div_width is not None:
        z.width = div_width
    if div_signed is not None:
        z.signed = div_signed
    if div_point is not None and point != div_point:
        z = stream.Cast(z, point=div_point)

    stream.sink(z, 'z')
    return stream


def div_const(m, clk, rst,
              x_datawidth, x_point, x_signed,
              y_datawidth, y_point, y_signed,
              div_width=None, div_point=None, div_signed=None):

    name = _tmp_name('div_const')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)

    z = stream.Div(x, y)
    if div_width is not None:
        z.width = div_width
    if div_signed is not None:
        z.signed = div_signed
    if div_point is not None and point != div_point:
        z = stream.Cast(z, point=div_point)

    stream.sink(z, 'z')
    return stream


def div_const_frac(m, clk, rst,
              x_datawidth, x_point, x_signed,
              y_datawidth, y_point, y_signed,
              div_width=None, div_point=None, div_signed=None):

    name = _tmp_name('div_const_frac')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)

    frac= stream.source('frac')
    frac.width = datawidth

    neg_frac = stream.Uminus(frac)
    neg_frac.width = datawidth
    neg_frac.latency = 0

    frac = stream.Mux(x >= 0, frac, neg_frac)
    frac.latency = 0
    frac.width = datawidth

    x_frac = stream.Add(x, frac)
    x_frac.latency = 0
    z = stream.Div(x_frac, y)
    if div_width is not None:
        z.width = div_width
    if div_signed is not None:
        z.signed = div_signed
    if div_point is not None and point != div_point:
        z = stream.Cast(z, point=div_point)

    stream.sink(z, 'z')
    return stream


def op3_tree(op, initval, latency, *args):
    if len(args) == 0:
        return initval

    if len(args) == 1:
        return args[0]

    if len(args) == 2:
        ret = op(args[0], args[1], initval)

        if latency is not None:
            ret.latency = latency

        return ret

    if len(args) == 3:
        ret = op(*args)

        if latency is not None:
            ret.latency = latency

        return ret

    log_num = int(math.ceil(math.log(len(args), 3))) - 1
    num = 3 ** log_num

    ret = op(op3_tree(op, initval, latency, *args[:num]),
             op3_tree(op, initval, latency, *args[num: num * 2]),
             op3_tree(op, initval, latency, *args[num * 2:]))

    if latency is not None:
        ret.latency = latency

    return ret


def Add3Tree(stream, *args):
    op = stream.AddN
    return op3_tree(op, vstream.Int(0, signed=True), None, *args)


def reduce_max(m, clk, rst,
               datawidth, point, signed):

    name = _tmp_name('_reduce_max')

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', datawidth, point, signed)
    size = stream.constant('size', signed=False)

    if signed:
        initval = - 2 ** (datawidth - 1)
    else:
        initval = 0

    data, valid = stream.ReduceMaxValid(x, size, initval=initval,
                                        width=datawidth, signed=signed)

    stream.sink(data, 'data')
    stream.sink(valid, 'valid')

    return stream


def lshift_rshift(m, clk, rst,
                  x_datawidth, x_point, x_signed,
                  y_datawidth, y_point, y_signed,
                  mul_width=None, mul_point=None, mul_signed=None):

    if y_point != 0:
        raise ValueError('not supported')

    name = _tmp_name('lshift')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1

    abs_y = stream.Abs(y)
    sign_y = stream.Sign(y)

    z = stream.Sll(x, abs_y)
    z.latency = 0
    z = stream.Cast(z, signed=x_signed)
    z.latency = 0
    z = stream.Mux(sign_y, stream.Complement2(z), z)
    z.latency = 0
    z = stream.Cast(z, signed=x_signed)
    z.latency = 0
    if mul_width is not None:
        z.width = mul_width
    if mul_signed is not None:
        z.signed = mul_signed
    if mul_point is not None and point != mul_point:
        z = stream.Cast(z, point=mul_point)
        z.latency = 0
    z = stream.SraRound(z, rshift)

    stream.sink(z, 'z')
    return stream


def updown_rshift(m, clk, rst,
                  x_datawidth, x_point, x_signed,
                  y_datawidth, y_point, y_signed,
                  mul_width=None, mul_point=None, mul_signed=None):

    if y_datawidth != 1:
        raise ValueError('not supported')

    if y_point != 0:
        raise ValueError('not supported')

    if y_signed:
        raise ValueError('not supported')

    name = _tmp_name('updown')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1

    z = stream.Mux(y, x, stream.Complement2(x))
    z.latency = 0
    z = stream.Cast(z, signed=x_signed)
    z.latency = 0
    if mul_width is not None:
        z.width = mul_width
    if mul_signed is not None:
        z.signed = mul_signed
    if mul_point is not None and point != mul_point:
        z = stream.Cast(z, point=mul_point)
        z.latency = 0
    z = stream.SraRound(z, rshift)

    stream.sink(z, 'z')
    return stream


def updown_mask_rshift(m, clk, rst,
                       x_datawidth, x_point, x_signed,
                       y_datawidth, y_point, y_signed,
                       mul_width=None, mul_point=None, mul_signed=None):

    if y_datawidth != 2:
        raise ValueError('not supported')

    if y_point != 0:
        raise ValueError('not supported')

    if not y_signed:
        raise ValueError('not supported')

    name = _tmp_name('updown_mask')
    datawidth = max(x_datawidth, y_datawidth)
    point = max(x_point, y_point)

    stream = vthread.Stream(m, name, clk, rst, datawidth)
    x = stream.source('x', x_datawidth, x_point, x_signed)
    y = stream.source('y', y_datawidth, y_point, y_signed)
    rshift = stream.source('rshift', signed=False)
    rshift.width = int(math.ceil(math.log(datawidth, 2))) + 1

    z = stream.Mux(y > 0, x,
                   stream.Mux(y < 0, stream.Complement2(x), 0))
    z.latency = 0
    z = stream.Cast(z, signed=x_signed)
    z.latency = 0
    if mul_width is not None:
        z.width = mul_width
    if mul_signed is not None:
        z.signed = mul_signed
    if mul_point is not None and point != mul_point:
        z = stream.Cast(z, point=mul_point)
        z.latency = 0
    z = stream.SraRound(z, rshift)

    stream.sink(z, 'z')
    return stream
