from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from . import util


def conv2d(visitor, node):

    input = node.args[0]
    filter = node.args[1]

    bias = node.args[node.args_dict['bias']] if node.has_bias else None
    scale = node.args[node.args_dict['scale']] if node.has_scale else None

    rshift_mul = (node.args[node.args_dict['vshamt_mul']]
                  if node.has_vshamt_mul else node.cshamt_mul)
    rshift_sum = (node.args[node.args_dict['vshamt_sum']]
                  if node.has_vshamt_sum else node.cshamt_sum)
    rshift_out = (node.args[node.args_dict['vshamt_out']]
                  if node.has_vshamt_out else node.cshamt_out)

    visitor.visit(input)
    visitor.visit(filter)

    if bias is not None:
        visitor.visit(bias)

    if scale is not None:
        visitor.visit(scale)

    if rshift_mul is not None:
        visitor.visit(rshift_mul)

    if rshift_sum is not None:
        visitor.visit(rshift_sum)

    if rshift_out is not None:
        visitor.visit(rshift_out)

    q_filter_value, filter_scale_factor = util.quantize_linear(filter.value, filter.dtype.width)
    filter.set_value(q_filter_value)
    filter.scale_factor = filter_scale_factor

    if bias is not None:
        bias_value = bias.value
        if isinstance(bias_value, (tuple, list)):
            bias_value = np.array(bias_value)

        q_bias_value = util.quantize_linear_by_scale_factor(
            bias_value, bias.dtype.width, input.scale_factor * filter_scale_factor)
        bias.set_value(q_bias_value)
        bias.scale_factor = input.scale_factor * filter_scale_factor
    else:
        q_bias_value = None

    if scale is not None:
        scale_value = scale.value
        if isinstance(scale_value, (tuple, list)):
            scale_value = np.array(scale_value)

        q_scale_value, scale_scale_factor = util.quantize_linear_scale(scale_value,
                                                                       scale.dtype.width)
        scale.set_value(q_scale_value)
        scale.scale_factor = scale_scale_factor
    else:
        scale_scale_factor = 1.0
        q_scale_value = None

    if ((rshift_mul is None or isinstance(rshift_mul, int)) and
        (rshift_sum is None or isinstance(rshift_sum, int)) and
            (rshift_out is None or isinstance(rshift_out, int))):

        q_filter_value_bits = int(np.max(np.abs(q_filter_value))).bit_length()
        q_rshift_mul, q_rshift_sum, q_rshift_out = find_optimal_rshift(
            node, q_filter_value, q_bias_value, q_scale_value,
            value_ranges=visitor.value_ranges,
            num_trials=visitor.num_trials,
            init_rshift_mul=0,
            init_rshift_sum=0,
            init_rshift_out=q_filter_value_bits)

        total_rshift = 0

        if node.cshamt_mul is not None:
            node.cshamt_mul += q_rshift_mul
            total_rshift += node.cshamt_mul
        elif q_rshift_mul > 0:
            node.cshamt_mul = q_rshift_mul
            total_rshift += node.cshamt_mul

        if node.cshamt_sum is not None:
            node.cshamt_sum += q_rshift_sum
            total_rshift += node.cshamt_sum
        elif q_rshift_sum > 0:
            node.cshamt_sum = q_rshift_sum
            total_rshift += node.cshamt_sum

        if node.cshamt_out is not None:
            node.cshamt_out += q_rshift_out
            total_rshift += node.cshamt_out
        elif q_rshift_out > 0:
            node.cshamt_out = q_rshift_out
            total_rshift += node.cshamt_out

        node.scale_factor = (input.scale_factor * filter_scale_factor *
                             scale_scale_factor / (2 ** total_rshift))

    else:
        node.scale_factor = (input.scale_factor * filter_scale_factor *
                             scale_scale_factor)


def find_optimal_rshift(node, filter, bias, scale,
                        value_ranges={}, num_trials=5,
                        allowed_rate=0.01, input_threshold=3.0,
                        init_rshift_mul=0, init_rshift_sum=0, init_rshift_out=0):

    rshift_mul = init_rshift_mul
    rshift_sum = init_rshift_sum
    rshift_out = init_rshift_out

    input_shape = node.args[0].shape
    input_length = node.args[0].length
    input_name = node.args[0].name

    if input_name in value_ranges:
        min_val, max_val = value_ranges[input_name]
        abs_min_val = abs(min_val)
        abs_max_val = abs(max_val)
        max_abs_range = max(abs_min_val, abs_max_val)
        input_bits = max_abs_range.bit_length() + 1
    else:
        input_bits = node.args[0].dtype.width

    out_length = node.length

    while True:
        acc_overflow = 0

        for _ in range(num_trials):
            input = np.random.normal(size=input_length).reshape(input_shape)
            input = np.clip(input, -input_threshold, input_threshold)
            input = input * (2.0 ** (input_bits - 1) - 1) / input_threshold
            input = np.round(input).astype(np.int64)

            acc_overflow += try_rshift(node, input, filter, bias, scale,
                                       rshift_mul, rshift_sum, rshift_out)

        rate = acc_overflow / (out_length * num_trials)
        if rate <= allowed_rate:
            break

        rshift_out += 1

    return rshift_mul, rshift_sum, rshift_out


def try_rshift(node, input, filter, bias, scale,
               rshift_mul, rshift_sum, rshift_out):

    import nngen.verify as verify

    name = node.__class__.__name__
    method = getattr(verify, name, None)

    strides = node.strides

    kwargs = {}
    kwargs['strides'] = strides
    kwargs['bias'] = bias
    kwargs['scale'] = scale
    kwargs['rshift_mul'] = rshift_mul
    kwargs['rshift_sum'] = rshift_sum
    kwargs['rshift_out'] = rshift_out
    kwargs['act_func'] = node.act_func
    kwargs['padding'] = node.padding
    kwargs['dtype'] = node.dtype
    kwargs['mul_dtype'] = node.mul_dtype
    kwargs['sum_dtype'] = node.sum_dtype
    kwargs['name'] = node.name
    kwargs['par_ich'] = node.par_ich
    kwargs['par_och'] = node.par_och
    kwargs['par_col'] = node.par_col
    kwargs['par_row'] = node.par_row
    kwargs['concur_och'] = node.concur_och
    kwargs['stationary'] = node.stationary

    if 'matmul' in method.__name__:
        del kwargs['strides']
        del kwargs['padding']
        del kwargs['par_ich']
        del kwargs['par_och']
        del kwargs['par_col']
        del kwargs['par_row']
        del kwargs['concur_och']

    rslt = method(input, filter, **kwargs)

    half_range = (2 ** (node.dtype.width - 1)) - 1
    neg_overflow = np.where(rslt <= - half_range,
                            np.ones_like(rslt), np.zeros_like(rslt))
    pos_overflow = np.where(rslt >= half_range,
                            np.ones_like(rslt), np.zeros_like(rslt))
    num_overflow = np.sum(neg_overflow + pos_overflow)

    return num_overflow
