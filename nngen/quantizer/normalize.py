from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from . import util


def normalize(visitor, node):
    input = node.args[0]
    scale = node.args[1]
    bias = node.args[2]
    shamt = node.args[3]

    visitor.visit(input)
    visitor.visit(scale)
    visitor.visit(bias)
    visitor.visit(shamt)

    scale_value = scale.value
    if isinstance(scale_value, (tuple, list)):
        scale_value = np.array(scale_value)

    q_scale_value, scale_scale_factor = util.quantize_linear_scale(scale_value,
                                                                   scale.dtype.width)
    scale.set_value(q_scale_value)
    scale.scale_factor = scale_scale_factor

    bias_value = bias.value
    if isinstance(bias_value, (tuple, list)):
        bias_value = np.array(bias_value)

    q_bias_value = util.quantize_linear_by_scale_factor(
        bias_value, bias.dtype.width, input.scale_factor * scale_scale_factor)
    bias.set_value(q_bias_value)
    bias.scale_factor = input.scale_factor * scale_scale_factor

    q_shamt = find_optimal_shamt_normalize(node, q_scale_value, q_bias_value,
                                           value_ranges=visitor.value_ranges,
                                           num_trials=visitor.num_trials)
    shamt.fill_value = q_shamt
    node.scale_factor = input.scale_factor * scale.scale_factor / (2 ** q_shamt)


def find_optimal_shamt_normalize(node, scale, bias,
                                 value_ranges={}, num_trials=5,
                                 allowed_rate=0.01, input_threshold=3.0,
                                 init_shamt=0):

    shamt = init_shamt

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

            acc_overflow += try_shamt_normalize(node, input, scale, bias, shamt)

        rate = acc_overflow / (out_length * num_trials)
        if rate <= allowed_rate:
            break

        shamt += 1

    return shamt


def try_shamt_normalize(node, x, y, z, shamt):

    import nngen.verify as verify

    name = node.__class__.__name__
    method = getattr(verify, name, None)

    kwargs = {}
    kwargs['dtype'] = node.dtype
    kwargs['sum_dtype'] = node.sum_dtype
    kwargs['name'] = node.name
    kwargs['par'] = node.par

    rslt = method(x, y, z, shamt, **kwargs)

    half_range = (2 ** (node.dtype.width - 1)) - 1
    neg_overflow = np.where(rslt <= - half_range,
                            np.ones_like(rslt), np.zeros_like(rslt))
    pos_overflow = np.where(rslt >= half_range,
                            np.ones_like(rslt), np.zeros_like(rslt))
    num_overflow = np.sum(neg_overflow + pos_overflow)

    return num_overflow


def scaled_add(visitor, node):
    a = node.args[0]
    b = node.args[1]

    visitor.visit(a)
    visitor.visit(b)

    max_scale_factor = max(a.scale_factor, b.scale_factor)
    a_scale_value = max_scale_factor / a.scale_factor
    b_scale_value = max_scale_factor / b.scale_factor

    if max_scale_factor == a.scale_factor:
        q_b_scale_value, b_scale_scale_factor = util.quantize_linear_scale(b_scale_value,
                                                                           b.dtype.width)
        a_scale_value = np.round(b_scale_scale_factor).astype(np.int64)
        q_a_scale_value = a_scale_value
        a_scale_scale_factor = a_scale_value
    elif max_scale_factor == b.scale_factor:
        q_a_scale_value, a_scale_scale_factor = util.quantize_linear_scale(a_scale_value,
                                                                           a.dtype.width)
        b_scale_value = np.round(a_scale_scale_factor).astype(np.int64)
        q_b_scale_value = b_scale_value
        b_scale_scale_factor = b_scale_value

    node.a_scale = int(q_a_scale_value)
    node.b_scale = int(q_b_scale_value)

    q_shamt = find_optimal_shamt_scaled_add(node, q_a_scale_value, q_b_scale_value,
                                            value_ranges=visitor.value_ranges,
                                            num_trials=visitor.num_trials)
    node.shamt = q_shamt
    node.scale_factor = max(a.scale_factor * a_scale_scale_factor,
                            b.scale_factor * b_scale_scale_factor) / (2 ** q_shamt)


def find_optimal_shamt_scaled_add(node, a_scale, b_scale,
                                  value_ranges={}, num_trials=5,
                                  allowed_rate=0.01, input_threshold=3.0,
                                  init_shamt=0):

    shamt = init_shamt

    a_input_shape = node.args[0].shape
    a_input_length = node.args[0].length
    a_input_name = node.args[0].name

    b_input_shape = node.args[1].shape
    b_input_length = node.args[1].length
    b_input_name = node.args[1].name

    if a_input_name in value_ranges:
        min_val, max_val = value_ranges[a_input_name]
        abs_min_val = abs(min_val)
        abs_max_val = abs(max_val)
        max_abs_range = max(abs_min_val, abs_max_val)
        a_input_bits = max_abs_range.bit_length() + 1
    else:
        a_input_bits = node.args[0].dtype.width

    if b_input_name in value_ranges:
        min_val, max_val = value_ranges[b_input_name]
        abs_min_val = abs(min_val)
        abs_max_val = abs(max_val)
        max_abs_range = max(abs_min_val, abs_max_val)
        b_input_bits = max_abs_range.bit_length() + 1
    else:
        b_input_bits = node.args[1].dtype.width

    out_length = node.length

    while True:
        acc_overflow = 0

        for _ in range(num_trials):
            a_input = np.random.normal(size=a_input_length).reshape(a_input_shape)
            a_input = np.clip(a_input, -input_threshold, input_threshold)
            a_input = a_input * (2.0 ** (a_input_bits - 1) - 1) / input_threshold
            a_input = np.round(a_input).astype(np.int64)

            b_input = np.random.normal(size=b_input_length).reshape(b_input_shape)
            b_input = np.clip(b_input, -input_threshold, input_threshold)
            b_input = b_input * (2.0 ** (b_input_bits - 1) - 1) / input_threshold
            b_input = np.round(b_input).astype(np.int64)

            acc_overflow += try_shamt_scaled_add(node, a_input, a_scale,
                                                 b_input, b_scale, shamt)

        rate = acc_overflow / (out_length * num_trials)
        if rate <= allowed_rate:
            break

        shamt += 1

    return shamt


def try_shamt_scaled_add(node, a, a_scale, b, b_scale, shamt):

    import nngen.verify as verify

    name = node.__class__.__name__
    method = getattr(verify, name, None)

    kwargs = {}
    kwargs['dtype'] = node.dtype
    kwargs['sum_dtype'] = node.sum_dtype
    kwargs['name'] = node.name
    kwargs['par'] = node.par

    rslt = method(a, b, a_scale, b_scale, shamt, **kwargs)

    half_range = (2 ** (node.dtype.width - 1)) - 1
    neg_overflow = np.where(rslt <= - half_range,
                            np.ones_like(rslt), np.zeros_like(rslt))
    pos_overflow = np.where(rslt >= half_range,
                            np.ones_like(rslt), np.zeros_like(rslt))
    num_overflow = np.sum(neg_overflow + pos_overflow)

    return num_overflow


def scaled_concat(visitor, node):
    values = node.args

    for value in values:
        visitor.visit(value)

    max_scale_factor = max(*[value.scale_factor for value in values])
    scale_values = [max_scale_factor / value.scale_factor for value in values]

    new_scale_values = []
    scale_scale_factors = []

    for value, scale_value in zip(values, scale_values):
        if max_scale_factor != value.scale_factor:
            q_scale_value, scale_scale_factor = util.quantize_linear_scale(scale_value,
                                                                           value.dtype.width)
            new_scale_values.append(q_scale_value)
            scale_scale_factors.append(scale_scale_factor)
        else:
            new_scale_values.append(1)
            scale_scale_factors.append(1.0)

    max_scale_scale_factor = max(*scale_scale_factors)

    new_scales = []
    new_scale_scale_factors = []

    for value, new_scale_value, scale_scale_factor in zip(values,
                                                          new_scale_values, scale_scale_factors):

        mag = np.round(max_scale_scale_factor / scale_scale_factor).astype(np.int64)
        scale_value = new_scale_value * mag
        q_scale_value = scale_value
        scale_scale_factor = scale_scale_factor * mag
        new_scales.append(int(scale_value))
        new_scale_scale_factors.append(scale_scale_factor)

    node.scales = new_scales

    q_shamt = find_optimal_shamt_scaled_concat(node, new_scales,
                                               value_ranges=visitor.value_ranges,
                                               num_trials=visitor.num_trials)

    node.shamt = q_shamt
    node.scale_factor = max(*[value.scale_factor * scale_scale_factor
                              for value, scale_scale_factor in zip(
                                  values, new_scale_scale_factors)]) / (2 ** q_shamt)


def find_optimal_shamt_scaled_concat(node, scales,
                                     value_ranges={}, num_trials=5,
                                     allowed_rate=0.01, input_threshold=3.0):

    shamt = 0

    input_bits_list = []
    for arg in node.args:
        if arg.name in value_ranges:
            min_val, max_val = value_ranges[arg.name]
            abs_min_val = abs(min_val)
            abs_max_val = abs(max_val)
            max_abs_range = max(abs_min_val, abs_max_val)
            input_bits = max_abs_range.bit_length() + 1
        else:
            input_bits = arg.dtype.width

        input_bits_list.append(input_bits)

    out_length = node.length

    while True:
        acc_overflow = 0

        for _ in range(num_trials):
            inputs = []
            for arg, input_bits in zip(node.args, input_bits_list):
                input = np.random.normal(size=arg.length).reshape(arg.shape)
                input = np.clip(input, -input_threshold, input_threshold)
                input = input * (2.0 ** (input_bits - 1) - 1) / input_threshold
                input = np.round(input).astype(np.int64)
                inputs.append(input)

            acc_overflow += try_shamt_scaled_concat(node, inputs, scales, shamt)

        rate = acc_overflow / (out_length * num_trials)
        if rate <= allowed_rate:
            break

        shamt += 1

    return shamt


def try_shamt_scaled_concat(node, values, scales, shamt):

    import nngen.verify as verify

    name = node.__class__.__name__
    method = getattr(verify, name, None)

    kwargs = {}
    kwargs['axis'] = node.axis
    kwargs['dtype'] = node.dtype
    kwargs['mul_dtype'] = node.mul_dtype
    kwargs['name'] = node.name

    rslt = method(values, scales, shamt, **kwargs)

    half_range = (2 ** (node.dtype.width - 1)) - 1
    neg_overflow = np.where(rslt <= - half_range,
                            np.ones_like(rslt), np.zeros_like(rslt))
    pos_overflow = np.where(rslt >= half_range,
                            np.ones_like(rslt), np.zeros_like(rslt))
    num_overflow = np.sum(neg_overflow + pos_overflow)

    return num_overflow
