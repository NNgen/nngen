from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import math


def quantize_linear(orig_weight, width=8):
    if width > 16:
        width = 16

    return _quantize_linear(orig_weight, width)


def _quantize_linear(orig_weight, width=8, range_rate=0.995):
    if isinstance(orig_weight, (tuple, list)):
        orig_weight = np.array(orig_weight)

    if width >= 8:
        max_value = np.max(orig_weight)
        min_value = np.min(orig_weight)
    else:
        sorted_abs_weight = np.sort(np.abs(orig_weight).reshape([-1]))
        max_value = sorted_abs_weight[math.ceil(orig_weight.size * range_rate) - 1]
        min_value = - max_value
        orig_weight = np.clip(orig_weight, min_value, max_value)

    abs_max = max(abs(max_value), abs(min_value))

    pos_num_quantized_bins = 2 ** (width - 1) - 1
    scale_factor = 1.0 * pos_num_quantized_bins / abs_max
    quantized_weight = np.round(orig_weight * scale_factor).astype(np.int64)

    return quantized_weight, scale_factor


def quantize_linear_by_scale_factor(value, width, scale_factor):
    half_range = 2 ** (width - 1) - 1
    v = np.round(value * scale_factor).astype(np.int64)
    v = np.clip(v, - half_range, half_range)
    return v


def quantize_linear_scale(scale_value, width, allowed_rate=0.01, step=1.414):
    scale_scale_factor = find_optimal_scale_scale_factor(scale_value, width,
                                                         allowed_rate, step)
    q_scale_value = np.round(scale_value * scale_scale_factor).astype(np.int64)
    return q_scale_value, scale_scale_factor


def find_optimal_scale_scale_factor(scale_value, width, allowed_rate=0.01, step=1.414):
    scale_scale_factor = 1.0

    abs_max = np.max(np.abs(scale_value))
    half_range = 2 ** (width - 1) - 1
    if abs_max > half_range:
        scale_scale_factor = 1.0 * half_range / abs_max

    prev_scale_scale_factor = scale_scale_factor

    while True:
        float_value = np.array(scale_value * scale_scale_factor)
        round_value = np.round(float_value)

        if np.max(np.abs(round_value)) > 2 ** (width - 1) - 1:
            scale_scale_factor = prev_scale_scale_factor
            break

        cont = False
        for i, (f, r) in enumerate(sorted(zip(float_value.reshape([-1]),
                                              round_value.reshape([-1])),
                                          key=lambda x: abs(x[0]), reverse=True)):

            rate = abs(r - f) / (f + 0.0000001)
            if rate > allowed_rate:
                cont = True
                break

            if i >= float_value.size / 10:
                break

        if not cont:
            break

        prev_scale_scale_factor = scale_scale_factor
        scale_scale_factor *= step

    return scale_scale_factor


def quantize_linear_torch_parameter_by_scale_factor(orig_param, scale_factor):
    try:
        import torch
    except:
        raise ImportError('pytorch is required.')

    v = orig_param.data.numpy() * scale_factor
    v = np.round(v).astype(np.int64).astype(np.float32)
    orig_param.data = torch.from_numpy(v)


def to_hist_with_orig_bins(targ_hist, targ_bins, orig_hist, orig_bins):
    targ_v = 0.0
    targ_i = 0
    targ_bin = targ_bins[0]
    ret_hist = np.zeros_like(orig_hist)

    for i, orig_bin in enumerate(orig_bins[:-1]):
        if targ_bin <= orig_bin:
            if targ_i < len(targ_bins) - 1:
                targ_v = targ_hist[targ_i]
                targ_i += 1
                targ_bin = targ_bins[targ_i]
            else:
                targ_v = 0.0
                targ_bin = orig_bin.max() + 1.0

        ret_hist[i] = targ_v

    return ret_hist
