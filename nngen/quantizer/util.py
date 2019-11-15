from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def quantize_linear(orig_weight, num_bits=8):
    if num_bits > 16:
        num_bits = 16

    return _quantize_linear(orig_weight, num_bits)


def _quantize_linear(orig_weight, num_bits=8):
    if isinstance(orig_weight, (tuple, list)):
        orig_weight = np.array(orig_weight)

    max_value = np.max(orig_weight)
    min_value = np.min(orig_weight)
    abs_max = max(abs(max_value), abs(min_value))

    pos_num_quantized_bins = 2 ** (num_bits - 1) - 1
    scale_factor = 1.0 * pos_num_quantized_bins / abs_max
    quantized_weight = np.round(orig_weight * scale_factor).astype(np.int64)

    return quantized_weight, scale_factor


def quantize_linear_by_scale_factor(value, width, scale_factor):
    half_range = 2 ** (width - 1) - 1
    v = np.round(value * scale_factor).astype(np.int64)
    v = np.clip(v, - half_range, half_range)
    return v


def quantize_linear_scale(scale_value, width, allowed_rate=0.05):
    scale_scale_factor = find_optimal_scale_scale_factor(scale_value, width, allowed_rate)
    q_scale_value = np.round(scale_value * scale_scale_factor).astype(np.int64)
    return q_scale_value, scale_scale_factor


def find_optimal_scale_scale_factor(scale_value, width, allowed_rate=0.05):
    scale_scale_factor = 1.0

    while True:
        float_value = scale_value * scale_scale_factor
        round_value = np.round(float_value)
        rate = np.max(np.abs((float_value - round_value) / float_value))

        if rate <= allowed_rate:
            break

        scale_scale_factor *= 2.0

        if np.max(np.abs(round_value)) >= 2 ** (width - 1) - 1:
            break

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
