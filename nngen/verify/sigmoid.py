from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def sigmoid(features,
            lut_addrwidth=8, lut_clip=6.0, range_rate=0.8,
            dtype=None, name=None, par=1,
            features_dtype=None, features_scale=1, features_shamt=0):

    features_point = 0 if features_dtype is None else features_dtype.point
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - features_point

    mul = features * features_scale
    sra = mul >> features_shamt

    mask = 2 ** lut_addrwidth - 1

    if dtype is None:
        raise ValueError('sigmoid requires dtype to determine the value range.')

    out_width = dtype.width
    out_point = dtype.point
    out_signed = dtype.signed
    if out_signed:
        out_scale = round((2 ** (out_width - 1)) * range_rate)
    else:
        out_scale = round((2 ** out_width) * range_rate)

    def _sigmoid(x):
        return np.around((1 / (1 + np.exp(-x))) * out_scale).astype(np.int64)

    addr_scale = lut_clip / (2 ** (lut_addrwidth - 1))
    lut = _sigmoid(sra * addr_scale)

    p_th = 2 ** (lut_addrwidth - 1) - 1
    n_th = -1 * p_th

    if out_point == 0:
        th_scale = out_scale
    elif out_point > 0:
        th_scale = out_scale >> out_point
    else:
        th_scale = out_scale << -1 * out_point

    p = np.where(sra > p_th, th_scale, lut)
    n = np.where(sra < n_th, -1 * th_scale, lut)
    out = np.where(sra >= 0, p, n)

    return out
