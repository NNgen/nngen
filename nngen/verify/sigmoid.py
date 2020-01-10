from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def sigmoid(features, features_scale, features_shamt,
            lut_addrwidth=8, lut_clip=6.0,
            dtype=None, name=None, par=1,
            features_dtype=None):

    features_point = 0 if features_dtype is None else features_dtype.point
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - features_point

    mul = features * features_scale
    sra = mul >> features_shamt

    mask = 2 ** lut_addrwidth - 1
    lut_addr = sra & mask

    if dtype is None:
        raise ValueError('sigmoid requires dtype to determine the value range.')

    out_width = dtype.width
    out_point = dtype.point
    out_signed = dtype.signed
    if out_signed:
        out_scale = 1 << (out_width - 1) - 1
    else:
        out_scale = 1 << out_width - 1

    def _sigmoid(x):
        return round((1 / (1 + np.exp(-x))) * out_scale)

    lut = _sigmoid(features)

    p_th = 2 ** (lut_addrwidth - 1) - 1
    n_th = -1 * input_p_th

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
