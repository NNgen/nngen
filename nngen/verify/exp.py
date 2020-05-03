from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def exp(features,
        lut_addrwidth=8, lut_clip=6.0, lut_bias=0.0, range_rate=0.95,
        dtype=None, name=None, par=1,
        features_dtype=None, features_scale=1, features_shamt=0, features_bias=0):

    features_point = 0 if features_dtype is None else features_dtype.point
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - features_point

    mul = features * features_scale + features_bias
    sra = mul >> features_shamt

    if dtype is None:
        raise ValueError('exp requires dtype to determine the value range.')

    addr_scale = lut_clip / (2 ** (lut_addrwidth - 1))

    out_width = dtype.width
    out_point = dtype.point
    out_signed = dtype.signed
    if out_signed:
        out_scale = round((2 ** (out_width - 1)) * range_rate /
                          np.exp(2 ** (lut_addrwidth - 1) * addr_scale + lut_bias))
    else:
        out_scale = round((2 ** out_width) * range_rate /
                          np.exp(2 ** (lut_addrwidth - 1) * addr_scale + lut_bias))

    def _exp(x):
        return np.around(np.exp(x) * out_scale).astype(np.int64)

    lut = _exp(sra * addr_scale + lut_bias)

    p_th = 2 ** (lut_addrwidth - 1) - 1
    n_th = -1 * p_th

    p = np.where(sra > p_th, _exp((p_th + 1) * addr_scale + lut_bias), lut)
    n = np.where(sra < n_th, _exp((n_th - 1) * addr_scale + lut_bias), lut)
    out = np.where(sra >= 0, p, n)

    return out
