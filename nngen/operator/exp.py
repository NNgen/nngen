from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import functools
import math
import numpy as np
from collections import OrderedDict

import nngen.basic_types as bt
from nngen.quantizer import util


class exp(bt._ElementwiseOperator):

    def __init__(self, features,
                 lut_addrwidth=8, lut_clip=6.0, lut_bias=0.0, range_rate=0.95,
                 dtype=None, name=None, par=1):

        shape = None
        if features.dtype is not None and features.dtype.width < 8:
            lut_addrwidth = features.dtype.width

        self.lut_addrwidth = lut_addrwidth
        self.lut_clip = lut_clip
        self.lut_bias = lut_bias
        self.range_rate = range_rate
        bt._ElementwiseOperator.__init__(self, features,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def _get_expected_scale_factor(self):
        return (2 ** (self.lut_addrwidth - 1)) / self.lut_clip

    def _get_features_scale_shamt_bias(self):
        expected_scale_factor = self._get_expected_scale_factor()

        features_scale = np.array([expected_scale_factor / self.args[0].scale_factor])
        q_features_scale, scale_factor = util.quantize_linear_scale(features_scale, 32)
        q_features_scale = int(q_features_scale[0])
        q_features_shamt = round(math.log(scale_factor, 2))
        q_features_bias = round(q_features_scale * self.lut_bias * -1.0)
        return q_features_scale, q_features_shamt, q_features_bias

    def get_local_control_param_values(self):
        (q_features_scale, q_features_shamt,
         q_features_bias) = self._get_features_scale_shamt_bias()
        return OrderedDict([('features_scale_cparam', q_features_scale),
                            ('features_shamt_cparam', q_features_shamt),
                            ('features_bias_cparam', q_features_bias)])

    def get_stream_hash(self):
        base = bt._ActFuncOperator.get_stream_hash(self)
        return (base, self.lut_addrwidth, self.lut_clip, self.lut_bias, self.range_rate)

    def op(self, strm, *args, **kwargs):
        features_signed = self.args[0].get_signed()

        features_scale = strm.ReinterpretCast(self.features_scale_cparam,
                                              width=self.features_scale_cparam.width + 1,
                                              signed=features_signed)
        features_bias = strm.ReinterpretCast(self.features_bias_cparam,
                                             width=self.features_bias_cparam.width + 1,
                                             signed=True)

        mul = strm.Madd(args[0], features_scale, features_bias)
        mul.width = max(mul.width + features_scale.width, features_bias.width)

        features_shamt = strm.ReinterpretCast(self.features_shamt_cparam,
                                              width=self.features_shamt_cparam.width,
                                              signed=False)
        sra = strm.Sra(mul, features_shamt)
        lut_addr = strm.Slice(sra, self.lut_addrwidth - 1, 0)

        addr_scale = 1 / self._get_expected_scale_factor()

        out_width = self.dtype.width
        out_point = self.dtype.point
        out_signed = self.dtype.signed
        if out_signed:
            out_scale = round((2 ** (out_width - 1)) * self.range_rate /
                              np.exp(2 ** (self.lut_addrwidth - 1) * addr_scale + self.lut_bias))
        else:
            out_scale = round((2 ** out_width) * self.range_rate /
                              np.exp(2 ** (self.lut_addrwidth - 1) * addr_scale + self.lut_bias))

        def _exp(x):
            return int(np.around(np.exp(x) * out_scale).astype(np.int64))

        patterns_p = [_exp(i * addr_scale + self.lut_bias)
                      for i in range(2 ** (self.lut_addrwidth - 1))]
        patterns_n = [_exp((-i - 1) * addr_scale + self.lut_bias)
                      for i in range(2 ** (self.lut_addrwidth - 1))]
        patterns_n.reverse()

        patterns = patterns_p + patterns_n

        lut = strm.LUT(lut_addr, patterns, out_width, out_point, out_signed)

        p_th = 2 ** (self.lut_addrwidth - 1) - 1
        n_th = -1 * p_th

        p = strm.Mux(sra > p_th, _exp((p_th + 1) * addr_scale + self.lut_bias), lut)
        n = strm.Mux(sra < n_th, _exp((n_th - 1) * addr_scale + self.lut_bias), lut)
        out = strm.Mux(sra >= 0, p, n)

        return out

    def get_eval_method(self):
        import nngen.verify as verify

        name = self.__class__.__name__
        method = getattr(verify, name, None)

        features_scale, features_shamt, features_bias = self._get_features_scale_shamt_bias()

        method = functools.partial(method,
                                   lut_addrwidth=self.lut_addrwidth,
                                   lut_clip=self.lut_clip,
                                   lut_bias=self.lut_bias,
                                   range_rate=self.range_rate,
                                   features_dtype=self.args[0].dtype,
                                   features_scale=features_scale,
                                   features_shamt=features_shamt,
                                   features_bias=features_bias)
        return method
