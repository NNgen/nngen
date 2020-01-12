from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import functools
import math
import numpy as np
from collections import OrderedDict

import nngen.basic_types as bt
from nngen.quantizer import util


class sigmoid(bt._ActFuncOperator):

    def __init__(self, features,
                 lut_addrwidth=8, lut_clip=6.0, range_rate=0.99,
                 dtype=None, name=None, par=1):

        shape = None
        if features.dtype is not None and features.dtype.width < 8:
            lut_addrwidth = features.dtype.width
        self.lut_addrwidth = lut_addrwidth
        self.lut_clip = lut_clip
        self.range_rate = range_rate
        bt._ActFuncOperator.__init__(self, features,
                                     dtype=dtype, shape=shape, name=name, par=par)

    def get_local_control_param_values(self):
        features_scale = np.array([((2 ** math.ceil(math.log(self.args[0].scale_factor, 2))) /
                                    self.args[0].scale_factor)])
        q_features_scale, scale_factor = util.quantize_linear_scale(features_scale, 32)
        q_features_shamt = round(math.log(scale_factor, 2))
        return OrderedDict([('features_scale_cparam', int(q_features_scale[0])),
                            ('features_shamt_cparam', q_features_shamt)])

    def get_stream_hash(self):
        base = bt._ActFuncOperator.get_stream_hash(self)
        return (base, self.lut_addrwidth, self.lut_clip, self.range_rate)

    def op(self, strm, *args, **kwargs):
        features_datawidth = self.args[0].get_op_width()
        features_point = self.args[0].get_op_point()
        features_signed = self.args[0].get_signed()

        features_scale = strm.ReinterpretCast(self.features_scale_cparam,
                                              width=features_datawidth,
                                              signed=features_signed)
        mul = strm.Times(args[0], features_scale)

        features_shamt = strm.ReinterpretCast(self.features_shamt_cparam,
                                              width=self.features_shamt_cparam.width,
                                              signed=False)
        sra = strm.Sra(mul, features_shamt)
        lut_addr = strm.Slice(sra, self.lut_addrwidth - 1, 0)

        out_width = self.dtype.width
        out_point = self.dtype.point
        out_signed = self.dtype.signed
        if out_signed:
            out_scale = round((2 ** (out_width - 1)) * self.range_rate)
        else:
            out_scale = round((2 ** out_width) * self.range_rate)

        def _sigmoid(x):
            return int(np.around((1 / (1 + np.exp(-x))) * out_scale).astype(np.int64))

        addr_scale = self.lut_clip / (2 ** (self.lut_addrwidth - 1))
        patterns_p = [_sigmoid(i * addr_scale)
                      for i in range(2 ** (self.lut_addrwidth - 1))]
        patterns_n = [_sigmoid((-i - 1) * addr_scale)
                      for i in range(2 ** (self.lut_addrwidth - 1))]
        patterns_n.reverse()

        patterns = patterns_p + patterns_n

        lut = strm.LUT(lut_addr, patterns, out_width, out_point, out_signed)

        p_th = 2 ** (self.lut_addrwidth - 1) - 1
        n_th = -1 * p_th

        if out_point == 0:
            th_scale = out_scale
        elif out_point > 0:
            th_scale = out_scale >> out_point
        else:
            th_scale = out_scale << (-1 * out_point)

        p = strm.Mux(sra > p_th, th_scale, lut)
        n = strm.Mux(sra < n_th, 0, lut)
        out = strm.Mux(sra >= 0, p, n)

        return out

    def get_eval_method(self):
        import nngen.verify as verify

        name = self.__class__.__name__
        method = getattr(verify, name, None)

        features_scale = np.array([((2 ** math.ceil(math.log(self.args[0].scale_factor, 2))) /
                                    self.args[0].scale_factor)])
        q_features_scale, scale_factor = util.quantize_linear_scale(features_scale, 32)
        q_features_shamt = round(math.log(scale_factor, 2))

        method = functools.partial(method,
                                   lut_addrwidth=self.lut_addrwidth,
                                   lut_clip=self.lut_clip,
                                   range_rate=self.range_rate,
                                   features_dtype=self.args[0].dtype,
                                   features_scale=int(q_features_scale[0]),
                                   features_shamt=q_features_shamt)
        return method
