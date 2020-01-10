from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.basic_types as bt


class sigmoid(bt._ActFuncOperator):

    def __init__(self, features, features_scale=1, features_shamt=0,
                 lut_addrwidth=8, lut_clip=6.0,
                 dtype=None, name=None, par=1):

        self.features_scale = features_scale
        self.features_shamt = features_shamt
        self.lut_addrwidth = lut_addrwidth
        self.lut_clip = lut_clip
        bt._ActFuncOperator.__init__(self, features,
                                     dtype=dtype, shape=shape, name=name, par=par)

    def get_local_control_param_values(self):
        return OrderedDict([('features_scale_cparam', self.features_scale),
                            ('features_shamt_cparam', self.features_shamt)])

    def get_stream_hash(self):
        base = bt._ActFuncOperator.get_stream_hash(self)
        return (base, self.lut_addrwidth, self.lut_clip)

    def op(self, strm, *args, **kwargs):
        features_datawidth = self.args[0].get_op_width()
        features_point = self.args[0].get_op_point()
        features_signed = self.args[0].get_signed()

        features_scale = strm.ReinterpretCast(self.features_scale_cparam,
                                              width=features_datawidth,
                                              signed=features_signed)
        mul = strm.Times(args[0], features_scale)

        features_shamt = strm.ReinterpretCast(self.features_shamt_cparam,
                                              width=self.shamt_cparam.width,
                                              signed=False)
        sra = strm.Sra(mul, features_shamt)
        lut_addr = strm.Slice(sra, self.lut_addrwidth - 1, 0)

        out_width = self.dtype.width
        out_point = self.dtype.point
        out_signed = self.dtype.signed
        if out_signed:
            out_scale = 1 << (out_width - 1) - 1
        else:
            out_scale = 1 << out_width - 1

        def _sigmoid(x):
            return round((1 / (1 + np.exp(-x))) * out_scale)

        patterns = [i * self.lut_clip * 2.0 / (2 ** self.lut_addrwidth) - self.lut_clip
                    for i in range(2 ** self.lut_addrwidth)]
        lut = strm.LUT(lut_addr, patterns, out_width, out_point, out_signed)

        p_th = 2 ** (self.lut_addrwidth - 1) - 1
        n_th = -1 * input_p_th

        if out_point == 0:
            th_scale = out_scale
        elif out_point > 0:
            th_scale = out_scale >> out_point
        else:
            th_scale = out_scale << -1 * out_point

        p = strm.Mux(sra > p_th, th_scale, lut)
        n = strm.Mux(sra < n_th, -1 * th_scale, lut)
        out = strm.Mux(sra >= 0, p, n)

        return out

    def eval(self, memo, input_dict, **kwargs):
        kwargs['features_scale'] = self.features_scale
        kwargs['features_shamt'] = self.features_shamt
        kwargs['lut_addrwidth'] = self.lut_addrwidth
        kwargs['lut_clip'] = self.lut_clip
        kwargs['features_dtype'] = self.args[0].dtype
        return bt._ActFuncOperator.eval(self, memo, input_dict, **kwargs)
