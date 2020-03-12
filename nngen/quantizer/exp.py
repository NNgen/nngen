from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def exp(visitor, node):
    features = node.args[0]

    visitor.visit(features)

    addr_scale = 1 / node._get_expected_scale_factor()

    out_width = node.dtype.width
    out_point = node.dtype.point
    out_signed = node.dtype.signed
    if out_signed:
        out_scale = round((2 ** (out_width - 1)) * node.range_rate /
                          np.exp(2 ** (node.lut_addrwidth - 1) * addr_scale + node.lut_bias))
    else:
        out_scale = round((2 ** out_width) * node.range_rate /
                          np.exp(2 ** (node.lut_addrwidth - 1) * addr_scale + node.lut_bias))

    if out_point == 0:
        th_scale = out_scale
    elif out_point > 0:
        th_scale = out_scale >> out_point
    else:
        th_scale = out_scale << -1 * out_point

    node.scale_factor = float(th_scale)
