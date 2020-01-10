from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math

from . import util


def sigmoid(visitor, node):
    features = node.args[0]

    visitor.visit(features)

    scale_value = ((2 ** math.ceil(math.log(features.scale_factor, 2))) /
                   features.scale_factor)

    q_scale_value, scale_scale_factor = util.quantize_linear_scale(scale_value, 32)
    q_shamt_value = round(math.log(scale_scale_factor, 2))

    node.features_scale = q_scale_value
    node.features_shamt = q_shamt_value

    out_width = node.dtype.width
    out_point = node.dtype.point
    out_signed = node.dtype.signed
    if out_signed:
        out_scale = 1 << (out_width - 1) - 1
    else:
        out_scale = 1 << out_width - 1

    if out_point == 0:
        th_scale = out_scale
    elif out_point > 0:
        th_scale = out_scale >> out_point
    else:
        th_scale = out_scale << -1 * out_point

    node.scale_factor = float(th_scale)
