from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


def sigmoid(visitor, node):
    features = node.args[0]

    visitor.visit(features)

    out_width = node.dtype.width
    out_point = node.dtype.point
    out_signed = node.dtype.signed
    if out_signed:
        out_scale = round((2 ** (out_width - 1)) * node.range_rate)
    else:
        out_scale = round((2 ** out_width) * node.range_rate)

    if out_point == 0:
        th_scale = out_scale
    elif out_point > 0:
        th_scale = out_scale >> out_point
    else:
        th_scale = out_scale << -1 * out_point

    node.scale_factor = float(th_scale)
