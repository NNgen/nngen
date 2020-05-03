from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.operator as operator

from . import util


def Concat(visitor, node):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    axis = None
    for attribute in node.attribute:
        if attribute.name == 'axis':
            axis = attribute.i

    srcs = [util.optimize_to_raw_value(src) for src in srcs]

    all_ndarray = True
    for src in srcs:
        if not isinstance(src, np.ndarray):
            all_ndarray = False
            break

    if all_ndarray:
        return np.concatenate(srcs, axis)

    name = util.get_name(node)

    scales = [1.0 for src in srcs]
    shamt = 0

    layout = None
    onnx_layout = None
    for src in srcs:
        l = src.get_layout()
        if l is None:
            continue
        if layout is None:
            layout = l
        elif layout != l:
            raise ValueError("layout mismatch: '%s' != '%s'" % (layout, l))

        l = src.get_onnx_layout()
        if l is None:
            continue
        if onnx_layout is None:
            onnx_layout = l
        elif onnx_layout != l:
            raise ValueError("onnx_layout mismatch: '%s' != '%s'" % (onnx_layout, l))

    if layout is not None and onnx_layout is not None:
        axis = layout.index(onnx_layout[axis])

    c = operator.scaled_concat(srcs, scales, shamt, axis, name=name)

    return c
