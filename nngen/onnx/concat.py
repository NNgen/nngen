from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.operator as operator
import nngen.dtype_list as dtype_list

from . import util


def Concat(visitor, node):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    for attribute in node.attribute:
        if attribute.name == 'axis':
            axis = attribute.i

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

    onnx_layout = visitor.onnx_input_layout

    layout = None
    for src in srcs:
        l = util.get_layout(src)
        if l is None:
            continue
        if layout is None:
            layout = l
        elif layout != l:
            raise ValueError("layout mismatch: '%s' != '%s'" % (layout, l))

    if layout is not None:
        axis = layout.index(onnx_layout[axis])

    c = operator.scaled_concat(srcs, scales, shamt, axis, name=name)

    return c
