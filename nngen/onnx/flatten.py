from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.operator as operator

from . import util


def Flatten(visitor, node, no_transpose=False):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    input = srcs[0]

    for attribute in node.attribute:
        if attribute.name == 'axis':
            axis = attribute.i

    if (not no_transpose and
        input.get_layout() is not None and input.get_onnx_layout() is not None and
            input.get_layout() != input.get_onnx_layout()):

        perm = [input.get_layout().index(l) for l in input.get_onnx_layout()]
        onnx_perm = [i for i, l in enumerate(input.get_onnx_layout())]

        input = operator.transpose(input, perm)
        input.transpose_onnx_perm = onnx_perm

        input.layout = input.get_onnx_layout()

    layout = input.get_layout()
    onnx_layout = input.get_onnx_layout()

    if layout is not None and onnx_layout is not None:
        axis = layout.index(onnx_layout[axis])

    name = util.get_name(node)

    if axis == 0:
        shape = [1, -1]
    else:
        shape = [int(np.prod(input.shape[:axis])), -1]

    c = operator.reshape(input, shape, name=name)

    return c
