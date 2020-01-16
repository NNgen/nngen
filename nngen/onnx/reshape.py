from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import functools

import nngen.operator as operator

from . import util


def Reshape(visitor, node, no_transpose=False):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    input = srcs[0]
    shape = srcs[1]

    if isinstance(shape, np.ndarray):
        shape = shape.tolist()

    if isinstance(input, (tuple, list)):
        input = np.array(input)

    if isinstance(input, np.ndarray):
        c = np.reshape(input, shape)
        return c

    if (not no_transpose and
        input.get_layout() is not None and input.get_onnx_layout() is not None and
            input.get_layout() != input.get_onnx_layout()):

        perm = [input.get_layout().index(l) for l in input.get_onnx_layout()]
        onnx_perm = [i for i, l in enumerate(input.get_onnx_layout())]

        input = operator.transpose(input, perm)
        input.transpose_onnx_perm = onnx_perm

        input.layout = input.get_onnx_layout()

    name = util.get_name(node)

    if not isinstance(shape, (tuple, list)):
        raise TypeError('shape must be tuple or list, not %s' % str(type(shape)))

    c = operator.reshape(input, shape, name=name)

    return c
