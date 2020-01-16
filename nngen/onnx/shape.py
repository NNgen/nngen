from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


def Shape(visitor, node):

    input = visitor.visit(node.input[0])

    shape = input.shape
    if (input.layout is not None and input.onnx_layout is not None and
            input.layout != input.onnx_layout):
        shape = [shape[input.layout.index(l)] for l in input.onnx_layout]

    return tuple(shape)
