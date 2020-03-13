from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


def Shape(visitor, node):

    input = visitor.visit(node.input[0])

    shape = input.shape
    if (input.get_layout() is not None and input.get_onnx_layout() is not None and
            input.get_layout() != input.get_onnx_layout()):
        shape = [shape[input.get_layout().index(l)] for l in input.get_onnx_layout()]

    return tuple(shape)
