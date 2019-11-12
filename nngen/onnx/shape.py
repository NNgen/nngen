from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


def Shape(visitor, node):

    input = visitor.visit(node.input[0])
    return tuple(input.shape)
