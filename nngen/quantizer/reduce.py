from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


def argmax(visitor, node):

    input_tensor = node.args[0]
    visitor.visit(input_tensor)

    node.scale_factor = 1.0


def argmin(visitor, node):

    input_tensor = node.args[0]
    visitor.visit(input_tensor)

    node.scale_factor = 1.0
