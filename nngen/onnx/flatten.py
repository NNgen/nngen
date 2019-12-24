from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.operator as operator

from . import util


def Flatten(visitor, node):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    input = srcs[0]

    for attribute in node.attribute:
        if attribute.name == 'axis':
            axis = attribute.i

    name = util.get_name(node)

    if axis == 0:
        shape = [1, -1]
    else:
        shape = [int(np.prod(input.shape[:axis])), -1]

    c = operator.reshape(input, shape, name=name)

    return c
