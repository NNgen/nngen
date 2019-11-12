from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.operator as operator

from . import util


def Reshape(visitor, node):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    input = srcs[0]
    shape = srcs[1]

    if isinstance(shape, np.ndarray):
        shape = shape.tolist()

    name = util.get_name(node)

    if not isinstance(shape, (tuple, list)):
        raise TypeError('shape must be tuple or list, not %s' % str(type(shape)))

    if isinstance(input, (tuple, list)):
        input = np.array(input)

    if isinstance(input, np.ndarray):
        c = np.reshape(input, shape)

    else:
        c = operator.reshape(input, shape, name=name)

    return c
