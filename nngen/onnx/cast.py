from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.basic_types as bt
import nngen.operator as operator

from . import util


def Cast(visitor, node):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    input = srcs[0]

    if not isinstance(input, bt._Numeric):
        return input

    name = util.get_name(node)

    if name in visitor.value_dtypes:
        dtype = visitor.value_dtypes[name]
    else:
        dtype = visitor.default_operator_dtype

    c = operator.cast(input, dtype)
    c.layout = input.layout
    c.onnx_layout = input.onnx_layout

    return c
