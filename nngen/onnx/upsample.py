from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import collections

import nngen.operator as operator

from . import util


def Upsample(visitor, node):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    input = srcs[0]

    # transpose data layout to nngen-compatible format
    input = util.transpose_layout(input, visitor.nngen_input_layout, visitor.onnx_input_layout)

    name = util.get_name(node)

    mode = 'nearest'
    scale = 1

    for attribute in node.attribute:
        if attribute.name == 'mode':
            mode = attribute.s

        # deprecated attribute since Upsample-9
        if attribute.name == 'scale':
            raise NotImplementedError()

    if len(srcs) > 1:
        scale = srcs[1]

    raise NotImplementedError()
