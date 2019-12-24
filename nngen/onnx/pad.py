from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import nngen.operator as operator

from . import util


def Pad(visitor, node):

    padding = [0, 0, 0, 0]  # Top, Bottom, Left, Right

    for attribute in node.attribute:
        if attribute.name == 'pads':
            all_pads_zero = True
            for pad in attribute.ints:
                if pad != 0:
                    all_pads_zero = False

            if all_pads_zero:
                node_name = util.get_name(node)
                src_name = node.input[0]
                src_op = visitor.visit(node.input[0])
                visitor.operators[node_name] = src_op
                return src_op

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    input = srcs[0]

    # transpose data layout to nngen-compatible format
    input = util.transpose_layout(input, visitor.nngen_input_layout, visitor.onnx_input_layout)

    name = util.get_name(node)

    if name in visitor.value_dtypes:
        dtype = visitor.value_dtypes[name]
    else:
        dtype = visitor.default_operator_dtype

    kwargs = collections.OrderedDict()
    kwargs['padding'] = tuple(padding)
    kwargs['dtype'] = dtype
    kwargs['name'] = name

    c = operator.pad(input, **kwargs)
    c.layout = visitor.nngen_input_layout
    c.onnx_layout = visitor.onnx_input_layout

    return c
