from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import collections

import nngen.operator as operator

from . import util


def Pad(visitor, node):

    mode = 'constant'
    pads = []
    pad_value = 0.0

    all_pads_zero = True

    for attribute in node.attribute:
        if attribute.name == 'mode':
            mode = attribute.s.decode()

        if attribute.name == 'value':
            pad_value = attribute.f

        if attribute.name == 'pads':
            for pad in attribute.ints:
                pads.append(pad)
                if pad != 0:
                    all_pads_zero = False

    if all_pads_zero:
        node_name = util.get_name(node)
        src_name = node.input[0]
        src_op = visitor.visit(node.input[0])
        visitor.operators[node_name] = src_op
        return src_op

    if mode != 'constant':
        raise ValueError("not supported padding mode: '%s'" % mode)

    if pad_value != 0.0:
        raise ValueError("not supported padding value: %d" % pad_value)

    pad_value = round(pad_value)

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    input = srcs[0]

    if len(input.shape) != 4:
        raise ValueError("not supported shape: %s" % str(tuple(shape)))

    # transpose data layout to nngen-compatible format
    input = util.transpose_layout(input, visitor.nngen_input_layout, visitor.onnx_input_layout)

    name = util.get_name(node)

    pads_pre = pads[:len(pads) // 2]
    pads_post = pads[len(pads) // 2:]

    pads_pre = [pads_pre[visitor.onnx_input_layout.index(l)]
                for l in visitor.nngen_input_layout]
    pads_post = [pads_post[visitor.onnx_input_layout.index(l)]
                 for l in visitor.nngen_input_layout]

    # padding layout: Top, Bottom, Left, Right
    padding = [pads_pre[1], pads_post[1], pads_pre[2], pads_post[2]]

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
