from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import collections

import nngen.operator as operator

from . import util


def Transpose(visitor, node):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    for attribute in node.attribute:
        if attribute.name == 'perm':
            perm = [i for i in attribute.ints]

    input = srcs[0]

    layout = input.get_layout()
    onnx_layout = input.get_onnx_layout()

    transposed_layout = None
    transposed_onnx_layout = None

    if layout is not None and onnx_layout is not None:
        onnx_perm = perm
        perm = util.convert_transpose_perm(perm, onnx_layout, layout)

        transposed_layout = ''.join([layout[p] for p in perm])
        transposed_onnx_layout = ''.join([onnx_layout[p] for p in onnx_perm])

        if layout == transposed_layout:
            v = operator.cast(input, input.dtype)
            v.layout = input.layout
            v.onnx_layout = input.layout
            return v

    kwargs = collections.OrderedDict()

    name = util.get_name(node)
    kwargs['name'] = name

    c = operator.transpose(input, perm, **kwargs)
    c.transpose_onnx_perm = onnx_perm

    if layout is not None and onnx_layout is not None:
        c.layout = transposed_layout
        c.onnx_layout = transposed_onnx_layout

    return c
