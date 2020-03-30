from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.operator as operator

from . import util
from . import reshape


def Flatten(visitor, node, no_transpose=False):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    input = srcs[0]

    for attribute in node.attribute:
        if attribute.name == 'axis':
            axis = attribute.i

    if (not no_transpose and
        input.get_layout() is not None and input.get_onnx_layout() is not None and
            input.get_layout() != input.get_onnx_layout()):

        perm = [input.get_layout().index(l) for l in input.get_onnx_layout()]
        onnx_perm = [i for i, l in enumerate(input.get_onnx_layout())]

        input = operator.transpose(input, perm)
        input.implicit = True
        input.transpose_onnx_perm = onnx_perm

        input.layout = input.get_onnx_layout()

    # no transpose is required below.
    layout = input.get_layout()
    onnx_layout = input.get_onnx_layout()

    input_shape = input.shape
    if layout is not None and onnx_layout is not None:
        input_shape = [input_shape[layout.index(ol)] for ol in onnx_layout]

    if axis == 0:
        shape = [1, -1]
    else:
        shape = [int(np.prod(input_shape[:axis])), -1]

    if layout is not None and onnx_layout is not None:
        length = input.get_length()
        shape = util.infer_shape(shape, length)

        out_onnx_layout = reshape.make_layout(input_shape, shape, onnx_layout)
        layout_map = reshape.layout_matching(input_shape, shape, onnx_layout, out_onnx_layout)
        layout_map = dict([(tuple(sorted(k)), v) for k, v in layout_map.items()])

        out_layout = []
        key_list = []
        for l in layout:
            key_list.append(l)
            key_list.sort()
            key = tuple(key_list)
            if key in layout_map:
                out_layout.extend(layout_map[key])
                key_list = []

        out_layout = tuple(out_layout)

        shape = [shape[out_onnx_layout.index(ol)]
                 for ol in out_layout]

    name = util.get_name(node)

    c = operator.reshape(input, shape, name=name)

    return c
