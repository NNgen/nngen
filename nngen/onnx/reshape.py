from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import functools
import collections

import nngen.operator as operator

from . import util


def Reshape(visitor, node, no_transpose=False):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    srcs = [util.optimize_to_raw_value(src) for src in srcs]

    input = srcs[0]
    out_shape = srcs[1]

    if not isinstance(out_shape, (tuple, list, np.ndarray)):
        raise TypeError('shape must be tuple, list, or np.ndarray, not %s' %
                        str(type(out_shape)))

    if isinstance(out_shape, np.ndarray):
        out_shape = out_shape.tolist()

    if isinstance(input, (tuple, list)):
        input = np.array(input)

    if isinstance(input, np.ndarray):
        c = np.reshape(input, out_shape)
        return c

    length = input.get_length()
    out_shape = util.infer_shape(out_shape, length)

    out_layout = None
    out_onnx_layout = None

    if input.get_onnx_layout() is not None and input.get_onnx_layout() is not None:
        layout = input.get_layout()
        onnx_layout = input.get_onnx_layout()
        onnx_shape = [input.shape[layout.index(ol)]
                      for ol in onnx_layout]
        out_onnx_layout = make_layout(onnx_shape, out_shape, onnx_layout)

        if no_transpose or input.get_layout() == input.get_onnx_layout():
            pass

        elif is_split_reshape(onnx_shape, out_shape, onnx_layout, out_onnx_layout):
            # keep the original input layout as possible
            layout_map = layout_matching(onnx_shape, out_shape, onnx_layout, out_onnx_layout)

            out_layout = []
            for l in layout:
                key = (l,)
                out_layout.extend(layout_map[key])

            out_layout = tuple(out_layout)
            out_shape = [out_shape[out_onnx_layout.index(ol)]
                         for ol in out_layout]

        else:
            # transpose the input tensor for applying Reshape to the correct layout
            out_layout = out_onnx_layout

            perm = [input.get_layout().index(l) for l in input.get_onnx_layout()]
            onnx_perm = [i for i, l in enumerate(input.get_onnx_layout())]

            input = operator.transpose(input, perm)
            input.transpose_onnx_perm = onnx_perm

            input.layout = input.get_onnx_layout()

    name = util.get_name(node)

    c = operator.reshape(input, out_shape, name=name)
    c.layout = out_layout
    c.onnx_layout = out_onnx_layout

    return c


def make_layout(src_shape, dst_shape, src_layout, prefix='X'):

    si = 0
    di = 0
    tmp = 0
    dst_layout = []

    while True:
        src_size = functools.reduce(lambda x, y: x * y, src_shape[:si], 1)
        dst_size = functools.reduce(lambda x, y: x * y, dst_shape[:di], 1)

        if si >= len(src_shape) and di >= len(dst_shape):
            break

        if src_size == dst_size:
            if src_shape[si] == dst_shape[di]:
                dst_layout.append(src_layout[si])
                si += 1
                di += 1
            else:
                si += 1

        elif src_size > dst_size:
            dst_layout.append('%s%d' % (prefix, tmp))
            tmp += 1
            di += 1

        elif src_size < dst_size:
            si += 1

    return tuple(dst_layout)


def layout_matching(src_shape, dst_shape, src_layout, dst_layout):

    si = 0
    di = 0
    key = []
    value = []
    matching = collections.OrderedDict()

    while True:
        src_size = functools.reduce(lambda x, y: x * y, src_shape[:si], 1)
        dst_size = functools.reduce(lambda x, y: x * y, dst_shape[:di], 1)

        if si >= len(src_shape) and di >= len(dst_shape):
            if len(key) > 0 and len(value) > 0:
                matching[tuple(key)] = tuple(value)
            break

        if src_size == dst_size:
            if len(key) > 0 and len(value) > 0:
                matching[tuple(key)] = tuple(value)
                key = []
                value = []

            key.append(src_layout[si])
            value.append(dst_layout[di])
            si += 1
            di += 1

        elif src_size > dst_size:
            value.append(dst_layout[di])
            di += 1

        elif src_size < dst_size:
            key.append(src_layout[si])
            si += 1

    return matching


def is_split_reshape(src_shape, dst_shape, src_layout, dst_layout):
    matching = layout_matching(src_shape, dst_shape, src_layout, dst_layout)

    for sl in src_layout:
        key = (sl,)
        if key not in matching:
            return False

    return True
