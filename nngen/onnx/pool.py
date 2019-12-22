from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import collections

import nngen.operator as operator
import nngen.dtype_list as dtype_list

from . import util


def _pool(pool_op, visitor, node, no_sum_dtype=False):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    input = srcs[0]

    # transpose data layout to nngen-compatible format
    input = util.transpose_layout(input, 'NHWC', visitor.onnx_input_layout)

    name = util.get_name(node)

    if name in visitor.value_dtypes:
        dtype = visitor.value_dtypes[name]
    else:
        dtype = visitor.default_operator_dtype

    sum_dtype = dtype_list.int32

    ksize, strides, padding = _get_ksize_strides_padding(node)

    kwargs = collections.OrderedDict()
    kwargs['ksize'] = ksize
    kwargs['strides'] = strides
    kwargs['padding'] = padding
    kwargs['dtype'] = dtype
    if not no_sum_dtype:
        kwargs['sum_dtype'] = sum_dtype
    kwargs['name'] = name

    c = pool_op(input, **kwargs)
    c.layout = 'NHWC'

    return c


def AveragePool(visitor, node):

    ksize, strides, padding = _get_ksize_strides_padding(node)

    if ksize == strides:
        op = operator.avg_pool_serial
    else:
        op = operator.avg_pool

    return _pool(op, visitor, node)


def GlobalAveragePool(visitor, node):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    input = srcs[0]

    # transpose data layout to nngen-compatible format
    input = util.transpose_layout(input, 'NHWC', visitor.onnx_input_layout)

    name = util.get_name(node)

    if name in visitor.value_dtypes:
        dtype = visitor.value_dtypes[name]
    else:
        dtype = visitor.default_operator_dtype

    sum_dtype = dtype_list.int32

    ksize = tuple([1, input.shape[1], input.shape[2], 1])
    strides = tuple([1, input.shape[1], input.shape[2], 1])
    padding = tuple([0, 0, 0, 0])

    kwargs = collections.OrderedDict()
    kwargs['ksize'] = ksize
    kwargs['strides'] = strides
    kwargs['padding'] = padding
    kwargs['dtype'] = dtype
    kwargs['name'] = name

    c = operator.avg_pool_serial(input, **kwargs)
    c.layout = 'NHWC'

    return c


def MaxPool(visitor, node):

    ksize, strides, padding = _get_ksize_strides_padding(node)

    if ksize == strides:
        op = operator.max_pool_serial
    else:
        op = operator.max_pool

    return _pool(op, visitor, node, no_sum_dtype=True)


def _get_ksize_strides_padding(node):

    ksize = [1, 1, 1, 1]  # B, H, W, C
    strides = [1, 1, 1, 1]  # B, H, W, C
    padding = [0, 0, 0, 0]  # Top, Bottom, Left, Right

    for attribute in node.attribute:
        if attribute.name == 'auto_pad':
            padding = 'SAME'

        elif attribute.name == 'pads':
            padding[0] = attribute.ints[0]
            padding[1] = attribute.ints[1]
            padding[2] = attribute.ints[2]
            padding[3] = attribute.ints[3]
            padding = tuple(padding)

        elif attribute.name == 'strides':
            strides[1] = attribute.ints[0]
            strides[2] = attribute.ints[1]
            strides = tuple(strides)

        elif attribute.name == 'kernel_shape':
            ksize[1] = attribute.ints[0]
            ksize[2] = attribute.ints[1]
            ksize = tuple(ksize)

    return ksize, strides, padding
