from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.operator as operator
import nngen.dtype_list as dtype_list

from . import util
from . import conv
from . import gemm


def BatchNormalization(visitor, node, act_func=None):

    srcs = []
    for src in node.input[1:]:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    # expand the expression
    a_value = srcs[0].value
    b_value = srcs[1].value
    mean_value = srcs[2].value
    variance_value = srcs[3].value
    epsilon = node.attribute[0].f
    div = np.sqrt(variance_value + epsilon)

    scale_value = a_value / div
    bias_value = b_value - a_value * mean_value / div

    # input
    node_name = util.get_name(node)

    src_name = node.input[0]
    src_node = util.search_node_from_model(visitor.model, src_name)

    if (not visitor.disable_fusion and
            src_node.op_type == 'Conv' and len(visitor.consumers[src_name]) == 1):

        src_op = conv.Conv(visitor, src_node,
                           batchnorm_scale=scale_value,
                           batchnorm_bias=bias_value,
                           act_func=act_func)
        visitor.operators[node_name] = src_op
        return src_op

    if (not visitor.disable_fusion and
            src_node.op_type == 'Gemm' and len(visitor.consumers[src_name]) == 1):

        src_op = gemm.Gemm(visitor, src_node,
                           batchnorm_scale=scale_value,
                           batchnorm_bias=bias_value,
                           act_func=act_func)
        visitor.operators[node_name] = src_op
        return src_op

    if src_node.op_type == 'Unsqueeze' and len(visitor.consumers[src_name]) == 1:

        src_src_name = src_node.input[0]
        src_src_node = util.search_node_from_model(visitor.model, src_src_name)

        if (not visitor.disable_fusion and
                src_src_node.op_type == 'Gemm' and
                len(visitor.consumers[src_src_name]) == 1):

            src_src_op = gemm.Gemm(visitor, src_src_node,
                                   batchnorm_scale=scale_value,
                                   batchnorm_bias=bias_value,
                                   act_func=act_func)
            visitor.operators[node_name] = src_src_op
            return src_src_op

    input = visitor.visit(node.input[0])
    scale = srcs[0]
    bias = srcs[1]

    scale.value = scale_value
    bias.value = bias_value

    scale.dtype = visitor.default_scale_dtype
    bias.dtype = visitor.default_bias_dtype

    shamt_shape = (1,)
    shamt_dtype = dtype_list.dtype_int(scale.dtype.width, signed=True)
    shamt = operator.full_imm(shamt_shape, 0, shamt_dtype)

    out = operator.normalize(input, scale, bias, shamt)

    if act_func is not None:
        out = act_func(out)

    return out
