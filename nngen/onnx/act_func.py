from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import nngen.operator as operator

from . import util
from . import basic
from . import batchnormalization
from . import conv
from . import gemm


def _act_func_base(act_func, visitor, node):

    node_name = util.get_name(node)

    src_name = node.input[0]
    src_node = util.search_node_from_model(visitor.model, src_name)

    if (not visitor.disable_fusion and
        src_node.op_type == 'BatchNormalization' and
            len(visitor.consumers[src_name]) == 1):

        src_op = batchnormalization.BatchNormalization(visitor, src_node,
                                                       act_func=act_func)
        visitor.operators[node_name] = src_op
        return src_op

    if (not visitor.disable_fusion and
            src_node.op_type == 'Conv' and len(visitor.consumers[src_name]) == 1):

        src_op = conv.Conv(visitor, src_node, act_func=act_func)
        visitor.operators[node_name] = src_op
        return src_op

    if (not visitor.disable_fusion and
            src_node.op_type == 'Gemm' and len(visitor.consumers[src_name]) == 1):

        src_op = gemm.Gemm(visitor, src_node, act_func=act_func)
        visitor.operators[node_name] = src_op
        return src_op

    return basic._elementwise(act_func, visitor, node)


def Relu(visitor, node):
    return _act_func_base(operator.relu, visitor, node)


def LeakyRelu(visitor, node):
    alpha = 0.01
    for attribute in node.attribute:
        if attribute.name == 'alpha':
            alpha = attribute.f

    rshift = 31
    slope = round(alpha * (2 ** 31))
    op = operator.get_leaky_relu_op(slope, rshift)
    return _act_func_base(op, visitor, node)


def Sigmoid(visitor, node):
    return _sigmoid(operator.sigmoid, visitor, node)
