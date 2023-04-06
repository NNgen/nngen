from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import functools

import nngen.operator as operator

from . import util
from . import basic
from . import batchnormalization
from . import conv
from . import gemm


def _act_func(method, visitor, node):

    node_name = util.get_name(node)

    src_name = node.input[0]
    src_node = util.search_node_from_model(visitor.model, src_name)

    if (not visitor.disable_fusion and
        src_node is not None and
        src_node.op_type == 'BatchNormalization' and
            len(visitor.consumers[src_name]) == 1):

        src_op = batchnormalization.BatchNormalization(visitor, src_node,
                                                       act_func=method)
        visitor.operators[node_name] = src_op
        return src_op

    if (not visitor.disable_fusion and
        src_node is not None and
            src_node.op_type == 'Conv' and len(visitor.consumers[src_name]) == 1):

        src_op = conv.Conv(visitor, src_node, act_func=method)
        visitor.operators[node_name] = src_op
        return src_op

    if (not visitor.disable_fusion and
        src_node is not None and
            src_node.op_type == 'Gemm' and len(visitor.consumers[src_name]) == 1):

        src_op = gemm.Gemm(visitor, src_node, act_func=method)
        visitor.operators[node_name] = src_op
        return src_op

    return basic._elementwise(method, visitor, node)


def Relu(visitor, node):
    return _act_func(operator.relu, visitor, node)


def LeakyRelu(visitor, node):
    alpha = 0.01
    for attribute in node.attribute:
        if attribute.name == 'alpha':
            alpha = attribute.f

    rshift = 31
    slope = round(alpha * (2 ** 31))
    op = operator.get_leaky_relu_op(slope, rshift)
    return _act_func(op, visitor, node)


def Sigmoid(visitor, node):
    return _act_func(operator.sigmoid, visitor, node)

def Tanh(visitor, node):
    return _act_func(operator.tanh, visitor, node)
