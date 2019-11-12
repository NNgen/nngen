from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.basic_types as bt
import nngen.storage as storage
import nngen.operator as operator


def get_name(node):
    name = node.name
    if name == '':  # if isinstance(node, onnx.FOO.BAR):
        name = '_'.join([output for output in node.output])
    return name


def search_node_from_model(model, name):
    for node in model.graph.node:
        node_name = get_name(node)

        if name == node_name:
            return node

    return None


def to_shape(node):
    return tuple([d.dim_value for d in node.type.tensor_type.shape.dim])


def transpose_layout(value, expected_layout, default_layout):

    if value.layout == expected_layout:
        return value

    if isinstance(value, bt._Storage) and value.layout is None and not value.consumers:
        if len(expected_layout) != len(default_layout):
            raise ValueError('layout format size mismatch: %d != %d' %
                             (len(expected_layout), len(default_layout)))

        perm = []
        new_shape = []
        for e in expected_layout:
            index = default_layout.find(e)
            perm.append(index)
            new_shape.append(value.shape[index])

        value.shape = tuple(new_shape)
        value.layout = expected_layout
        value.perm = tuple(perm)

        if value.value is not None:
            value.value = np.transpose(value.value, perm)

        return value

    #current_layout = value.layout if value.layout is not None else default_layout
    current_layout = get_layout(value)

    if current_layout == expected_layout:
        return value

    if current_layout is None:
        current_layout = default_layout

    perm = []
    for e in expected_layout:
        index = current_layout.find(e)
        perm.append(index)

    value = operator.transpose(value, perm)
    value.layout = expected_layout

    return value


def get_layout(value):
    if value.layout is not None:
        return value.layout

    if not isinstance(value, bt._Operator):
        return None

    if isinstance(value, operator.conv2d):
        return get_layout(value.args[0])

    if isinstance(value, (operator.normalize, operator.scaled_add)):
        return get_layout(value.args[0])

    if isinstance(value, bt._ElementwiseOperator):
        for arg in value.args:
            ret = get_layout(arg)
            if ret is not None:
                return ret

    raise ValueError('can not identify layout.')
