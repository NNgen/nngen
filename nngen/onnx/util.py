from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.basic_types as bt
import nngen.operator as operator
import nngen.storage as storage


def get_name(node):
    name = node.name
    if name == '':  # if isinstance(node, onnx.FOO.BAR):
        name = '_'.join([output for output in node.output])
    return name


def get_output_names(node):
    return [name for name in node.output]


def search_node_from_model(model, name):
    for node in model.graph.node:
        output_names = get_output_names(node)

        for output_name in output_names:
            if name == output_name:
                return node

    return None


def to_shape(node, value_shapes):
    shape = tuple([d.dim_value for d in node.type.tensor_type.shape.dim])
    if node.name in value_shapes:
        shape = value_shapes[node.name]
    return shape


def transpose_layout(value, expected_layout, onnx_layout):

    if value.get_layout() == expected_layout:
        return value

    if isinstance(value, bt._Storage) and value.get_layout() is None and not value.consumers:
        if len(expected_layout) != len(onnx_layout):
            raise ValueError('layout format size mismatch: %d != %d' %
                             (len(expected_layout), len(onnx_layout)))

        perm = tuple([onnx_layout.index(e) for e in expected_layout])
        new_shape = tuple([value.shape[p] for p in perm])

        value.shape = new_shape
        value.layout = expected_layout
        value.onnx_layout = onnx_layout
        value.perm = perm

        if value.value is not None:
            value.value = np.transpose(value.value, perm)

        return value

    current_layout = value.get_layout()

    if current_layout == expected_layout:
        return value

    if current_layout is None:
        current_layout = onnx_layout

    perm = tuple([current_layout.index(e) for e in expected_layout])

    value = operator.transpose(value, perm)
    value.layout = expected_layout
    value.onnx_layout = onnx_layout

    return value


def convert_transpose_perm(perm, src_layout, dst_layout):
    return tuple([dst_layout.index(src_layout[p]) for p in perm])


def optimize_to_raw_value(value):

    while True:
        if isinstance(value, bt._View) and value.dtype == value.args[0].dtype:
            value = value.args[0]
            continue

        if isinstance(value, bt._LazyReshape) and not value.args[0].shape:
            value = value.args[0]
            continue

        break

    if isinstance(value, storage.variable):
        value = value.value
        if not value.shape:
            value = np.reshape(value, (1,))
        return value

    return value


def infer_shape(shape, length):

    if not isinstance(shape, list):
        shape = list(shape)

    use_minus_one = False
    minus_one_index = 0
    all_mul = 1

    for i, s in enumerate(shape):
        if s is None or s == -1:
            use_minus_one = True
            minus_one_index = i
        elif s < 0:
            raise ValueError('not supported value for shape: %d' % s)
        else:
            all_mul *= s

    if use_minus_one:
        shape[minus_one_index] = length // all_mul

    shape = tuple(shape)
    return shape
