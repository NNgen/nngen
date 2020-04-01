from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import collections
import numpy as np

import nngen.storage as storage
import nngen.operator as operator

from . import util


def _reduce(method, visitor, node, np_method=None):

    args = []

    for src in node.input:
        src_obj = visitor.visit(src)
        args.append(src_obj)

    input = args[0]
    if isinstance(input, (tuple, list)):
        input = np.array(input)

    shape = input.shape

    keepdims = False
    axes = None

    for attribute in node.attribute:
        if attribute.name == 'axes':
            axes = [v for v in attribute.ints]

        elif attribute.name == 'keepdims':
            keepdims = attribute.i != 0

    if axes == [i for i in range(len(shape))]:
        axes = None

    opt_args = [util.optimize_to_raw_value(arg) for arg in args]

    all_ndarray = True
    for arg in opt_args:
        if not isinstance(arg, np.ndarray):
            all_ndarray = False
            break

    if all_ndarray and np_method is not None:
        return np_method(*opt_args, axis=axes, keepdims=keepdims)

    layout = input.get_layout()
    onnx_layout = input.get_onnx_layout()

    if layout is not None and onnx_layout is not None and axes is not None:
        axes = [layout.index(onnx_layout[axis]) for axis in axes]
        axes.sort()

    kwargs = collections.OrderedDict()

    name = util.get_name(node)
    kwargs['name'] = name

    kwargs['axis'] = axes
    kwargs['keep_dims'] = keepdims

    if name in visitor.value_dtypes:
        kwargs['dtype'] = visitor.value_dtypes[name]

    return method(*args, **kwargs)


def _arg(method, visitor, node, np_method=None):

    args = []

    for src in node.input:
        src_obj = visitor.visit(src)
        args.append(src_obj)

    input = args[0]
    if isinstance(input, (tuple, list)):
        input = np.array(input)

    keepdims = False
    axis = 0
    select_last_index = False

    for attribute in node.attribute:
        if attribute.name == 'axis':
            axis = attribute.i

        elif attribute.name == 'keepdims':
            keepdims = attribute.i != 0

        elif attribute.name == 'select_last_index':
            select_last_index = attribute.i != 0

    opt_args = [util.optimize_to_raw_value(arg) for arg in args]

    all_ndarray = True
    for arg in opt_args:
        if not isinstance(arg, np.ndarray):
            all_ndarray = False
            break

    if all_ndarray and np_method is not None:
        return np_method(*opt_args, axis=axes, keepdims=keepdims)

    if select_last_index:
        raise ValueError('select_last_index is not supported.')

    layout = input.get_layout()
    onnx_layout = input.get_onnx_layout()

    if layout is not None and onnx_layout is not None:
        axis = layout.index(onnx_layout[axis])

    kwargs = collections.OrderedDict()

    name = util.get_name(node)
    kwargs['name'] = name

    kwargs['axis'] = axis
    kwargs['keep_dims'] = keepdims

    if name in visitor.value_dtypes:
        kwargs['dtype'] = visitor.value_dtypes[name]

    return method(*args, **kwargs)


def ReduceSum(visitor, node):

    return _reduce(operator.reduce_sum, visitor, node, np.sum)


def ReduceMax(visitor, node):

    return _reduce(operator.reduce_max, visitor, node, np.max)


def ReduceMin(visitor, node):

    return _reduce(operator.reduce_min, visitor, node, np.min)


def ArgMax(visitor, node):

    return _arg(operator.argmax, visitor, node, np.argmax)


def ArgMin(visitor, node):

    return _arg(operator.argmin, visitor, node, np.argmin)
