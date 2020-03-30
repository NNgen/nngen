from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.operator as operator

from . import util


def Slice(visitor, node):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    srcs = [util.optimize_to_raw_value(src) for src in srcs]

    input = srcs[0]
    if isinstance(input, (tuple, list)):
        input = np.array(input)

    shape = input.shape

    starts = [0 for s in shape]
    ends = [s for s in shape]
    axes = [i for i in range(len(shape))]
    steps = [1 for s in shape]

    # for Slice-1 (Deprecated)
    for attribute in node.attribute:
        if attribute.name == 'starts':
            starts = [v for v in attribute.ints]

        elif attribute.name == 'ends':
            ends = [v for v in attribute.ints]

        elif attribute.name == 'axes':
            axes = [v for v in attribute.ints]

    if len(srcs) > 1:
        starts = srcs[1]
        ends = srcs[2]
        axes = srcs[3]
        steps = srcs[4]

    if isinstance(input, (tuple, list, np.ndarray)):
        input = np.array(input)
        v = get_sliced_value(input, starts, ends, axes, steps)
        return v

    layout = input.get_layout()
    onnx_layout = input.get_onnx_layout()

    if layout is not None and onnx_layout is not None:
        axes = [layout.index(onnx_layout[axis]) for axis in axes]
        axes.sort()

    starts, ends, steps = extract_slices(input, starts, ends, axes, steps)
    return operator.slice_(input, starts, ends, steps)


def get_sliced_value(input, starts, ends, axes, steps):

    slices = to_slices(input, starts, ends, axes, steps)
    return input[slices]


def to_slices(input, starts, ends, axes, steps):
    slices = []
    index = 0

    for start, end, axis, step in sorted(zip(starts, ends, axes, steps),
                                         key=lambda x: x[2]):

        while index < axis:
            slices.append(slice(0, input.shape[index]))
            index += 1

        slices.append(slice(start, end, step))
        index += 1

    return tuple(slices)


def extract_slices(input, starts, ends, axes, steps):
    ret_starts = []
    ret_ends = []
    ret_steps = []
    index = 0

    for start, end, axis, step in sorted(zip(starts, ends, axes, steps),
                                         key=lambda x: x[2]):

        while index < axis:
            ret_starts.append(0)
            ret_ends.append(input.shape[index])
            ret_steps.append(1)
            index += 1

        ret_starts.append(start)
        ret_ends.append(end)
        ret_steps.append(step)
        index += 1

    return tuple(ret_starts), tuple(ret_ends), tuple(ret_steps)
