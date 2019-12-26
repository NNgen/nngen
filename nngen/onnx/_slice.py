from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from . import util


def Slice(visitor, node):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    input = srcs[0]
    starts = srcs[1]
    ends = srcs[2]
    axes = srcs[3]
    steps = srcs[4]

    if isinstance(input, (tuple, list, np.ndarray)):
        input = np.array(input)
        starts = util.optimize_to_raw_value(starts)
        ends = util.optimize_to_raw_value(ends)
        axes = util.optimize_to_raw_value(axes)
        steps = util.optimize_to_raw_value(steps)
        v = get_sliced_value(input, starts, ends, axes, steps)
        return v

    raise NotImplementedError("not supported input type: '%s'" % str(type(input)))


def get_sliced_value(input, starts, ends, axes, steps):
    slices = []
    index = 0

    for start, end, axis, step in sorted(zip(starts, ends, axes, steps),
                                         key=lambda x: x[2]):

        while index < axis:
            slices.append(slice(0, input.shape))
            index += 1

        slices.append(slice(start, end, step))
        index += 1

    return input[tuple(slices)]
