from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from . import util


def Gather(visitor, node):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    srcs = [util.optimize_to_raw_value(src) for src in srcs]

    input = srcs[0]
    indices = srcs[1]

    if not isinstance(input, (tuple, list, np.ndarray)):
        raise NotImplementedError("not supported input type: '%s'" % str(type(input)))

    if not isinstance(input, np.ndarray):
        input = np.array(input)

    if not isinstance(indices, (tuple, list, np.ndarray)):
        raise NotImplementedError("not supported indices type: '%s'" % str(type(indices)))

    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)

    for attribute in node.attribute:
        if attribute.name == 'axis':
            axis = attribute.i

    c = _gather(input, axis, indices)

    return c


def _gather(v, axis, indices):

    if axis > 0:
        ret = []
        for i in range(v.shape[0]):
            ret.append(_gather(v[i], axis - 1, indices))

        return np.array(ret)

    if indices.ndim == 0:
        return np.array(v[indices])

    if indices.ndim > 1:
        ret = []
        for i in range(indices.shape[0]):
            ret.append(_gather(v, axis, indices[i]))

        return np.array(ret)

    ret = []
    for index in indices:
        ret.append(v[index])

    return np.array(ret)
