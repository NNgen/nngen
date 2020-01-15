from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import collections
import numpy as np

import nngen.operator as operator

from . import util


def Upsample(visitor, node):

    mode = 'nearest'
    scale_value = 1.0

    for attribute in node.attribute:
        if attribute.name == 'mode':
            mode = attribute.s.decode()

        # deprecated attribute since Upsample-9
        if attribute.name == 'scale':
            scale_value = attribute.f

    if mode != 'nearest':
        raise ValueError("not supported upsampling mode: '%s'" % mode)

    if round(scale_value) != scale_value:
        raise ValueError("not supported upsampling factor: %f" % scale)

    scale_value = round(scale_value)

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    srcs = [util.optimize_to_raw_value(src) for src in srcs]

    input = srcs[0]

    if len(input.shape) != 4:
        raise ValueError("not supported shape: %s" % str(tuple(shape)))

    # transpose data layout to nngen-compatible format
    input = util.transpose_layout(input, visitor.nngen_input_layout, visitor.onnx_input_layout)

    name = util.get_name(node)

    if len(srcs) > 1:
        factors = srcs[1]
        if not isinstance(factors, (np.ndarray, np.float, np.int, float, int)):
            raise TypeError("Upsampling factor must be constant, not %s" % str(type(factors)))

        if not isinstance(factors, np.ndarray):
            factors = np.array(factors)

        if len(factors) == 4:
            factors = [int(round(factors[visitor.onnx_input_layout.index(l)]))
                       for l in visitor.nngen_input_layout]
        else:
            factors = np.array([int(round(factors.reshape([-1])[0]))] * 4)

    else:
        factors = np.array([scale_value] * 4)

    if name in visitor.value_dtypes:
        dtype = visitor.value_dtypes[name]
    else:
        dtype = visitor.default_operator_dtype

    kwargs = collections.OrderedDict()
    kwargs['factors'] = factors
    kwargs['dtype'] = dtype
    kwargs['name'] = name

    c = operator.upsampling2d(input, **kwargs)
    c.layout = visitor.nngen_input_layout
    c.onnx_layout = visitor.onnx_input_layout

    return c
