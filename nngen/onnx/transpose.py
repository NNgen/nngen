from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import collections

import nngen.operator as operator

from . import util


def Transpose(visitor, node):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    for attribute in node.attribute:
        if attribute.name == 'perm':
            perm = [i for i in attribute.ints]

    input = srcs[0]

#    orig_layout = input.get_original_layout()
#    if orig_layout is not None:
#        perm = util.convert_transpose_perm(perm, visitor.onnx_input_layout,
#                                           visitor.nngen_input_layout)

    raise NotImplementedError()
    kwargs = collections.OrderedDict()

    name = util.get_name(node)
    kwargs['name'] = name

    return operator.transpose(input, perm, **kwargs)
