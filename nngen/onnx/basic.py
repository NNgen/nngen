from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import collections

import nngen.storage as storage
import nngen.operator as operator
import nngen.dtype_list as dtype_list

from . import util


def _elementwise(method, visitor, node):

    args = []

    for src in list(node.input):
        src_obj = visitor.visit(src)
        args.append(src_obj)

    kwargs = collections.OrderedDict()

    name = util.get_name(node)
    kwargs['name'] = name

    if name in visitor.value_dtypes:
        kwargs['dtype'] = visitor.value_dtypes[name]

    return method(*args, **kwargs)


def _normalize_elementwise(method, pre_methods, visitor, node):

    srcs = []
    all_placeholder = True

    for src in list(node.input):
        src_obj = visitor.visit(src)
        srcs.append(src_obj)
        if not isinstance(src_obj, storage.placeholder):
            all_placeholder = False

    # if all sources are placeholder, no scaling is required.
    if all_placeholder:
        return _elementwise(method, visitor, node)

    name = util.get_name(node)

    if pre_methods is None:
        pre_methods = [None for _ in srcs]

    if len(pre_methods) != len(srcs):
        raise ValueError('length mismatch: %d != %d' % (len(pre_methods), len(srcs)))

    args = []
    for i, (src, pre_method) in enumerate(zip(srcs, pre_methods)):
        if pre_method is None:
            args.append(src)
        else:
            args.append(pre_method(src))

    for src in srcs:
        scale = 1
        args.append(scale)

    shamt = 0
    args.append(shamt)

    return operator.scaled_add(*args)


def Add(visitor, node):

    return _normalize_elementwise(operator.add, None, visitor, node)


def Sub(node, visitor):

    pre_methods = (None, operator.neg)
    return _normalize_elementwise(operator.sub, pre_methods, visitor, node)


def Mul(visitor, node):

    return _elementwise(operator.multiply, visitor, node)


def Div(visitor, node):

    return _elementwise(operator.div, visitor, node)
