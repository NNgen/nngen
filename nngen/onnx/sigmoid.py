from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


def Sigmoid(visitor, node):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    raise NotImplementedError()
