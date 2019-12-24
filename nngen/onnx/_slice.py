from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


def Slice(visitor, node):

    srcs = []

    for src in node.input:
        src_obj = visitor.visit(src)
        srcs.append(src_obj)

    raise NotImplementedError()

#    input = srcs[0]
#    starts = srcs[1]
#    ends = srcs[2]
#    axes = srcs[3]
#    steps = srcs[4]
