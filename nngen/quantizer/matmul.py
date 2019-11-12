from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


from . import conv2d


def matmul(visitor, node):

    return conv2d.conv2d(visitor, node)
