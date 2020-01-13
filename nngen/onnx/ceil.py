from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from . import identity


def Ceil(visitor, node):

    return identity.Identity(visitor, node)
