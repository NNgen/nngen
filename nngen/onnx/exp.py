from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.operator as operator

from . import basic


def Exp(visitor, node):

    return basic._elementwise(operator.exp, visitor, node, np.exp)
