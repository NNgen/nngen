from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import functools


def leaky_relu(features, slope, rshift, dtype=None, name=None,
               features_dtype=None):

    if rshift is None:
        rshift = dtype.width if dtype is not None else 31

    features_point = 0 if features_dtype is None else features_dtype.point
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - features_point

    negs = (features * slope) >> rshift
    comp = features >= 0

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(np.where(comp, features, negs))

    return ret


def get_leaky_relu_op(slope, rshift=None, dtype=None):
    return functools.partial(leaky_relu,
                             slope=slope, rshift=rshift, dtype=dtype)
