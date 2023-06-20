from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import functools
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN


def leaky_relu(features, slope, rshift, dtype=None, name=None, par=1,
               features_dtype=None):

    if rshift is None:
        rshift = dtype.width if dtype is not None else 31

    features_point = 0 if features_dtype is None else features_dtype.point
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - features_point

    features_shape = features.shape
    negs = np.array(list(map(lambda x: Decimal(str(x * slope / (2**rshift))).quantize(Decimal('0'),
                                                                                      rounding=ROUND_HALF_UP), features.flatten()))).astype(np.int64).reshape(features_shape)

    comp = features >= 0

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(np.where(comp, features, negs))

    return ret


def get_leaky_relu_op(slope, rshift=None, dtype=None):
    return functools.partial(leaky_relu,
                             slope=slope, rshift=rshift, dtype=dtype)
