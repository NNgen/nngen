from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def relu(features, dtype=None, name=None, par=1,
         features_dtype=None):

    features_point = 0 if features_dtype is None else features_dtype.point
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - features_point

    zeros = np.zeros_like(features, dtype=np.int64)
    comp = features >= 0

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(np.where(comp, features, zeros))

    return ret


def relu6(features, dtype=None, name=None, par=1,
          features_dtype=None):

    features_point = 0 if features_dtype is None else features_dtype.point
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - features_point

    zeros = np.zeros_like(features, dtype=np.int64)
    comp0 = features >= 0
    sixs = np.zeros_like(features, dtype=np.int64) + [6]
    comp6 = features > 6

    out_op = ((lambda x: x << out_shift) if out_shift >= 0 else
              (lambda x: x >> -out_shift))

    ret = out_op(np.where(comp0, np.where(comp6, sixs, features), zeros))

    return ret
