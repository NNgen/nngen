from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def upsampling2d(value, factors,
                 dtype=None, name=None, par=1,
                 value_dtype=None):

    ret_shape = [s * f for s, f in zip(value.shape, factors)]
    ret = np.zeros(ret_shape)

    factor_col = factors[2]
    factor_row = factors[1]

    for bat in range(ret_shape[0]):
        for row in range(ret_shape[1]):
            for col in range(ret_shape[2]):
                ret[bat][row][col][:] = value[bat][row // factor_row][col // factor_col]

    return ret
