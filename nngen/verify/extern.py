from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def extern(values, opcode,
           shape=None, dtype=None, name=None,
           value_dtypes=None, func=None):

    if not isinstance(values, (tuple, list)):
        values = [values]

    if not isinstance(opcode, int):
        raise TypeError("opcode must be int, not '%s'." %
                        str(type(opcode)))

    if shape is None:
        shape = values[0].shape

    if dtype is None:
        dtype = values[0].dtype

    if func is None:
        return np.ones(shape, dtype=np.int64)

    return func(*values)
