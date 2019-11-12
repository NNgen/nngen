from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def concat(values, axis, dtype=None, name=None):
    return np.concatenate(values, axis=axis)
