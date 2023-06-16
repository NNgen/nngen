from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng

import rshift_round


#a_shape = (1, 256, 10, 10)
a_shape = (1, 1)
b_shape = (1, 1)
c_shape = (1, 1)
a_dtype = ng.int8
b_dtype = ng.int8
c_dtype = ng.int8
d_dtype = ng.int8
par = 1

if __name__ == '__main__':
    rslt = rshift_round.run(a_shape, b_shape, c_shape,
                                           a_dtype, b_dtype, c_dtype, d_dtype,
                                           par)

    #print(rslt)