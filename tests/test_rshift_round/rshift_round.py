from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np

if sys.version_info.major < 3:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng

def run(a_shape=(15, 15), b_shape=(15, 15), c_shape=(15, 15),
        a_dtype=ng.int32, b_dtype=ng.int32, c_dtype=ng.int32, d_dtype=ng.int32,
        par=1):

    # create target hardware
    a = ng.placeholder(a_dtype, shape=a_shape, name='a')
    b = ng.placeholder(b_dtype, shape=b_shape, name='b')
    c = ng.placeholder(c_dtype, shape=c_shape, name='c')
    x = ng.multiply(a, b, par=par, dtype=ng.int16)
    x = ng.rshift_round(x, c, dtype=ng.int16, par=par)
    d = ng.clip(x, dtype=d_dtype)

    # verification data
    va = np.array([-8,8,255], dtype=np.int64)
    vb = np.array([104], dtype=np.int64)
    vc = np.array([7], dtype=np.int64)


    eval_outs = ng.eval([d], a=va, b=vb, c=vc)
    print("eval_outs: " + str(eval_outs[0]))
    return eval_outs[0]


if __name__ == '__main__':
    rslt = run()
    print(rslt)