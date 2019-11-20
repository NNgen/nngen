from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng
import veriloggen

import onnx_matrix_avg_pool


act_shape = (1, 7, 7, 3)
act_dtype = ng.int32
ksize = 2
stride = 2
padding = 0
par = 1
chunk_size = 64
axi_datawidth = 32


def test(request, silent=True):
    veriloggen.reset()

    simtype = request.config.getoption('--sim')

    rslt = onnx_matrix_avg_pool.run(act_shape, act_dtype,
                                    ksize, stride, padding,
                                    par,
                                    chunk_size,
                                    axi_datawidth, silent,
                                    filename=None, simtype=simtype,
                                    outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')

    verify_rslt = rslt.splitlines()[-1]
    assert(verify_rslt == '# verify: PASSED')


if __name__ == '__main__':
    rslt = onnx_matrix_avg_pool.run(act_shape, act_dtype,
                                    ksize, stride, padding,
                                    par,
                                    chunk_size,
                                    axi_datawidth, silent=False,
                                    filename='tmp.v',
                                    outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')
    print(rslt)
