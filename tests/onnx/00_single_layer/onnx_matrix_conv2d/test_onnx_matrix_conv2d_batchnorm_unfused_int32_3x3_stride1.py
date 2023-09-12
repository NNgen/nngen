from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng
import veriloggen

import onnx_matrix_conv2d


act_shape = (1, 7, 7, 3)
weight_shape = (9, 3, 3, 3)
act_dtype = ng.int32
weight_dtype = ng.int32
stride = 1
padding = 0
with_batchnorm = True
act_func = 'ReLU'
disable_fusion = True
par_ich = 1
par_och = 1
par_col = 1
par_row = 1
concur_och = None
stationary = 'filter'
chunk_size = 64
axi_datawidth = 32


def test(request, silent=True):
    veriloggen.reset()

    simtype = request.config.getoption('--sim')

    rslt = onnx_matrix_conv2d.run(act_shape, weight_shape,
                                  act_dtype, weight_dtype,
                                  stride, padding,
                                  with_batchnorm, act_func, disable_fusion,
                                  par_ich, par_och, par_col, par_row,
                                  concur_och, stationary,
                                  chunk_size,
                                  axi_datawidth, silent,
                                  filename=None, simtype=simtype,
                                  outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')

    verify_rslt = [line for line in rslt.splitlines() if line.startswith('# verify:')][0]
    assert(verify_rslt == '# verify: PASSED')


if __name__ == '__main__':
    rslt = onnx_matrix_conv2d.run(act_shape, weight_shape,
                                  act_dtype, weight_dtype,
                                  stride, padding,
                                  with_batchnorm, act_func, disable_fusion,
                                  par_ich, par_och, par_col, par_row,
                                  concur_och, stationary,
                                  chunk_size,
                                  axi_datawidth, silent=False,
                                  filename='tmp.v',
                                  outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')
    print(rslt)
