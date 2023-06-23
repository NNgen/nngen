from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng
import veriloggen

import onnx_matrix_linear


act_shape = (1, 15)
weight_shape = (13, 15)
bias_shape = None
scale_shape = None
act_dtype = ng.int32
weight_dtype = ng.int32
bias_dtype = ng.int32
scale_dtype = ng.int32
with_batchnorm = False
act_func = 'ReLU'
disable_fusion = False
par_left_col = 1
par_left_row = 1
par_out_col = 1
concur_out_col = None
stationary = 'right'
chunk_size = 64
axi_datawidth = 32


def test(request, silent=True):
    veriloggen.reset()

    simtype = request.config.getoption('--sim')

    rslt = onnx_matrix_linear.run(act_shape, weight_shape,
                                  bias_shape, scale_shape,
                                  act_dtype, weight_dtype,
                                  bias_dtype, scale_dtype,
                                  with_batchnorm, act_func, disable_fusion,
                                  par_left_col, par_left_row, par_out_col,
                                  concur_out_col, stationary,
                                  chunk_size,
                                  axi_datawidth, silent,
                                  filename=None, simtype=simtype,
                                  outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')

    verify_rslt = [line for line in rslt.splitlines() if line.startswith('# verify:')][0]
    assert(verify_rslt == '# verify: PASSED')


if __name__ == '__main__':
    rslt = onnx_matrix_linear.run(act_shape, weight_shape,
                                  bias_shape, scale_shape,
                                  act_dtype, weight_dtype,
                                  bias_dtype, scale_dtype,
                                  with_batchnorm, act_func, disable_fusion,
                                  par_left_col, par_left_row, par_out_col,
                                  concur_out_col, stationary,
                                  chunk_size,
                                  axi_datawidth, silent=False,
                                  filename='tmp.v',
                                  outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')
    print(rslt)
