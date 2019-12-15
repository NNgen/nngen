from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng
import veriloggen

import onnx_vgg11


act_shape = (1, 32, 32, 3)
act_dtype = ng.int8
weight_dtype = ng.int8
bias_dtype = ng.int32
scale_dtype = ng.int8
with_batchnorm = True
disable_fusion = False
conv2d_par_ich = 1
conv2d_par_och = 1
conv2d_par_col = 1
conv2d_par_row = 1
conv2d_concur_och = None
conv2d_stationary = 'filter'
pool_par = 1
elem_par = 1
chunk_size = 64
axi_datawidth = 32


def test(request, silent=True):
    veriloggen.reset()

    simtype = request.config.getoption('--sim')

    rslt = onnx_vgg11.run(act_shape,
                          act_dtype, weight_dtype,
                          bias_dtype, scale_dtype,
                          with_batchnorm, disable_fusion,
                          conv2d_par_ich, conv2d_par_och, conv2d_par_col, conv2d_par_row,
                          conv2d_concur_och, conv2d_stationary,
                          pool_par, elem_par,
                          chunk_size,
                          axi_datawidth, silent,
                          filename=None, simtype=simtype,
                          outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')

    verify_rslt = rslt.splitlines()[-1]
    assert(verify_rslt == '# verify: PASSED')


if __name__ == '__main__':
    rslt = onnx_vgg11.run(act_shape,
                          act_dtype, weight_dtype,
                          bias_dtype, scale_dtype,
                          with_batchnorm, disable_fusion,
                          conv2d_par_ich, conv2d_par_och, conv2d_par_col, conv2d_par_row,
                          conv2d_concur_och, conv2d_stationary,
                          pool_par, elem_par,
                          chunk_size,
                          axi_datawidth, silent=False,
                          filename='tmp.v',
                          outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')
    print(rslt)
