from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng
import veriloggen

import matrix_matmul_relu


a_shape = (15, 15)
b_shape = (15, 15)
bias_shape = None
scale_shape = None
a_dtype = ng.int32
b_dtype = ng.int32
bias_dtype = ng.int32
scale_dtype = ng.int32
c_dtype = ng.int32

rshift_sum = None
rshift_out = None
act_func = None
par_left_col = 1
par_left_row = 1
par_out_col = 1
concur_out_col = None
stationary = 'right'
left_ram_size = None
right_ram_size = None
bias_ram_size = None
scale_ram_size = None
out_ram_size = None
axi_datawidth = 32


def test(request, silent=True):
    veriloggen.reset()

    simtype = request.config.getoption('--sim')

    rslt = matrix_matmul_relu.run(a_shape, b_shape,
                                  bias_shape, scale_shape,
                                  a_dtype, b_dtype,
                                  bias_dtype, scale_dtype,
                                  c_dtype,
                                  rshift_sum, rshift_out,
                                  act_func,
                                  par_left_col, par_left_row, par_out_col,
                                  concur_out_col, stationary,
                                  left_ram_size, right_ram_size,
                                  bias_ram_size, scale_ram_size,
                                  out_ram_size,
                                  axi_datawidth, silent,
                                  filename=None, simtype=simtype,
                                  outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')

    verify_rslt = rslt.splitlines()[-1]
    assert(verify_rslt == '# verify: PASSED')


if __name__ == '__main__':
    rslt = matrix_matmul_relu.run(a_shape, b_shape,
                                  bias_shape, scale_shape,
                                  a_dtype, b_dtype,
                                  bias_dtype, scale_dtype,
                                  c_dtype,
                                  rshift_sum, rshift_out,
                                  act_func,
                                  par_left_col, par_left_row, par_out_col,
                                  concur_out_col, stationary,
                                  left_ram_size, right_ram_size,
                                  bias_ram_size, scale_ram_size,
                                  out_ram_size,
                                  axi_datawidth, silent=False,
                                  filename='tmp.v',
                                  outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')
    print(rslt)
