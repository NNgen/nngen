from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng
import veriloggen

import matrix_conv2d_conv2d


act_shape = (1, 7, 7, 15)
weight1_shape = (7, 3, 3, 15)
bias1_shape = None
scale1_shape = None
weight2_shape = (9, 3, 3, 7)
bias2_shape = None
scale2_shape = None
act_dtype = ng.int32
weight1_dtype = ng.int32
bias1_dtype = ng.int32
scale1_dtype = ng.int32
weight2_dtype = ng.int32
bias2_dtype = ng.int32
scale2_dtype = ng.int32
tmp_dtype = ng.int32
out_dtype = ng.int32
stride1 = (1, 1, 1, 1)
stride2 = (1, 1, 1, 1)
rshift_mul1 = None
rshift_sum1 = None
rshift_out1 = None
rshift_mul2 = None
rshift_sum2 = None
rshift_out2 = None
act_func1 = ng.relu
act_func2 = ng.relu
par_ich1 = 1
par_och1 = 1
par_col1 = 1
par_row1 = 1
concur_och1 = None
stationary1 = 'filter'
input_ram_size1 = None
filter_ram_size1 = None
bias_ram_size1 = None
scale_ram_size1 = None
out_ram_size1 = None
par_ich2 = 1
par_och2 = 1
par_col2 = 1
par_row2 = 1
concur_och2 = None
stationary2 = 'filter'
input_ram_size2 = None
filter_ram_size2 = None
bias_ram_size2 = None
scale_ram_size2 = None
out_ram_size2 = None
axi_datawidth = 32


def test(request, silent=True):
    veriloggen.reset()

    simtype = request.config.getoption('--sim')

    rslt = matrix_conv2d_conv2d.run(act_shape,
                                    weight1_shape, bias1_shape, scale1_shape,
                                    weight2_shape, bias2_shape, scale2_shape,
                                    act_dtype,
                                    weight1_dtype, bias1_dtype, scale1_dtype,
                                    weight2_dtype, bias2_dtype, scale2_dtype,
                                    tmp_dtype,
                                    out_dtype,
                                    stride1, stride2,
                                    rshift_mul1, rshift_sum1, rshift_out1,
                                    rshift_mul2, rshift_sum2, rshift_out2,
                                    act_func1, act_func2,
                                    par_ich1, par_och1, par_col1, par_row1,
                                    concur_och1, stationary1,
                                    par_ich2, par_och2, par_col2, par_row2,
                                    concur_och2, stationary2,
                                    input_ram_size1, filter_ram_size1,
                                    bias_ram_size1, scale_ram_size1,
                                    out_ram_size1,
                                    input_ram_size2, filter_ram_size2,
                                    bias_ram_size2, scale_ram_size2,
                                    out_ram_size2,
                                    axi_datawidth, silent,
                                    filename=None, simtype=simtype,
                                    outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')

    verify_rslt = rslt.splitlines()[-1]
    assert(verify_rslt == '# verify: PASSED')


if __name__ == '__main__':
    rslt = matrix_conv2d_conv2d.run(act_shape,
                                    weight1_shape, bias1_shape, scale1_shape,
                                    weight2_shape, bias2_shape, scale2_shape,
                                    act_dtype,
                                    weight1_dtype, bias1_dtype, scale1_dtype,
                                    weight2_dtype, bias2_dtype, scale2_dtype,
                                    tmp_dtype,
                                    out_dtype,
                                    stride1, stride2,
                                    rshift_mul1, rshift_sum1, rshift_out1,
                                    rshift_mul2, rshift_sum2, rshift_out2,
                                    act_func1, act_func2,
                                    par_ich1, par_och1, par_col1, par_row1,
                                    concur_och1, stationary1,
                                    par_ich2, par_och2, par_col2, par_row2,
                                    concur_och2, stationary2,
                                    input_ram_size1, filter_ram_size1,
                                    bias_ram_size1, scale_ram_size1,
                                    out_ram_size1,
                                    input_ram_size2, filter_ram_size2,
                                    bias_ram_size2, scale_ram_size2,
                                    out_ram_size2,
                                    axi_datawidth, silent=False,
                                    filename='tmp.v',
                                    outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')
    print(rslt)
