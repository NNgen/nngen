from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng
import veriloggen

import matrix_conv2d


act_shape = (1, 3, 3, 33)
weight_shape = (65, 3, 1, 33)
bias_shape = None
scale_shape = None
act_dtype = ng.int16
weight_dtype = ng.int16
bias_dtype = ng.int16
scale_dtype = ng.int16
out_dtype = ng.int16
stride = (1, 1, 1, 1)

rshift_sum = None
rshift_out = None
act_func = None
par_ich = 1
par_och = 2
par_col = 1
par_row = 1
concur_och = 64
stationary = 'filter'
input_ram_size = None
filter_ram_size = None
bias_ram_size = None
scale_ram_size = None
out_ram_size = None
axi_datawidth = 32


def test(request, silent=True):
    veriloggen.reset()

    simtype = request.config.getoption('--sim')

    rslt = matrix_conv2d.run(act_shape, weight_shape,
                             bias_shape, scale_shape,
                             act_dtype, weight_dtype,
                             bias_dtype, scale_dtype,
                             out_dtype,
                             stride,
                             rshift_sum, rshift_out,
                             act_func,
                             par_ich, par_och, par_col, par_row,
                             concur_och, stationary,
                             input_ram_size, filter_ram_size,
                             bias_ram_size, scale_ram_size,
                             out_ram_size,
                             axi_datawidth, silent,
                             filename=None, simtype=simtype,
                             outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')

    verify_rslt = rslt.splitlines()[-1]
    assert(verify_rslt == '# verify: PASSED')


if __name__ == '__main__':
    rslt = matrix_conv2d.run(act_shape, weight_shape,
                             bias_shape, scale_shape,
                             act_dtype, weight_dtype,
                             bias_dtype, scale_dtype,
                             out_dtype,
                             stride,
                             rshift_sum, rshift_out,
                             act_func,
                             par_ich, par_och, par_col, par_row,
                             concur_och, stationary,
                             input_ram_size, filter_ram_size,
                             bias_ram_size, scale_ram_size,
                             out_ram_size,
                             axi_datawidth, silent=False,
                             filename='tmp.v',
                             outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')
    print(rslt)
