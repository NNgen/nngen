from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng
import veriloggen

import matrix_upsampling2d_header


act_shape = (1, 7, 7, 15)
act_dtype = ng.int32
out_dtype = ng.int32
factors = (1, 2, 2, 1)
par = 1
axi_datawidth = 32
header0 = 100
header1 = 200
header2 = 300
header3 = 400


def test(request, silent=True):
    veriloggen.reset()

    simtype = request.config.getoption('--sim')

    rslt = matrix_upsampling2d_header.run(act_shape,
                                          act_dtype, out_dtype,
                                          factors,
                                          par,
                                          axi_datawidth,
                                          header0, header1, header2, header3,
                                          silent,
                                          filename=None, simtype=simtype,
                                          outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')

    verify_rslt = [line for line in rslt.splitlines() if line.startswith('# verify:')][0]
    assert(verify_rslt == '# verify: PASSED')


if __name__ == '__main__':
    rslt = matrix_upsampling2d_header.run(act_shape,
                                          act_dtype, out_dtype,
                                          factors,
                                          par,
                                          axi_datawidth,
                                          header0, header1, header2, header3,
                                          silent,
                                          filename='tmp.v',
                                          outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')
    print(rslt)
