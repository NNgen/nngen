from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng
import veriloggen

import matrix_upsampling2d


act_shape = (1, 7, 7, 15)
act_dtype = ng.int16
out_dtype = ng.int16
factors = (1, 3, 2, 1)
par = 1
axi_datawidth = 32


def test(request, silent=True):
    veriloggen.reset()

    simtype = request.config.getoption('--sim')

    rslt = matrix_upsampling2d.run(act_shape,
                                   act_dtype, out_dtype,
                                   factors,
                                   par,
                                   axi_datawidth, silent,
                                   filename=None, simtype=simtype,
                                   outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')

    verify_rslt = rslt.splitlines()[-1]
    assert(verify_rslt == '# verify: PASSED')


if __name__ == '__main__':
    rslt = matrix_upsampling2d.run(act_shape,
                                   act_dtype, out_dtype,
                                   factors,
                                   par,
                                   axi_datawidth, silent=False,
                                   filename='tmp.v',
                                   outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')
    print(rslt)
