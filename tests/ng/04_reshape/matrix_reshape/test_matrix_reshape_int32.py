from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng
import veriloggen

import matrix_reshape


a_shape = (15, 15)
b_shape = (75, 3)
a_dtype = ng.int32
b_dtype = ng.int32
axi_datawidth = 32


def test(request, silent=True):
    veriloggen.reset()

    simtype = request.config.getoption('--sim')

    rslt = matrix_reshape.run(a_shape, b_shape,
                              a_dtype, b_dtype,
                              axi_datawidth, silent,
                              filename=None, simtype=simtype,
                              outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')

    verify_rslt = [line for line in rslt.splitlines() if line.startswith('# verify:')][0]
    assert(verify_rslt == '# verify: PASSED')


if __name__ == '__main__':
    rslt = matrix_reshape.run(a_shape, b_shape,
                              a_dtype, b_dtype,
                              axi_datawidth, silent=False,
                              filename='tmp.v',
                              outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')
    print(rslt)
