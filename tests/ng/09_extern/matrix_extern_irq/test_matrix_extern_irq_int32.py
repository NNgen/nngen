from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng
import veriloggen

import matrix_extern_irq


#a_shape = (15, 15)
#b_shape = (15, 15)
a_shape = (1, 8)
b_shape = (1, 8)
a_dtype = ng.int32
b_dtype = ng.int32
c_dtype = ng.int32
par = 1
axi_datawidth = 32
interrupt_name = 'IRQ'


def test(request, silent=True):
    veriloggen.reset()

    simtype = request.config.getoption('--sim')

    rslt = matrix_extern_irq.run(a_shape, b_shape,
                                 a_dtype, b_dtype, c_dtype,
                                 par, axi_datawidth, interrupt_name, silent,
                                 filename=None, simtype=simtype,
                                 outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')

    verify_rslt = [line for line in rslt.splitlines() if line.startswith('# verify:')][0]
    assert(verify_rslt == '# verify: PASSED')


if __name__ == '__main__':
    rslt = matrix_extern_irq.run(a_shape, b_shape,
                                 a_dtype, b_dtype, c_dtype,
                                 par, axi_datawidth, interrupt_name, silent=False,
                                 filename='tmp.v',
                                 outputfile=os.path.splitext(os.path.basename(__file__))[0] + '.out')
    print(rslt)
