from __future__ import absolute_import
from __future__ import print_function

from pynq import Overlay, Xlnk
import numpy as np

import nngen_ctrl


def run_my_cnn(path, name):
    overlay = Overlay(path)
    ip = nngen_ctrl.nngen_ip(overlay, name)

    xlnk = Xlnk()
    buf = xlnk.cma_array(16 * 1024, dtype=np.int32)
    for i in range(len(buf)):
        buf[i] = i

    ip.set_global_buffer(buf)
    ip.run()
    ip.wait()

    print(buf[:16])


run_my_cnn('/home/xilinx/top.bit', 'my_cnn')
