from __future__ import absolute_import
from __future__ import print_function

from pynq import Overlay, Xlnk, allocate
import numpy as np

import nngen_ctrl

# setting address according to memory map
OUTPUT_BASE = 0
INPUT_BASE = 64
PARAM_BASE = 8256
MEMORY_SIZE = 8679103
OUTPUT_CLASSES = 10

param_data = np.load("../param_data.npy")
input_data = np.load("../input_data.npy").astype("uint16")

# padding for 
input_data = np.append(input_data, np.zeros([1, input_data.shape[1], input_data.shape[2], 1]), axis = 3)
input_data = np.reshape(input_data, [-1])

def run_my_cnn(path, name, param_data, input_data):
    overlay = Overlay(path)
    ip = nngen_ctrl.nngen_ip(overlay, name)

    buf = allocate(shape = (MEMORY_SIZE, ), dtype = np.uint8)

    for i in range(len(param_data)):
        buf[PARAM_BASE+i] = param_data[i]

    for i in range(len(input_data)):
        buf[INPUT_BASE + 2 * i] = input_data[i]
        buf[INPUT_BASE + 2 * i + 1] = input_data[i] >> 8

    ip.set_global_offset(buf)
    
    ip.run()
    ip.wait()

    rslt = []
    for i in range(OUTPUT_CLASSES):
        x = buf[2 * i + 1] << 8 | buf[2 * i]
        rslt.append(x.astype("int16"))

    print("output:", rslt)

run_my_cnn('/home/xilinx/top.bit', 'my_cnn', param_data, input_data)
