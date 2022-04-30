"""
   NNgen: A Fully-Customizable Hardware Synthesis Compiler for Deep Neural Network

   Copyright 2017, Shinya Takamaeda-Yamazaki and Contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import os

import nngen as ng


# --------------------
# (1) Represent a DNN model as a dataflow by NNgen operators
# --------------------

# data types
act_dtype = ng.int8
weight_dtype = ng.int8
bias_dtype = ng.int32
scale_dtype = ng.int8
batchsize = 1

# input
input_layer = ng.placeholder(dtype=act_dtype,
                             shape=(batchsize, 32, 32, 3),  # N, H, W, C
                             name='input_layer')

# layer 0: conv2d (with bias and scale (= batchnorm)), relu, max_pool
w0 = ng.variable(dtype=weight_dtype,
                 shape=(64, 3, 3, 3),  # Och, Ky, Kx, Ich
                 name='w0')
b0 = ng.variable(dtype=bias_dtype,
                 shape=(w0.shape[0],), name='b0')
s0 = ng.variable(dtype=scale_dtype,
                 shape=(w0.shape[0],), name='s0')

a0 = ng.conv2d(input_layer, w0,
               strides=(1, 1, 1, 1),
               bias=b0,
               scale=s0,
               act_func=ng.relu,
               dtype=act_dtype,
               sum_dtype=ng.int32)

a0p = ng.max_pool_serial(a0,
                         ksize=(1, 2, 2, 1),
                         strides=(1, 2, 2, 1))

# layer 1: conv2d, relu, reshape
w1 = ng.variable(weight_dtype,
                 shape=(64, 3, 3, a0.shape[-1]),
                 name='w1')
b1 = ng.variable(bias_dtype,
                 shape=(w1.shape[0],),
                 name='b1')
s1 = ng.variable(scale_dtype,
                 shape=(w1.shape[0],),
                 name='s1')

a1 = ng.conv2d(a0p, w1,
               strides=(1, 1, 1, 1),
               bias=b1,
               scale=s1,
               act_func=ng.relu,
               dtype=act_dtype,
               sum_dtype=ng.int32)

a1r = ng.reshape(a1, [batchsize, -1])

# layer 2: full-connection, relu
w2 = ng.variable(weight_dtype,
                 shape=(256, a1r.shape[-1]),
                 name='w2')
b2 = ng.variable(bias_dtype,
                 shape=(w2.shape[0],),
                 name='b2')
s2 = ng.variable(scale_dtype,
                 shape=(w2.shape[0],),
                 name='s2')

a2 = ng.matmul(a1r, w2,
               bias=b2,
               scale=s2,
               transposed_b=True,
               act_func=ng.relu,
               dtype=act_dtype,
               sum_dtype=ng.int32)

# layer 3: full-connection, relu
w3 = ng.variable(weight_dtype,
                 shape=(10, a2.shape[-1]),
                 name='w3')
b3 = ng.variable(bias_dtype,
                 shape=(w3.shape[0],),
                 name='b3')
s3 = ng.variable(scale_dtype,
                 shape=(w3.shape[0],),
                 name='s3')

# output
output_layer = ng.matmul(a2, w3,
                         bias=b3,
                         scale=s3,
                         transposed_b=True,
                         name='output_layer',
                         dtype=act_dtype,
                         sum_dtype=ng.int32)


# --------------------
# (2) Assign weights to the NNgen operators
# --------------------

# In this example, random floating-point values are assigned.
# In a real case, you should assign actual weight values
# obtianed by a training on DNN framework.

# If you don't you NNgen's quantizer, you can assign integer weights to each tensor.


import numpy as np

w0_value = np.random.normal(size=w0.length).reshape(w0.shape)
w0_value = np.clip(w0_value, -3.0, 3.0)
w0.set_value(w0_value)

b0_value = np.random.normal(size=b0.length).reshape(b0.shape)
b0_value = np.clip(b0_value, -3.0, 3.0)
b0.set_value(b0_value)

s0_value = np.ones(s0.shape)
s0.set_value(s0_value)

w1_value = np.random.normal(size=w1.length).reshape(w1.shape)
w1_value = np.clip(w1_value, -3.0, 3.0)
w1.set_value(w1_value)

b1_value = np.random.normal(size=b1.length).reshape(b1.shape)
b1_value = np.clip(b1_value, -3.0, 3.0)
b1.set_value(b1_value)

s1_value = np.ones(s1.shape)
s1.set_value(s1_value)

w2_value = np.random.normal(size=w2.length).reshape(w2.shape)
w2_value = np.clip(w2_value, -3.0, 3.0)
w2.set_value(w2_value)

b2_value = np.random.normal(size=b2.length).reshape(b2.shape)
b2_value = np.clip(b2_value, -3.0, 3.0)
b2.set_value(b2_value)

s2_value = np.ones(s2.shape)
s2.set_value(s2_value)

w3_value = np.random.normal(size=w3.length).reshape(w3.shape)
w3_value = np.clip(w3_value, -3.0, 3.0)
w3.set_value(w3_value)

b3_value = np.random.normal(size=b3.length).reshape(b3.shape)
b3_value = np.clip(b3_value, -3.0, 3.0)
b3.set_value(b3_value)

s3_value = np.ones(s3.shape)
s3.set_value(s3_value)

# Quantizing the floating-point weights by the NNgen quantizer.
# Alternatively, you can assign integer weights by yourself to each tensor.

imagenet_mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225]).astype(np.float32)

if act_dtype.width > 8:
    act_scale_factor = 128
else:
    act_scale_factor = int(round(2 ** (act_dtype.width - 1) * 0.5))

input_scale_factors = {'input_layer': act_scale_factor}
input_means = {'input_layer': imagenet_mean * act_scale_factor}
input_stds = {'input_layer': imagenet_std * act_scale_factor}

ng.quantize([output_layer], input_scale_factors, input_means, input_stds)


# --------------------
# (3) Assign hardware attributes
# --------------------

# conv2d, matmul
# par_ich: parallelism in input-channel
# par_och: parallelism in output-channel
# par_col: parallelism in pixel column
# par_row: parallelism in pixel row

par_ich = 2
par_och = 2

a0.attribute(par_ich=par_ich, par_och=par_och)
a1.attribute(par_ich=par_ich, par_och=par_och)
a2.attribute(par_ich=par_ich, par_och=par_och)
output_layer.attribute(par_ich=par_ich, par_och=par_och)

# cshamt_out: right shift amount after applying bias/scale
# If you assign integer weights by yourself to each tensor,
# cshamt (constant shift amount) must be assigned to each operator.

# a0.attribute(cshamt_out=weight_dtype.width + 1)
# a1.attribute(cshamt_out=weight_dtype.width + 1)
# a2.attribute(cshamt_out=weight_dtype.width + 1)
# output_layer.attribute(cshamt_out=weight_dtype.width + 1)

# max_pool
# par: parallelism in in/out channel

par = par_och

a0p.attribute(par=par)


# --------------------
# (4) Verify the DNN model behavior by executing the NNgen dataflow as a software
# --------------------

# In this example, random integer values are assigned.
# In real case, you should assign actual integer activation values, such as an image.

input_layer_value = np.random.normal(size=input_layer.length).reshape(input_layer.shape)
input_layer_value = input_layer_value * imagenet_std + imagenet_mean
input_layer_value = np.clip(input_layer_value, -3.0, 3.0)
input_layer_value = input_layer_value * act_scale_factor
input_layer_value = np.clip(input_layer_value,
                            -1 * 2 ** (act_dtype.width - 1) - 1, 2 ** (act_dtype.width - 1))
input_layer_value = np.round(input_layer_value).astype(np.int64)

eval_outs = ng.eval([output_layer], input_layer=input_layer_value)
output_layer_value = eval_outs[0]

print(output_layer_value)


# --------------------
# (5) Convert the NNgen dataflow to a hardware description (Verilog HDL and IP-XACT)
# --------------------

silent = False
axi_datawidth = 32

# to Veriloggen object
# targ = ng.to_veriloggen([output_layer], 'hello_nngen', silent=silent,
#                        config={'maxi_datawidth': axi_datawidth})

# to IP-XACT (the method returns Veriloggen object, as well as to_veriloggen)
targ = ng.to_ipxact([output_layer], 'hello_nngen', silent=silent,
                    config={'maxi_datawidth': axi_datawidth})
print('# IP-XACT was generated. Check the current directory.')

# to Verilog HDL RTL (the method returns a source code text)
# rtl = ng.to_verilog([output_layer], 'hello_nngen', silent=silent,
#                    config={'maxi_datawidth': axi_datawidth})


# --------------------
# (6) Save the quantized weights
# --------------------

# convert weight values to a memory image:
# on a real FPGA platform, this image will be used as a part of the model definition.

param_filename = 'hello_nngen.npz'
chunk_size = 64

param_data = ng.export_ndarray([output_layer], chunk_size)
np.savez_compressed(param_filename, param_data)


# --------------------
# (7) Simulate the generated hardware by Veriloggen and Verilog simulator
# --------------------

# If you don't check the RTL behavior, exit here.
# print('# Skipping RTL simulation. If you simulate the RTL behavior, comment out the next line.')
# sys.exit()

import math
from veriloggen import *
import veriloggen.thread as vthread
import veriloggen.types.axi as axi

outputfile = 'hello_nngen.out'
filename = 'hello_nngen.v'
# simtype = 'iverilog'
simtype = 'verilator'

param_bytes = len(param_data)

variable_addr = int(
    math.ceil((input_layer.addr + input_layer.memory_size) / chunk_size)) * chunk_size
check_addr = int(math.ceil((variable_addr + param_bytes) / chunk_size)) * chunk_size
tmp_addr = int(math.ceil((check_addr + output_layer.memory_size) / chunk_size)) * chunk_size

memimg_datawidth = 32
mem = np.zeros([1024 * 1024 * 256 // memimg_datawidth], dtype=np.int64)
mem = mem + [100]

# placeholder
axi.set_memory(mem, input_layer_value, memimg_datawidth,
               act_dtype.width, input_layer.addr,
               max(int(math.ceil(axi_datawidth / act_dtype.width)), par_ich))

# parameters (variable and constant)
axi.set_memory(mem, param_data, memimg_datawidth,
               8, variable_addr)

# verification data
axi.set_memory(mem, output_layer_value, memimg_datawidth,
               act_dtype.width, check_addr,
               max(int(math.ceil(axi_datawidth / act_dtype.width)), par_och))

# test controller
m = Module('test')
params = m.copy_params(targ)
ports = m.copy_sim_ports(targ)
clk = ports['CLK']
resetn = ports['RESETN']
rst = m.Wire('RST')
rst.assign(Not(resetn))

# AXI memory model
if outputfile is None:
    outputfile = os.path.splitext(os.path.basename(__file__))[0] + '.out'

memimg_name = 'memimg_' + outputfile

memory = axi.AxiMemoryModel(m, 'memory', clk, rst,
                            datawidth=axi_datawidth,
                            memimg=mem, memimg_name=memimg_name,
                            memimg_datawidth=memimg_datawidth)
memory.connect(ports, 'maxi')

# AXI-Slave controller
_saxi = vthread.AXIMLite(m, '_saxi', clk, rst, noio=True)
_saxi.connect(ports, 'saxi')

# timer
time_counter = m.Reg('time_counter', 32, initval=0)
seq = Seq(m, 'seq', clk, rst)
seq(
    time_counter.inc()
)


def ctrl():
    for i in range(100):
        pass

    ng.sim.set_global_addrs(_saxi, tmp_addr)

    start_time = time_counter.value
    ng.sim.start(_saxi)

    print('# start')

    ng.sim.wait(_saxi)
    end_time = time_counter.value

    print('# end')
    print('# execution cycles: %d' % (end_time - start_time))

    # verify
    ok = True
    for bat in range(output_layer.shape[0]):
        for x in range(output_layer.shape[1]):
            orig = memory.read_word(bat * output_layer.aligned_shape[1] + x,
                                    output_layer.addr, act_dtype.width)
            check = memory.read_word(bat * output_layer.aligned_shape[1] + x,
                                     check_addr, act_dtype.width)

            if vthread.verilog.NotEql(orig, check):
                print('NG (', bat, x,
                      ') orig: ', orig, ' check: ', check)
                ok = False
            else:
                print('OK (', bat, x,
                      ') orig: ', orig, ' check: ', check)

    if ok:
        print('# verify: PASSED')
    else:
        print('# verify: FAILED')

    vthread.finish()


th = vthread.Thread(m, 'th_ctrl', clk, rst, ctrl)
fsm = th.start()

uut = m.Instance(targ, 'uut',
                 params=m.connect_params(targ),
                 ports=m.connect_ports(targ))

# simulation.setup_waveform(m, uut)
simulation.setup_clock(m, clk, hperiod=5)
init = simulation.setup_reset(m, resetn, m.make_reset(), period=100, polarity='low')

init.add(
    Delay(10000000),
    Systask('finish'),
)

# output source code
if filename is not None:
    m.to_verilog(filename)

# run simulation
sim = simulation.Simulator(m, sim=simtype)
rslt = sim.run(outputfile=outputfile)

print(rslt)
