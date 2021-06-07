NNgen
==============================

A Fully-Customizable Hardware Synthesis Compiler for Deep Neural Network

Copyright 2017, Shinya Takamaeda-Yamazaki and Contributors


License
==============================

Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


What's NNgen?
==============================

NNgen is an open-sourced compiler to synthesize a model-specific hardware accelerator for deep neural networks. NNgen generates a Verilog HDL source code and an IP-core package (IP-XACT) of a DNN accelerator from an input model definition.

Generated hardware is all-inclusive, which includes processing engine, on-chip memory, on-chip network, DMA controller, and control circuits. So the generated hardware does not require any additional controls from an external circuit or the CPU after the processing is started.

The backend of NNgen employes Veriloggen, an open-sourced mixed-paradigm high-level synthesis compiler in Python. So you can customize NNgen for new DNN algorithms and applications.


Contribute to NNgen
==============================

NNgen project always welcomes questions, bug reports, feature proposals, and pull requests on [GitHub](https://github.com/NNgen/nngen).

Community manager
--------------------

As a manager of this project, the community manager leads community management, and promote software development and diffusion.

Committers
--------------------

Committers are individuals who are granted the write access to the project. In order to contribute as a committer, the approval of the community manager is required. The area of contribution can take all forms, including code contributions and code reviews, documents, education, and outreach. Committers are essential for a high quality and healthy project. The community actively looks for new committers from contributors.

Reviewers
--------------------

Reviewers are individuals who actively contributed to the project and are willing to participate in the code review of new contributions. We identify reviewers from active contributors. The committers should explicitly solicit reviews from reviewers. High-quality code reviews prevent technical debt for long-term and are crucial to the success of the project. A pull request to the project has to be reviewed by at least one reviewer in order to be merged.

for questions, bug reports, and feature proposals
--------------------

Please leave your comment on the [issue tracker](https://github.com/NNgen/nngen/issues) on GitHub.

for pull requests
--------------------

Please check "CONTRIBUTORS.md" for the contributors who provided pull requests.

NNgen uses **pytest** for the integration testing. **When you send a pull request, please include a testing example with pytest.**
To write a testing code, please refer the existing testing examples in "tests" directory.

If the pull request code passes all the tests successfully and has no obvious problem, it will be merged to the *develop* branch by the committers.


Installation
==============================

Requirements
--------------------

- Python3: 3.7 or later
- Icarus Verilog: 10.1 or later

```
sudo apt install iverilog
```

- Veriloggen: 1.9.0 or later
- NumPy: 1.17 or later
- ONNX: 1.6.0

```
pip3 install veriloggen numpy onnx
```

Install
--------------------

Now you can install NNgen using setup.py script.

```
python3 setup.py install
```

Optional requirements for testing
--------------------

These are required for automatic testing of **tests**.
We recommend to install these testing library to verify experimental features.

- pytest: 3.2 or later
- pytest-pythonpath: 0.7 or later
- PyTorch: 1.3.1 or later
- torchvision: 0.4.2 or later

```
pip3 install pytest pytest-pythonpath torch torchvision
```

For fast RTL simulation, we recommend to install Verilator.

- Verilator: 3.916 or later

```
sudo apt install verilator
```

Optional requirements for documentation
--------------------

If you want generate a document file from the source code, please install these dependen softwares.

- TeX Live: 2015 or later
- dvipng: 1.15 or later

```
sudo apt install texlive-science texlive-fonts-recommended texlive-fonts-extra dvipng
```

- Sphinx: 2.10 or later
- sphinx_rtd_theme : 0.4.3 or later

```
pip3 install sphinx sphinx_rtd_theme
```

Another installation way
--------------------

The current NNgen and Veriloggen are under the aggresive development.
Instead of the standard installation, you can download (or git clone) and install the latest version of NNgen, Veriloggen, and other libraries from GitHub.

### Download the latest NNgen, Veriloggen, and Pyverilog from GitHub

```
git clone https://github.com/NNgen/nngen.git
git clone https://github.com/PyHDI/veriloggen.git
git clone https://github.com/PyHDI/Pyverilog.git
```

### Create symbolic links to Veriloggen and the other libraries from NNgen

Instead of the actual installations, please create symbolic links to the dependent libraries.

```
cd nngen
ln -s ../veriloggen/veriloggen
ln -s ../Pyverilog/pyverilog
```

Docker
--------------------

Dockerfile is available. You can try NNgen on Docker without any installation on your host platform.

```
cd docker
sudo docker build -t user/nngen .
sudo docker run --name nngen -i -t user/nngen /bin/bash
cd nngen/examples/mlp/
make
```


Examples and testing
==============================

There are some exapmles in **examples** and various testing codes in **tests**.
The testing codes are actually good small examples suggesting how to represent a desired function.

To run the testing codes, please type the following commands.

```
cd tests
python3 -m pytest .
```

If you use Verilator instead of Icarus Verilog for RTL simulation, set "--sim" option.

```
python3 -m pytest --sim=verilator .
```


Getting started
==============================

Let's begin NNgen by an example.
For the complete example, see "hello_nngen.py".

(1) Represent a DNN model as a dataflow by NNgen operators
--------------------

In NNgen, a DNN model is defined by "define and run" manner.
You can build up a DNN model by chaining NNgen operators.

For the supported NNgen operator list, please see "nngen/operators/".

```python
from __future__ import absolute_import
from __future__ import print_function

import sys
import os

import nngen as ng


# data types
act_dtype = ng.int16
weight_dtype = ng.int16
bias_dtype = ng.int16
scale_dtype = ng.int16

# input
input_layer = ng.placeholder(dtype=act_dtype,
                             shape=(1, 32, 32, 3),  # N, H, W, C
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
               sum_dtype=ng.int64)

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
               sum_dtype=ng.int64)

a1r = ng.reshape(a1, [1, -1])

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
               sum_dtype=ng.int64)

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
                         sum_dtype=ng.int64)
```

### (Alternative) Import a existing model on a DNN framework via ONNX

Instead of such the explicit model construction, you can import an existing model via ONNX-importer.

For example, you can create your own model on Pytorch, or simply download a pre-defined model from Torchvision. Then you can translate the model into an ONNX file. Finally, the ONNX file can be imported as an NNgen model definition by "ng.from_onnx" method.

``` python
import torch
import torchvision

# model definition on Pytorch, or download a pre-defined model from torchvision
model = torchvision.models.resnet18(pretrained=True)

# Pytorch to ONNX
onnx_filename = 'resnet18_imagenet.onnx'
dummy_input = torch.randn(*act_shape).transpose(1, 3)
input_names = ['act']
output_names = ['out']
model.eval()
torch.onnx.export(model, dummy_input, onnx_filename,
                  input_names=input_names, output_names=output_names)

# ONNX to NNgen
dtypes = {}
(outputs, placeholders, variables,
 constants, operators) = ng.from_onnx(onnx_filename,
                                      value_dtypes=dtypes,
                                      default_placeholder_dtype=act_dtype,
                                      default_variable_dtype=weight_dtype,
                                      default_constant_dtype=weight_dtype,
                                      default_operator_dtype=act_dtype,
                                      default_scale_dtype=scale_dtype,
                                      default_bias_dtype=bias_dtype,
                                      disable_fusion=disable_fusion)
```

(2) Assign quantized weights to the NNgen operators
--------------------

Constructed NNgen operators contain no weight values. To verify the constructed NNgen dataflow as a software in an integer precision, weight values must be assigned to each ng.variable by "set_value" method.

In this example, random integer values are produced by NumPy, and are assigned. However, in real cases, actual integer weight values obtained by a DNN framework should be assigned.

``` python
import numpy as np

w0_value = np.random.normal(size=w0.length).reshape(w0.shape)
w0_value = np.clip(w0_value, -5.0, 5.0)
w0_value = w0_value * (2.0 ** (weight_dtype.width - 1) - 1) / 5.0
w0_value = np.round(w0_value).astype(np.int64)
w0.set_value(w0_value)

b0_value = np.random.normal(size=b0.length).reshape(b0.shape)
b0_value = np.clip(b0_value, -5.0, 5.0)
b0_value = b0_value * (2.0 ** (weight_dtype.width - 1) - 1) / 5.0 / 100.0
b0_value = np.round(b0_value).astype(np.int64)
b0.set_value(b0_value)

s0_value = np.ones(s0.shape, dtype=np.int64)
s0.set_value(s0_value)

w1_value = np.random.normal(size=w1.length).reshape(w1.shape)
w1_value = np.clip(w1_value, -5.0, 5.0)
w1_value = w1_value * (2.0 ** (weight_dtype.width - 1) - 1) / 5.0
w1_value = np.round(w1_value).astype(np.int64)
w1.set_value(w1_value)

b1_value = np.random.normal(size=b1.length).reshape(b1.shape)
b1_value = np.clip(b1_value, -5.0, 5.0)
b1_value = b1_value * (2.0 ** (weight_dtype.width - 1) - 1) / 5.0 / 100.0
b1_value = np.round(b1_value).astype(np.int64)
b1.set_value(b1_value)

s1_value = np.ones(s1.shape, dtype=np.int64)
s1.set_value(s1_value)

w2_value = np.random.normal(size=w2.length).reshape(w2.shape)
w2_value = np.clip(w2_value, -5.0, 5.0)
w2_value = w2_value * (2.0 ** (weight_dtype.width - 1) - 1) / 5.0
w2_value = np.round(w2_value).astype(np.int64)
w2.set_value(w2_value)

b2_value = np.random.normal(size=b2.length).reshape(b2.shape)
b2_value = np.clip(b2_value, -5.0, 5.0)
b2_value = b2_value * (2.0 ** (weight_dtype.width - 1) - 1) / 5.0 / 100.0
b2_value = np.round(b2_value).astype(np.int64)
b2.set_value(b2_value)

s2_value = np.ones(s2.shape, dtype=np.int64)
s2.set_value(s2_value)

w3_value = np.random.normal(size=w3.length).reshape(w3.shape)
w3_value = np.clip(w3_value, -5.0, 5.0)
w3_value = w3_value * (2.0 ** (weight_dtype.width - 1) - 1) / 5.0
w3_value = np.round(w3_value).astype(np.int64)
w3.set_value(w3_value)

b3_value = np.random.normal(size=b3.length).reshape(b3.shape)
b3_value = np.clip(b3_value, -5.0, 5.0)
b3_value = b3_value * (2.0 ** (weight_dtype.width - 1) - 1) / 5.0 / 100.0
b3_value = np.round(b3_value).astype(np.int64)
b3.set_value(b3_value)

s3_value = np.ones(s3.shape, dtype=np.int64)
s3.set_value(s3_value)
```

### (Alternative) Assign quantized parameters from floating-point parameters using Quantizer

If you import an existing model via ONNX, each variable has "float" weight parameters, not integer. Software-based verification and generated hardware of NNgen do not support such floating-point representation. Therefore, such floating-point parameters must be translated into integer.

NNgen provides a simple (but experimental) quantizer that converts floating-point parameters into integer ones. The quantizer automatically determines scaling factors for all operators, which are magnitudes (differences) compared to original floating-point based computations. Based on the scaling factors, the quantizer assigns the amount of right-shift operation at the end of each operator, to avoid overflows.

You can use quantizer even if you assign "float" parameters to variables by "set_value" method. Note that it is still experimental implementation. If you have an own better quantizer, please use it.

input_scale_factors is required to calculate right-shift amounts from input numerical ranges.
The quantizer assumes the input of every layer has a uniform distribution. For a better quantization, distribution parameters (input_means and input_stds) should be assigned.

```python
if act_dtype.width > 8:
    act_scale_factor = 128
else:
    act_scale_factor = int(round(2 ** (act_dtype.width - 1) * 0.5))

input_scale_factors = {'act': act_scale_factor}
input_means = {'act': imagenet_mean * act_scale_factor}
input_stds = {'act': imagenet_std * act_scale_factor}

ng.quantize(outputs, input_scale_factors, input_means, input_stds)
```

For more information about the quantizer, please see torchvision_onnx_resnet18 and torchvision_onnx_vgg11 in examples.
They generate an accelerator from a pre-trained model which is available from torchvision.

(3) Assign hardware attributes
--------------------

The default hardware organization is not properly parallelized. According to a performance requirement and resource constraints, parallelism in various directions can be configured via "attribute" method of each operator.

NNgen hardware executes a DNN model in integer precision. Thus, right-shift operations are inserted to the tail of (almost) each operator. The amount of right-shift (shamt) also can be assigned via "attribute" method.

``` python
# conv2d, matmul
# par_ich: parallelism in input-channel
# par_och: parallelism in output-channel
# par_col: parallelism in pixel column
# par_row: parallelism in pixel row
# cshamt_out: right shift amount after applying bias/scale

par_ich = 2
par_och = 2
cshamt_out = weight_dtype.width + 1

a0.attribute(par_ich=par_ich, par_och=par_och,
             cshamt_out=weight_dtype.width + 1)
a1.attribute(par_ich=par_ich, par_och=par_och,
             cshamt_out=weight_dtype.width + 1)
a2.attribute(par_ich=par_ich, par_och=par_och,
             cshamt_out=weight_dtype.width + 1)
output_layer.attribute(par_ich=par_ich, par_och=par_och,
                       cshamt_out=weight_dtype.width + 1)

# max_pool
# par: parallelism in in/out channel

par = par_och

a0p.attribute(par=par)
```

(4) Verify the DNN model behavior by executing the NNgen dataflow as a software
--------------------

After weight values are assigned, the constructed NNgen dataflow can be executed as a software to verify a quantized DNN model. "ng.eval" method evaluates the NNgen dataflow according to input values passed via method arguments.

In this example, random integer values are produced by NumPy, and are assigned as an input. However, actual integer input values, such as image data opened by PIL, should be assigned.

``` python
input_layer_value = np.random.normal(size=input_layer.length).reshape(input_layer.shape)
input_layer_value = np.clip(input_layer_value, -5.0, 5.0)
input_layer_value = input_layer_value * (2.0 ** (input_layer.dtype.width - 1) - 1) / 5.0
input_layer_value = np.round(input_layer_value).astype(np.int64)

eval_outs = ng.eval([output_layer], input_layer=input_layer_value)
output_layer_value = eval_outs[0]

print(output_layer_value)
```

(5) Convert the NNgen dataflow to a hardware description (Verilog HDL and IP-XACT)
--------------------

After all the weights are assigned and the hardware attributes are configured, the NNgen dataflow is ready to be converted to an actual hardware description.

You can specify the hardware parameters, such as a data width of the AXI interface and system-wide signal names, via the "config" argument. Please see "nngen/verilog.py" for all the list of configurable hardware parameters.

NNgen generates an all-inclusive dedicated hardware design for an input DNN model, which includes parallel processing elements, on-chip memories, on-chip network between the processing elements and the on-chip memories, a DMA controller between off-chip memories and on-chip memories, and FSM-based control circuits. Therefore, no external control, such as DMA on CPU is required after the generated hardware begins a computation.

NNgen supports 3 types of output: 1) Veriloggen object, which is Python-based high-level hardware abstraction, 2) IP-XACT, which is a common IP-core format, and 3) Verilog HDL RTL as a text file.
A generated Veriloggen object can be easily verified by a testing mechanism of Veriloggen and a Verilog simulator.
A generated IP-XACT IP-core can be integrated with other components via AMBA AXI4 interface on an FPGA.

All weight parameters are zipped into a single np.ndarray by "ng.export_ndarray" method. This array will be utilized in actual FPGA platform later. So please save it using "np.save" method as a binary file.

``` python
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

# to memory image:
# on a real FPGA platform, this image will be used as a part of the model definition.
param_filename = 'hello_nngen.npy'
chunk_size = 64

param_data = ng.export_ndarray([output_layer], chunk_size)
np.save(param_filename, param_data)

# If you don't check the RTL behavior, exit here.
# print('# Skipping RTL simulation. If you simulate the RTL behavior, comment out the next line.')
# sys.exit()
```

(6) Simulate the generated hardware by Veriloggen and Verilog simulator
--------------------

If you want to reduce the development time, you can skip this section for Verilog simulation.

If you generate a hardware as Veriloggen object or IP-XACT, you can simulate the hardware behavior on Verilog simulator via the testing mechanism on Veriloggen.

Before the hardware runs, the input data and weight values should be located on the shared off-chip memory. In Verilog simulation in the example, there is a np.ndarray object to represent a dump image of the off-chip memory. You can copy the pre-computed values to the memory image by "axi.set_memory" method.

"param_data" is the unified parameter data of all variables and constants. Locations of the located data are configurable, which can be changed from the CPU via the configuration register of the NNgen hardware. In the following example, the head address of unified parameter data (variblae_addr) is calculated by the same rule as the address calculator in the NNgen compiler.

"ctrl" method in the following example is an emulation of a control program on the CPU, which is actually an FSM circuit of the control sequence synthesized by the procedural high-level synthesis compiler of Veriloggen. By "ng.sim.start" method, the program writes '1' to the "start" register of the NNgen hardware. Then the hardware begins the computation, and the CPU waits until the computation finishes by "ng.sim.wait" method.

### Data alignment, and "word_alignment" and "aligned_shape"

**Note that all the input, weight, and output data should be located along with their alignments.** Especially, using a narrower data width (for any data) than the AXI interconnect interface and applying the parallelization via the hardware attribute will require special cares of data arrangement. In a synthesis log, you can find the **word_alignment** and **aligned_shape** for each placeholder, variable, operator. When putting corresponding data on an off-chip memory, a padding will be required according to the word alignment. The difference between the original shape and the aligned shape is the size of padding. In NNgen, padding is required only at an inner-most dimension.

Unified variable images, such as "param_data", are already aligned according to the word alignment. So you don't have to rearrange the data alignment.

``` python
import math
from veriloggen import *
import veriloggen.thread as vthread
import veriloggen.types.axi as axi

chunk_size = 64
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
```

Let's run the example.

``` sh
python3 hello_nngen.py
```

You will see a compilation result like the following.

```
[[-10533  13055  -7565  -5662  -4482    350  -7702   5641   3247   5189]]
NNgen: Neural Network Accelerator Generator (version 1.0)
[IP-XACT]
  Output: hello_nngen
[Configuration]
(AXI Master Interface)
  Data width   : 32
  Address width: 32
(AXI Slave Interface)
  Data width   : 32
  Address width: 32
[Schedule Table]
(Stage 0)
(Stage 1)
  <conv2d None dtype:int16 shape:(1, 32, 32, 64) strides:(1, 1, 1, 1) padding:'SAME'-(1, 1, 1, 1) bias:(64,) scale:(64,) cshamt_out:17 act_func:relu sum_dtype:int64 par_ich:2 par_och:2 concur_och:4 stationary:filter keep_input default_addr:8481984 g_index:0 l_index:1 word_alignment:2 aligned_shape:(1, 32, 32, 64) scale_factor:1.000000>
  | <placeholder input_layer dtype:int16 shape:(1, 32, 32, 3) default_addr:64 g_index:2 word_alignment:2 aligned_shape:(1, 32, 32, 4) scale_factor:1.000000>
  | <variable w0 dtype:int16 shape:(64, 3, 3, 3) default_addr:8256 g_index:3 word_alignment:2 aligned_shape:(64, 3, 3, 4) scale_factor:1.000000>
  | <variable b0 dtype:int16 shape:(64,) default_addr:8256 g_index:3 word_alignment:2 aligned_shape:(64,) scale_factor:1.000000>
  | <variable s0 dtype:int16 shape:(64,) default_addr:8256 g_index:3 word_alignment:2 aligned_shape:(64,) scale_factor:1.000000>
(Stage 2)
  <max_pool_serial None dtype:int16 shape:(1, 16, 16, 64) ksize:(1, 2, 2, 1) strides:(1, 2, 2, 1) padding:'SAME'-(0, 0, 0, 0) par:2 no_reuse default_addr:8613056 g_index:0 l_index:2 word_alignment:2 aligned_shape:(1, 16, 16, 64) scale_factor:1.000000>
  | <conv2d None dtype:int16 shape:(1, 32, 32, 64) strides:(1, 1, 1, 1) padding:'SAME'-(1, 1, 1, 1) bias:(64,) scale:(64,) cshamt_out:17 act_func:relu sum_dtype:int64 par_ich:2 par_och:2 concur_och:4 stationary:filter keep_input default_addr:8481984 g_index:0 l_index:1 word_alignment:2 aligned_shape:(1, 32, 32, 64) scale_factor:1.000000>
(Stage 3)
  <conv2d None dtype:int16 shape:(1, 16, 16, 64) strides:(1, 1, 1, 1) padding:'SAME'-(1, 1, 1, 1) bias:(64,) scale:(64,) cshamt_out:17 act_func:relu sum_dtype:int64 par_ich:2 par_och:2 concur_och:4 stationary:filter default_addr:8645824 g_index:0 l_index:3 word_alignment:2 aligned_shape:(1, 16, 16, 64) scale_factor:1.000000>
  | <max_pool_serial None dtype:int16 shape:(1, 16, 16, 64) ksize:(1, 2, 2, 1) strides:(1, 2, 2, 1) padding:'SAME'-(0, 0, 0, 0) par:2 no_reuse default_addr:8613056 g_index:0 l_index:2 word_alignment:2 aligned_shape:(1, 16, 16, 64) scale_factor:1.000000>
  | <variable w1 dtype:int16 shape:(64, 3, 3, 64) default_addr:8256 g_index:3 word_alignment:2 aligned_shape:(64, 3, 3, 64) scale_factor:1.000000>
  | <variable b1 dtype:int16 shape:(64,) default_addr:8256 g_index:3 word_alignment:2 aligned_shape:(64,) scale_factor:1.000000>
  | <variable s1 dtype:int16 shape:(64,) default_addr:8256 g_index:3 word_alignment:2 aligned_shape:(64,) scale_factor:1.000000>
(Stage 4)
  <_lazy_reshape None dtype:int16 shape:(1, 16384) alias_of:<conv2d> default_addr:8645824 g_index:0 l_index:3 word_alignment:2 aligned_shape:(1, 16384) scale_factor:1.000000>
  | <conv2d None dtype:int16 shape:(1, 16, 16, 64) strides:(1, 1, 1, 1) padding:'SAME'-(1, 1, 1, 1) bias:(64,) scale:(64,) cshamt_out:17 act_func:relu sum_dtype:int64 par_ich:2 par_och:2 concur_och:4 stationary:filter default_addr:8645824 g_index:0 l_index:3 word_alignment:2 aligned_shape:(1, 16, 16, 64) scale_factor:1.000000>
(Stage 5)
  <matmul None dtype:int16 shape:(1, 256) bias:(256,) scale:(256,) cshamt_out:17 act_func:relu sum_dtype:int64 par_left_col:2 par_out_col:2 concur_out_col:2 stationary:right keep_left default_addr:8678592 g_index:0 l_index:4 word_alignment:2 aligned_shape:(1, 256) scale_factor:1.000000>
  | <_lazy_reshape None dtype:int16 shape:(1, 16384) alias_of:<conv2d> default_addr:8645824 g_index:0 l_index:3 word_alignment:2 aligned_shape:(1, 16384) scale_factor:1.000000>
  | <variable w2 dtype:int16 shape:(256, 16384) default_addr:8256 g_index:3 word_alignment:2 aligned_shape:(256, 16384) scale_factor:1.000000>
  | <variable b2 dtype:int16 shape:(256,) default_addr:8256 g_index:3 word_alignment:2 aligned_shape:(256,) scale_factor:1.000000>
  | <variable s2 dtype:int16 shape:(256,) default_addr:8256 g_index:3 word_alignment:2 aligned_shape:(256,) scale_factor:1.000000>
(Stage 6)
  <matmul output_layer dtype:int16 shape:(1, 10) bias:(10,) scale:(10,) cshamt_out:17 sum_dtype:int64 par_left_col:2 par_out_col:2 concur_out_col:128 stationary:right keep_left keep_right default_addr:0 g_index:1 word_alignment:2 aligned_shape:(1, 10) scale_factor:1.000000>
  | <matmul None dtype:int16 shape:(1, 256) bias:(256,) scale:(256,) cshamt_out:17 act_func:relu sum_dtype:int64 par_left_col:2 par_out_col:2 concur_out_col:2 stationary:right keep_left default_addr:8678592 g_index:0 l_index:4 word_alignment:2 aligned_shape:(1, 256) scale_factor:1.000000>
  | <variable w3 dtype:int16 shape:(10, 256) default_addr:8256 g_index:3 word_alignment:2 aligned_shape:(10, 256) scale_factor:1.000000>
  | <variable b3 dtype:int16 shape:(10,) default_addr:8256 g_index:3 word_alignment:2 aligned_shape:(10,) scale_factor:1.000000>
  | <variable s3 dtype:int16 shape:(10,) default_addr:8256 g_index:3 word_alignment:2 aligned_shape:(10,) scale_factor:1.000000>
[RAM (spec: num)]
  32-bit 16384-entry 2-port 1-bank RAM: 2
  32-bit 8192-entry 2-port 1-bank RAM: 1
  32-bit 512-entry 2-port 1-bank RAM: 9
  32-bit 256-entry 2-port 1-bank RAM: 2
  32-bit 128-entry 2-port 1-bank RAM: 22
[Substream (spec: num)]
  ('acc_rshift_round_frac', (64, 0, True, 64, 0, True)): 2
  ('add_tree', (64, 0, True, 2)): 2
  ('add_tree', (64, 0, True, 18)): 2
  ('mul_rshift_clip', (64, 0, True, 16, 0, True, 80, 0, True, 16, 0, True)): 2
  ('mul_rshift_round_madd', (16, 0, True, 16, 0, True, 32, 0, True)): 36
  ('reduce_max', (16, 0, True)): 2
[Stream (spec: num)]
  (((<class 'nngen.operator.conv2d.conv2d'>, <dtype int16>, <dtype int16>, <dtype int16>, <dtype int16>), <dtype int16>, 1), 3, 3, None, <dtype int64>, 2, 2, 1, 1, 9, 36): 1
  (((<class 'nngen.operator.pool_serial.max_pool_serial'>, <dtype int16>), <dtype int16>, 2), 2, 2, True, 2): 1
  (((<class 'nngen.operator.basic._lazy_reshape'>, <dtype int16>), <dtype int16>, 1), True): 1
  (((<class 'nngen.operator.matmul.matmul'>, <dtype int16>, <dtype int16>, <dtype int16>, <dtype int16>), <dtype int16>, 1), 1, 1, None, <dtype int64>, 2, 2, 1, 1, 1, 4): 1
[Control (name (# states: num))]
  main_fsm (# states: 58)
  control_conv2d_4 (# states: 56)
  control_max_pool_serial_5 (# states: 26)
  control_matmul_14 (# states: 41)
[Register Map]
   0 (O): header0 (default: 0)
   4 (O): header1 (default: 0)
   8 (O): header2 (default: 0)
  12 (O): header3 (default: 0)
  16 (I): Start (set '1' to run)
  20 (O): Busy (returns '1' when running)
  24 (I): Reset (set '1' to initialize internal logic)
  28 (O): Opcode from extern objects to SW (returns '0' when idle)
  32 (I): Resume extern objects (set '1' to resume)
  36 (I): Global address offset (default: 0)
  40 (I): Address of temporal storages (size: 193KB)
  44 (I): Address of output (matmul) 'output_layer' (size: 64B, dtype: int16, shape: (1, 10), alignment: 2 words (4 bytes)), aligned shape: (1, 10)
  48 (I): Address of placeholder 'input_layer' (size: 8KB, dtype: int16, shape: (1, 32, 32, 3), alignment: 2 words (4 bytes)), aligned shape: (1, 32, 32, 4)
  52 (I): Address of variables 'w0', 'b0', 's0', 'w1', 'b1', 's1', 'w2', 'b2', 's2', 'w3', 'b3', 's3' (size: 8276KB)
[Default Memory Map (start - end)] (entire range: [0 - 8679103], size: 8476KB)
  [      0 -      63]: output (matmul) 'output_layer' (size: 64B, dtype: int16, shape: (1, 10), alignment: 2 words (4 bytes)), aligned shape: (1, 10)
  [     64 -    8255]: placeholder 'input_layer' (size: 8KB, dtype: int16, shape: (1, 32, 32, 3), alignment: 2 words (4 bytes)), aligned shape: (1, 32, 32, 4)
  [   8256 -   12863]: variable 'w0' (size: 5KB, dtype: int16, shape: (64, 3, 3, 3), alignment: 2 words (4 bytes)), aligned shape: (64, 3, 3, 4)
  [  12864 -   12991]: variable 'b0' (size: 128B, dtype: int16, shape: (64,), alignment: 2 words (4 bytes)), aligned shape: (64,)
  [  12992 -   13119]: variable 's0' (size: 128B, dtype: int16, shape: (64,), alignment: 2 words (4 bytes)), aligned shape: (64,)
  [  13120 -   86847]: variable 'w1' (size: 72KB, dtype: int16, shape: (64, 3, 3, 64), alignment: 2 words (4 bytes)), aligned shape: (64, 3, 3, 64)
  [  86848 -   86975]: variable 'b1' (size: 128B, dtype: int16, shape: (64,), alignment: 2 words (4 bytes)), aligned shape: (64,)
  [  86976 -   87103]: variable 's1' (size: 128B, dtype: int16, shape: (64,), alignment: 2 words (4 bytes)), aligned shape: (64,)
  [  87104 - 8475711]: variable 'w2' (size: 8192KB, dtype: int16, shape: (256, 16384), alignment: 2 words (4 bytes)), aligned shape: (256, 16384)
  [8475712 - 8476223]: variable 'b2' (size: 512B, dtype: int16, shape: (256,), alignment: 2 words (4 bytes)), aligned shape: (256,)
  [8476224 - 8476735]: variable 's2' (size: 512B, dtype: int16, shape: (256,), alignment: 2 words (4 bytes)), aligned shape: (256,)
  [8476736 - 8481855]: variable 'w3' (size: 5KB, dtype: int16, shape: (10, 256), alignment: 2 words (4 bytes)), aligned shape: (10, 256)
  [8481856 - 8481919]: variable 'b3' (size: 64B, dtype: int16, shape: (10,), alignment: 2 words (4 bytes)), aligned shape: (10,)
  [8481920 - 8481983]: variable 's3' (size: 64B, dtype: int16, shape: (10,), alignment: 2 words (4 bytes)), aligned shape: (10,)
  [8481984 - 8679103]: temporal storages (size: 193KB)
# IP-XACT was generated. Check the current directory.
# start
# end
# execution cycles:     3724629
OK (           0           0 ) orig:       -10533  check:       -10533
OK (           0           1 ) orig:        13055  check:        13055
OK (           0           2 ) orig:        -7565  check:        -7565
OK (           0           3 ) orig:        -5662  check:        -5662
OK (           0           4 ) orig:        -4482  check:        -4482
OK (           0           5 ) orig:          350  check:          350
OK (           0           6 ) orig:        -7702  check:        -7702
OK (           0           7 ) orig:         5641  check:         5641
OK (           0           8 ) orig:         3247  check:         3247
OK (           0           9 ) orig:         5189  check:         5189
# verify: PASSED
```

To control the generated hardware from a real software on CPU, please check **[Register Map]** and **[Default Memory Map]**.
"Register Map" indicates the memory address map of control registers which can be accessed from a software.

- "Start" register (address 16): A software starts the computation by writing '1' to this register.
- "Busy" register (address 20): A software can check the busy/idle state by reading this register.
- "Global address offset" register (address 36): A software can change the address offset for all DMA accesses by the NNgen hardware. In many cases, a shared memory space between CPU and hardware is used. To avoid illegal memory access by the hardware, please carefully assign the correct address to this register.
- In addition to "Global address offset", you can specify relative addresses for temporal memory space (Address of temporal storages, address 40), output data (Address of output, address 44 in this example, but it can be changed if you use a different model definition), input data (Address of placeholder, address 48 in this example, but it may be different. There will be multiple registers, if you use multiple placeholders in your model definition), parameter data (Address of variables, address 52 in this example, but it will be different, if you use multiple placeholders and outputs).

```
[Register Map]
   0 (O): header0 (default: 0)
   4 (O): header1 (default: 0)
   8 (O): header2 (default: 0)
  12 (O): header3 (default: 0)
  16 (I): Start (set '1' to run)
  20 (O): Busy (returns '1' when running)
  24 (I): Reset (set '1' to initialize internal logic)
  28 (O): Opcode from extern objects to SW (returns '0' when idle)
  32 (I): Resume extern objects (set '1' to resume)
  36 (I): Global address offset (default: 0)
  40 (I): Address of temporal storages (size: 193KB)
  44 (I): Address of output (matmul) 'output_layer' (size: 64B, dtype: int16, shape: (1, 10), alignment: 2 words (4 bytes)), aligned shape: (1, 10)
  48 (I): Address of placeholder 'input_layer' (size: 8KB, dtype: int16, shape: (1, 32, 32, 3), alignment: 2 words (4 bytes)), aligned shape: (1, 32, 32, 4)
  52 (I): Address of variables 'w0', 'b0', 's0', 'w1', 'b1', 's1', 'w2', 'b2', 's2', 'w3', 'b3', 's3' (size: 8276KB)
```

(7) Implement the generated NNgen hardware on an FPGA
--------------------

If you generated an IP-XACT IP-core, please integrate it on the vender IDE, such as Vivado, according to the IP-core based design flow.

(8) Run the synthesized hardware on an FPGA
--------------------

There are actually various alternatives to access the generated hardware from a software.
The control sequence of the software is very simple:

- Write input data on the off-chip memory by a software. Note that all placeholders, variables, and operators have the dedicated memory alignments. **Please check the "word_alignment" and "aligned_shape" of each object in the synthesis log**. If the word alignment is greater than 1 and the original shape and aligned_shape are different, a padding must be inserted to the original data according to the the difference between the original shape and the aligned shape. In most cases, you can convert a original data to a padded data easily by "np.pad" method.
- Load the weight parameter file (saved above by "np.save" method) and write it on the off-chip memory. 
- Write a global address offset and relative addresses for temporal space, output data, input data, and variable data via the corresponding registers.
- Write '1' to Start register (address 16)
- Polling Busy register (address 20) by a while-loop
- Read the computation results from the output address. Note that the output data also has a dedicated aligned shape. **Please check the "aligned_shape" in the synthesis log.**


Related project
==============================

[Veriloggen](https://github.com/PyHDI/veriloggen)
- A Mixed-Paradigm Hardware Construction Framework

[Pyverilog](https://github.com/PyHDI/Pyverilog)
- Python-based Hardware Design Processing Toolkit for Verilog HDL
