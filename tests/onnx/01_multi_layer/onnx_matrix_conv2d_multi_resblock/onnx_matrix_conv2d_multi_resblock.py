from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import functools
import math
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng

from veriloggen import *
import veriloggen.thread as vthread
import veriloggen.types.axi as axi


class MatrixConv2dMultiResblock(nn.Module):
    def __init__(self, weight_shape, stride=1, padding=0,
                 with_batchnorm=False, act_func='ReLU'):

        super(MatrixConv2dMultiResblock, self).__init__()

        # layer 1
        self.conv1 = nn.Conv2d(weight_shape[3], weight_shape[0], weight_shape[1],
                               stride=stride, padding=padding, bias=not with_batchnorm)

        if with_batchnorm:
            self.bn1 = nn.BatchNorm2d(weight_shape[0])
        else:
            self.bn1 = None

        if act_func is not None:
            self.f1 = getattr(nn, act_func)()
        else:
            self.f1 = None

        # layer 2
        self.conv2 = nn.Conv2d(weight_shape[3], weight_shape[0], weight_shape[1],
                               stride=stride, padding=padding, bias=not with_batchnorm)

        if with_batchnorm:
            self.bn2 = nn.BatchNorm2d(weight_shape[0])
        else:
            self.bn2 = None

        if act_func is not None:
            self.f2 = getattr(nn, act_func)()
        else:
            self.f2 = None

    def forward(self, x):
        # layer 1
        y = self.conv1(x)

        if self.bn1 is not None:
            y = self.bn1(y)

        if self.f1 is not None:
            y = self.f1(y)

        y = torch.add(x, y)

        # layer 2
        x = y
        y = self.conv2(y)

        if self.bn2 is not None:
            y = self.bn2(y)

        if self.f2 is not None:
            y = self.f2(y)

        y = torch.add(x, y)

        return y


def run(act_shape=(1, 7, 7, 15), weight_shape=(15, 3, 3, 15),
        act_dtype=ng.int32, weight_dtype=ng.int32,
        stride=1, padding=1,
        with_batchnorm=False, act_func='ReLU', disable_fusion=False,
        par_ich=1, par_och=1, par_col=1, par_row=1,
        concur_och=None, stationary='filter',
        chunk_size=64,
        axi_datawidth=32, silent=False,
        filename=None, simtype='iverilog', outputfile=None):

    if weight_shape[0] != weight_shape[3]:
        raise ValueError('not supported shape: weight_shape[0] != weight_shape[3]')

    # pytorch model
    model = MatrixConv2dMultiResblock(weight_shape, stride, padding,
                                      with_batchnorm, act_func)

    # overwrite weight values for test
    # model.conv.weight.data = torch.from_numpy(np.ones_like(model.conv.weight.data.numpy()))
    # model.conv.bias.data = torch.from_numpy(np.zeros_like(model.conv.bias.data.numpy()))

    # Pytorch to ONNX
    onnx_filename = 'onnx_matrix_conv2d_multi_resblock.onnx'
    dummy_input = torch.randn(*act_shape).transpose(1, 3)
    input_names = ['act']
    output_names = ['out']
    model.eval()
    torch.onnx.export(model, dummy_input, onnx_filename,
                      input_names=input_names, output_names=output_names)

    # --------------------
    # (1) Represent a DNN model as a dataflow by NNgen operators
    # --------------------

    # ONNX to NNgen
    value_dtypes = {'act': act_dtype,
                    '0.weight': weight_dtype,
                    'out': act_dtype}

    (outputs, placeholders, variables,
     constants, operators) = ng.from_onnx(onnx_filename,
                                          value_dtypes=value_dtypes,
                                          default_placeholder_dtype=act_dtype,
                                          default_variable_dtype=weight_dtype,
                                          default_constant_dtype=weight_dtype,
                                          default_operator_dtype=act_dtype,
                                          default_scale_dtype=ng.int32,
                                          default_bias_dtype=ng.int32,
                                          disable_fusion=disable_fusion)

    # --------------------
    # (2) Assign quantized weights to the NNgen operators
    # --------------------

    if act_dtype.width > 8:
        act_scale_factor = 128
    else:
        act_scale_factor = int(round(2 ** (act_dtype.width - 1) * 0.5))

    input_scale_factors = {'act': act_scale_factor}

    ng.quantize(outputs, input_scale_factors)

    # --------------------
    # (3) Assign hardware attributes
    # --------------------

    for op in operators.values():
        if isinstance(op, ng.conv2d):
            op.attribute(par_ich=par_ich, par_och=par_och,
                         par_row=par_row, par_col=par_col,
                         concur_och=concur_och)

    # --------------------
    # (4) Verify the DNN model behavior by executing the NNgen dataflow as a software
    # --------------------

    act = placeholders['act']
    out = outputs['out']

    # verification data
    # random data
    std = 0.2
    mean = 0.5
    img = np.random.normal(size=act.length).astype(np.float32).reshape(act.shape)
    img = img * std + mean

    # execution on pytorch
    model_input = img

    if act.perm is not None:
        model_input = np.transpose(model_input, act.reversed_perm)

    model.eval()
    model_out = model(torch.from_numpy(model_input)).detach().numpy()
    if act.perm is not None and len(model_out.shape) == len(act.shape):
        model_out = np.transpose(model_out, act.perm)
    scaled_model_out = model_out * out.scale_factor

    # software-based verification
    vact = img * act_scale_factor
    vact = np.clip(vact,
                   -1.0 * (2 ** (act.dtype.width - 1) - 1),
                   1.0 * (2 ** (act.dtype.width - 1) - 1))
    vact = np.round(vact).astype(np.int64)

    eval_outs = ng.eval([out], act=vact)
    vout = eval_outs[0]

    mean_square_error = np.sum((vout - scaled_model_out) ** 2) / vout.size
    corrcoef = np.corrcoef(model_out.reshape([-1]), vout.reshape([-1]))

    # breakpoint()

    # --------------------
    # (5) Convert the NNgen dataflow to a hardware description (Verilog HDL and IP-XACT)
    # --------------------

    targ = ng.to_veriloggen([out], 'onnx_matrix_conv2d', silent=silent,
                            config={'maxi_datawidth': axi_datawidth,
                                    'chunk_size': chunk_size})

    # --------------------
    # (6) Simulate the generated hardware by Veriloggen and Verilog simulator
    # --------------------

    if simtype is None:
        sys.exit()

    # to memory image
    param_data = ng.export_ndarray([out], chunk_size)
    param_bytes = len(param_data)

    variable_addr = int(math.ceil((act.addr + act.memory_size) / chunk_size)) * chunk_size
    check_addr = int(math.ceil((variable_addr + param_bytes) / chunk_size)) * chunk_size
    tmp_addr = int(math.ceil((check_addr + out.memory_size) / chunk_size)) * chunk_size

    memimg_datawidth = 32
    mem = np.zeros([1024 * 1024 * 8 // (memimg_datawidth // 8)], dtype=np.int64)
    mem = mem + [100]

    # placeholder
    axi.set_memory(mem, vact, memimg_datawidth,
                   act_dtype.width, act.addr,
                   max(int(math.ceil(axi_datawidth / act_dtype.width)), par_ich))

    # parameters (variable and constant)
    axi.set_memory(mem, param_data, memimg_datawidth,
                   8, variable_addr)

    # verification data
    axi.set_memory(mem, vout, memimg_datawidth,
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
        for bat in range(out.shape[0]):
            for y in range(out.shape[1]):
                for x in range(out.shape[2]):
                    for ch in range(out.shape[3]):
                        orig = memory.read_word(
                            bat * out.aligned_shape[1] * out.aligned_shape[2] * out.aligned_shape[3] +
                            y * out.aligned_shape[2] * out.aligned_shape[3] +
                            x * out.aligned_shape[3] + ch,
                            out.addr, act_dtype.width)
                        check = memory.read_word(
                            bat * out.aligned_shape[1] * out.aligned_shape[2] * out.aligned_shape[3] +
                            y * out.aligned_shape[2] * out.aligned_shape[3] +
                            x * out.aligned_shape[3] + ch,
                            check_addr, act_dtype.width)

                        if vthread.verilog.NotEql(orig, check):
                            print('NG (', bat, y, x, ch,
                                  ') orig: ', orig, ' check: ', check)
                            ok = False
                        # else:
                        #    print('OK (', bat, y, x, ch,
                        #          ') orig: ', orig, ' check: ', check)

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

    return rslt


if __name__ == '__main__':
    rslt = run(silent=False, filename='tmp.v')
    print(rslt)
