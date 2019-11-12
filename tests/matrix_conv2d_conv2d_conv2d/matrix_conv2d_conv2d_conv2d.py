from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import functools
import math
import numpy as np

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng

from veriloggen import *
import veriloggen.thread as vthread
import veriloggen.types.axi as axi


def run(act_shape=(1, 7, 7, 15),
        weight1_shape=(7, 3, 3, 15), bias1_shape=None, scale1_shape=None,
        weight2_shape=(9, 3, 3, 7), bias2_shape=None, scale2_shape=None,
        weight3_shape=(9, 3, 3, 9), bias3_shape=None, scale3_shape=None,
        act_dtype=ng.int32,
        weight1_dtype=ng.int32, bias1_dtype=ng.int32, scale1_dtype=ng.int32,
        weight2_dtype=ng.int32, bias2_dtype=ng.int32, scale2_dtype=ng.int32,
        weight3_dtype=ng.int32, bias3_dtype=ng.int32, scale3_dtype=ng.int32,
        tmp_dtype=ng.int32,
        out_dtype=ng.int32,
        stride1=(1, 1, 1, 1), stride2=(1, 1, 1, 1), stride3=(1, 1, 1, 1),
        rshift_mul1=None, rshift_sum1=None, rshift_out1=None,
        rshift_mul2=None, rshift_sum2=None, rshift_out2=None,
        rshift_mul3=None, rshift_sum3=None, rshift_out3=None,
        act_func1=None, act_func2=None, act_func3=None,
        par_ich1=1, par_och1=1, par_col1=1, par_row1=1,
        concur_och1=None, stationary1='filter',
        par_ich2=1, par_och2=1, par_col2=1, par_row2=1,
        concur_och2=None, stationary2='filter',
        par_ich3=1, par_och3=1, par_col3=1, par_row3=1,
        concur_och3=None, stationary3='filter',
        input_ram_size1=None, filter_ram_size1=None,
        bias_ram_size1=None, scale_ram_size1=None,
        out_ram_size1=None,
        input_ram_size2=None, filter_ram_size2=None,
        bias_ram_size2=None, scale_ram_size2=None,
        out_ram_size2=None,
        input_ram_size3=None, filter_ram_size3=None,
        bias_ram_size3=None, scale_ram_size3=None,
        out_ram_size3=None,
        axi_datawidth=32, silent=False,
        filename=None, simtype='iverilog', outputfile=None):

    # create target hardware
    act = ng.placeholder(act_dtype, shape=act_shape, name='act')

    weight1 = ng.variable(weight1_dtype, shape=weight1_shape,
                          name='weight1')

    if bias1_shape is not None:
        bias1 = ng.variable(bias1_dtype, bias1_shape, name='bias1')
    else:
        bias1 = None

    if scale1_shape is not None:
        scale1 = ng.variable(scale1_dtype, scale1_shape, name='scale1')
    else:
        scale1 = None

    weight2 = ng.variable(weight2_dtype, shape=weight2_shape,
                          name='weight2')

    if bias2_shape is not None:
        bias2 = ng.variable(bias2_dtype, bias2_shape, name='bias2')
    else:
        bias2 = None

    if scale2_shape is not None:
        scale2 = ng.variable(scale2_dtype, scale2_shape, name='scale2')
    else:
        scale2 = None

    weight3 = ng.variable(weight3_dtype, shape=weight3_shape,
                          name='weight3')

    if bias3_shape is not None:
        bias3 = ng.variable(bias3_dtype, bias3_shape, name='bias3')
    else:
        bias3 = None

    if scale3_shape is not None:
        scale3 = ng.variable(scale3_dtype, scale3_shape, name='scale3')
    else:
        scale3 = None

    tmp = ng.conv2d(act, weight1, stride1,
                    bias1, scale1,
                    rshift_mul1, rshift_sum1, rshift_out1,
                    act_func1, 'SAME',
                    tmp_dtype, ng.int32, ng.int32,
                    'conv2d_1',
                    par_ich1, par_och1, par_col1, par_row1,
                    concur_och1, stationary1,
                    input_ram_size1, filter_ram_size1,
                    bias_ram_size1, scale_ram_size1,
                    None, None, None,
                    out_ram_size1)

    tmp = ng.conv2d(tmp, weight2, stride2,
                    bias2, scale2,
                    rshift_mul2, rshift_sum2, rshift_out2,
                    act_func2, 'SAME',
                    out_dtype, ng.int32, ng.int32,
                    'conv2d_2',
                    par_ich2, par_och2, par_col2, par_row2,
                    concur_och2, stationary2,
                    input_ram_size2, filter_ram_size2,
                    bias_ram_size2, scale_ram_size2,
                    None, None, None,
                    out_ram_size2)

    out = ng.conv2d(tmp, weight3, stride3,
                    bias3, scale3,
                    rshift_mul3, rshift_sum3, rshift_out3,
                    act_func3, 'SAME',
                    out_dtype, ng.int32, ng.int32,
                    'conv2d_3',
                    par_ich3, par_och3, par_col3, par_row3,
                    concur_och3, stationary3,
                    input_ram_size3, filter_ram_size3,
                    bias_ram_size3, scale_ram_size3,
                    None, None, None,
                    out_ram_size3)

    targ = ng.to_veriloggen([out], 'matrix_conv2d_conv2d_conv2d', silent=silent,
                            config={'maxi_datawidth': axi_datawidth})

    # verification data
    vact = np.arange(act.length, dtype=np.int64).reshape(act.shape) % [16]

    vweight1 = np.arange(weight1.length,
                         dtype=np.int64).reshape(weight1_shape) % [32] - [16]

    if bias1 is not None:
        vbias1 = np.arange(bias1.length,
                           dtype=np.int64).reshape(bias1.shape) % [16]
    else:
        vbias1 = None

    if scale1 is not None:
        vscale1 = np.arange(scale1.length,
                            dtype=np.int64).reshape(scale1.shape) % [8]
    else:
        vscale1 = None

    vweight2 = np.arange(weight2.length,
                         dtype=np.int64).reshape(weight2_shape) % [32] - [16]

    if bias2 is not None:
        vbias2 = np.arange(bias2.length,
                           dtype=np.int64).reshape(bias2.shape) % [16]
    else:
        vbias2 = None

    if scale2 is not None:
        vscale2 = np.arange(scale2.length,
                            dtype=np.int64).reshape(scale2.shape) % [8]
    else:
        vscale2 = None

    vweight3 = np.arange(weight3.length,
                         dtype=np.int64).reshape(weight3_shape) % [32] - [16]

    if bias3 is not None:
        vbias3 = np.arange(bias3.length,
                           dtype=np.int64).reshape(bias3.shape) % [16]
    else:
        vbias3 = None

    if scale3 is not None:
        vscale3 = np.arange(scale3.length,
                            dtype=np.int64).reshape(scale3.shape) % [8]
    else:
        vscale3 = None

    vtmp = ng.verify.conv2d(vact, vweight1, stride1,
                            vbias1, vscale1,
                            rshift_mul1, rshift_sum1, rshift_out1,
                            act_func1, 'SAME',
                            tmp_dtype, ng.int32, ng.int32,
                            'conv2d_1',
                            par_ich1, par_och1, par_col1, par_row1,
                            concur_och1, stationary1,
                            input_ram_size1, filter_ram_size1,
                            bias_ram_size1, scale_ram_size1,
                            None, None, None,
                            out_ram_size1,
                            False,
                            act_dtype, weight1_dtype)

    vtmp = ng.verify.conv2d(vtmp, vweight2, stride2,
                            vbias2, vscale2,
                            rshift_mul2, rshift_sum2, rshift_out2,
                            act_func2, 'SAME',
                            out_dtype, ng.int32, ng.int32,
                            'conv2d_2',
                            par_ich2, par_och2, par_col2, par_row2,
                            concur_och2, stationary2,
                            input_ram_size2, filter_ram_size2,
                            bias_ram_size2, scale_ram_size2,
                            None, None, None,
                            out_ram_size2,
                            False,
                            tmp_dtype, weight2_dtype)

    vout = ng.verify.conv2d(vtmp, vweight3, stride3,
                            vbias3, vscale3,
                            rshift_mul3, rshift_sum3, rshift_out3,
                            act_func3, 'SAME',
                            out_dtype, ng.int32, ng.int32,
                            'conv2d_3',
                            par_ich3, par_och3, par_col3, par_row3,
                            concur_och3, stationary3,
                            input_ram_size3, filter_ram_size3,
                            bias_ram_size3, scale_ram_size3,
                            None, None, None,
                            out_ram_size3,
                            False,
                            tmp_dtype, weight3_dtype)

    # to memory image
    size_max = int(math.ceil(max(act.memory_size, weight1.memory_size,
                                 bias1.memory_size if bias1 is not None else 0,
                                 scale1.memory_size if scale1 is not None else 0,
                                 weight2.memory_size,
                                 bias2.memory_size if bias2 is not None else 0,
                                 scale2.memory_size if scale2 is not None else 0,
                                 weight3.memory_size,
                                 bias3.memory_size if bias3 is not None else 0,
                                 scale3.memory_size if scale3 is not None else 0,
                                 out.memory_size) / 4096)) * 4096
    check_addr = max(act.addr, weight1.addr,
                     bias1.addr if bias1 is not None else -1,
                     scale1.addr if scale1 is not None else -1,
                     weight2.addr,
                     bias2.addr if bias2 is not None else -1,
                     scale2.addr if scale2 is not None else -1,
                     weight3.addr,
                     bias3.addr if bias3 is not None else -1,
                     scale3.addr if scale3 is not None else -1,
                     out.addr) + size_max
    size_check = size_max
    tmp_addr = check_addr + size_check

    memimg_datawidth = 32
    mem = np.zeros([1024 * 1024 * 8 // memimg_datawidth], dtype=np.int64)
    mem = mem + [100]

    axi.set_memory(mem, vact, memimg_datawidth,
                   act_dtype.width, act.addr,
                   max(int(math.ceil(axi_datawidth / act_dtype.width)), par_ich1))

    axi.set_memory(mem, vweight1, memimg_datawidth,
                   weight1_dtype.width, weight1.addr,
                   max(int(math.ceil(axi_datawidth / weight1_dtype.width)), par_ich1))
    if bias1_shape is not None:
        axi.set_memory(mem, vbias1, memimg_datawidth,
                       bias1_dtype.width, bias1.addr,
                       max(int(math.ceil(axi_datawidth / bias1_dtype.width)), par_och1))
    if scale1_shape is not None:
        axi.set_memory(mem, vscale1, memimg_datawidth,
                       scale1_dtype.width, scale1.addr,
                       max(int(math.ceil(axi_datawidth / scale1_dtype.width)), par_och1))

    axi.set_memory(mem, vweight2, memimg_datawidth,
                   weight2_dtype.width, weight2.addr,
                   max(int(math.ceil(axi_datawidth / weight2_dtype.width)), par_ich2))
    if bias2_shape is not None:
        axi.set_memory(mem, vbias2, memimg_datawidth,
                       bias2_dtype.width, bias2.addr,
                       max(int(math.ceil(axi_datawidth / bias2_dtype.width)), par_och2))
    if scale2_shape is not None:
        axi.set_memory(mem, vscale2, memimg_datawidth,
                       scale2_dtype.width, scale2.addr,
                       max(int(math.ceil(axi_datawidth / scale2_dtype.width)), par_och2))

    axi.set_memory(mem, vweight3, memimg_datawidth,
                   weight3_dtype.width, weight3.addr,
                   max(int(math.ceil(axi_datawidth / weight3_dtype.width)), par_ich3))
    if bias3_shape is not None:
        axi.set_memory(mem, vbias3, memimg_datawidth,
                       bias3_dtype.width, bias3.addr,
                       max(int(math.ceil(axi_datawidth / bias3_dtype.width)), par_och3))
    if scale3_shape is not None:
        axi.set_memory(mem, vscale3, memimg_datawidth,
                       scale3_dtype.width, scale3.addr,
                       max(int(math.ceil(axi_datawidth / scale3_dtype.width)), par_och3))

    axi.set_memory(mem, vout, memimg_datawidth,
                   out_dtype.width, check_addr,
                   max(int(math.ceil(axi_datawidth / out_dtype.width)), par_och3))

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
                            out.addr, out_dtype.width)
                        check = memory.read_word(
                            bat * out.aligned_shape[1] * out.aligned_shape[2] * out.aligned_shape[3] +
                            y * out.aligned_shape[2] * out.aligned_shape[3] +
                            x * out.aligned_shape[3] + ch,
                            check_addr, out_dtype.width)
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
    lines = rslt.splitlines()
    if simtype == 'verilator' and lines[-1].startswith('-'):
        rslt = '\n'.join(lines[:-1])
    return rslt


if __name__ == '__main__':
    rslt = run(silent=False, filename='tmp.v')
    print(rslt)
