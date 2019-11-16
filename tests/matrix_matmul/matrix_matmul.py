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


def run(a_shape=(15, 15), b_shape=(15, 15),
        bias_shape=None, scale_shape=None,
        a_dtype=ng.int32, b_dtype=ng.int32,
        bias_dtype=ng.int32, scale_dtype=ng.int32,
        c_dtype=ng.int32,
        rshift_mul=None, rshift_sum=None, rshift_out=None,
        act_func=None,
        par_left_col=1, par_left_row=1, par_out_col=1,
        concur_out_col=None, stationary='right',
        left_ram_size=None, right_ram_size=None,
        bias_ram_size=None, scale_ram_size=None,
        out_ram_size=None,
        axi_datawidth=32, silent=False,
        filename=None, simtype='iverilog', outputfile=None):

    # create target hardware
    a = ng.placeholder(a_dtype, shape=a_shape, name='a')
    b = ng.placeholder(b_dtype, shape=b_shape, name='b')

    if bias_shape is not None:
        bias = ng.placeholder(bias_dtype, bias_shape, name='bias')
    else:
        bias = None

    if scale_shape is not None:
        scale = ng.placeholder(scale_dtype, scale_shape, name='scale')
    else:
        scale = None

    transposed_a = False
    transposed_b = True

    c = ng.matmul(a, b,
                  bias, scale,
                  transposed_a, transposed_b,
                  rshift_mul, rshift_sum, rshift_out,
                  act_func,
                  c_dtype, ng.int32, ng.int32,
                  'matmul',
                  par_left_col, par_left_row, par_out_col,
                  concur_out_col, stationary,
                  left_ram_size, right_ram_size,
                  bias_ram_size, scale_ram_size,
                  None, None, None,
                  out_ram_size)

    targ = ng.to_veriloggen([c], 'matrix_matmul', silent=silent,
                            config={'maxi_datawidth': axi_datawidth})

    # verification data
    va = np.arange(a.length, dtype=np.int64).reshape(a.shape) % [5]
    vb = np.arange(b.length, dtype=np.int64).reshape(b.shape) % [5] - [3]

    if bias is not None:
        vbias = np.arange(bias.length,
                          dtype=np.int64).reshape(bias.shape) % [4]
    else:
        vbias = None

    if scale is not None:
        vscale = np.arange(scale.length,
                           dtype=np.int64).reshape(scale.shape) % [6]
    else:
        vscale = None

    eval_outs = ng.eval([c], a=va, b=vb, bias=vbias, scale=vscale)
    vc = eval_outs[0]

    # to memory image
    size_max = int(math.ceil(max(a.memory_size, b.memory_size,
                                 bias.memory_size if bias is not None else 0,
                                 scale.memory_size if scale is not None else 0,
                                 c.memory_size) / 4096)) * 4096
    check_addr = max(a.addr, b.addr,
                     bias.addr if bias is not None else -1,
                     scale.addr if scale is not None else -1,
                     c.addr) + size_max
    size_check = size_max
    tmp_addr = check_addr + size_check

    memimg_datawidth = 32
    mem = np.zeros([1024 * 1024 * 8 // memimg_datawidth], dtype=np.int64)
    mem = mem + [100]

    axi.set_memory(mem, va, memimg_datawidth,
                   a_dtype.width, a.addr,
                   max(int(math.ceil(axi_datawidth / a_dtype.width)), par_left_col))

    axi.set_memory(mem, vb, memimg_datawidth,
                   b_dtype.width, b.addr,
                   max(int(math.ceil(axi_datawidth / b_dtype.width)), par_left_col))

    if bias is not None:
        axi.set_memory(mem, vbias, memimg_datawidth,
                       bias_dtype.width, bias.addr,
                       max(int(math.ceil(axi_datawidth / bias_dtype.width)), par_out_col))

    if scale is not None:
        axi.set_memory(mem, vscale, memimg_datawidth,
                       scale_dtype.width, scale.addr,
                       max(int(math.ceil(axi_datawidth / scale_dtype.width)), par_out_col))

    axi.set_memory(mem, vc, memimg_datawidth,
                   c_dtype.width, check_addr,
                   max(int(math.ceil(axi_datawidth / c_dtype.width)), par_out_col))

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
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                orig = memory.read_word(i * c.aligned_shape[1] + j,
                                        c.addr, c_dtype.width)
                check = memory.read_word(i * c.aligned_shape[1] + j,
                                         check_addr, c_dtype.width)

                if vthread.verilog.NotEql(orig, check):
                    print(i, j, orig, check)
                    ok = False

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
        Delay(1000000),
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
