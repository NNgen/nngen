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


def run(act_shape=(15,),
        act_dtype=ng.int32, out_dtype=ng.int32,
        begins=(2,), ends=(9,), strides=(1,),
        par=1,
        axi_datawidth=32, silent=False,
        filename=None, simtype='iverilog', outputfile=None):

    # create target hardware
    act = ng.placeholder(act_dtype, shape=act_shape, name='act')
    out = ng.slice_(act, begins, ends, strides,
                    dtype=out_dtype, name='out',
                    par=par)

    targ = ng.to_veriloggen([out], 'matrix_slice', silent=silent,
                            config={'maxi_datawidth': axi_datawidth})

    # verification data
    vact = np.arange(act.length, dtype=np.int64).reshape(act.shape)

    eval_outs = ng.eval([out], act=vact)
    vout = eval_outs[0]

    # to memory image
    size_max = int(math.ceil(max(act.memory_size, out.memory_size) / 4096)) * 4096
    check_addr = max(act.addr, out.addr) + size_max
    size_check = size_max
    tmp_addr = check_addr + size_check

    memimg_datawidth = 32
    mem = np.zeros([1024 * 1024 * 8 // (memimg_datawidth // 8)], dtype=np.int64)
    mem = mem + [100]

    axi.set_memory(mem, vact, memimg_datawidth,
                   act_dtype.width, act.addr,
                   max(int(math.ceil(axi_datawidth / act_dtype.width)), par))
    axi.set_memory(mem, vout, memimg_datawidth,
                   out_dtype.width, check_addr,
                   max(int(math.ceil(axi_datawidth / out_dtype.width)), par))

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
        for x in range(out.shape[0]):
            orig = memory.read_word(x,
                                    out.addr, out_dtype.width)
            check = memory.read_word(x,
                                     check_addr, out_dtype.width)

            if vthread.verilog.NotEql(orig, check):
                print('NG (', x,
                      ') orig: ', orig, ' check: ', check)
                ok = False
            # else:
            #    print('OK (', x,
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
        Delay(1000000),
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
