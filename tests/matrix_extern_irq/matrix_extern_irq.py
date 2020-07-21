from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import functools
import math
import numpy as np

if sys.version_info.major < 3:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng

from veriloggen import *
import veriloggen.thread as vthread
import veriloggen.types.axi as axi


def run(a_shape=(15, 15), b_shape=(15, 15),
        a_dtype=ng.int32, b_dtype=ng.int32, c_dtype=ng.int32,
        par=1, axi_datawidth=32, interrupt_name='irq', silent=False,
        filename=None, simtype='iverilog', outputfile=None):

    # create target hardware
    a = ng.placeholder(a_dtype, shape=a_shape, name='a')
    b = ng.placeholder(b_dtype, shape=b_shape, name='b')

    d = ng.add(a, b, dtype=c_dtype, par=par)
    e = ng.add(b, a, dtype=c_dtype, par=par)

    # SW returns ng.add(x, y)
    f = ng.extern([d, e], shape=a_shape, opcode=0x1,
                  func=lambda x, y: x + y)
    g = ng.sub(f, a)

    # SW returns d as-is
    h = ng.extern([g], shape=a_shape, opcode=0x2,
                  func=lambda x: x)
    c = ng.sub(h, b)

    targ = ng.to_veriloggen([c], 'matrix_extern', silent=silent,
                            config={'maxi_datawidth': axi_datawidth,
                                    'interrupt_name': interrupt_name})

    # verification data
    va = np.arange(a.length, dtype=np.int64).reshape(a.shape) % [16]
    vb = np.arange(b.length, dtype=np.int64).reshape(b.shape) % [32] + [16]

    eval_outs = ng.eval([c], a=va, b=vb)
    vc = eval_outs[0]

    # to memory image
    size_max = int(math.ceil(max(a.memory_size, b.memory_size, c.memory_size) / 4096)) * 4096
    check_addr = max(a.addr, b.addr, c.addr) + size_max
    size_check = size_max
    tmp_addr = check_addr + size_check

    memimg_datawidth = 32
    mem = np.zeros([1024 * 1024 * 8 // (memimg_datawidth // 8)], dtype=np.int64)
    mem = mem + [100]

    axi.set_memory(mem, va, memimg_datawidth,
                   a_dtype.width, a.addr,
                   max(int(math.ceil(axi_datawidth / a_dtype.width)), par))
    axi.set_memory(mem, vb, memimg_datawidth,
                   b_dtype.width, b.addr,
                   max(int(math.ceil(axi_datawidth / b_dtype.width)), par))
    axi.set_memory(mem, vc, memimg_datawidth,
                   c_dtype.width, check_addr,
                   max(int(math.ceil(axi_datawidth / c_dtype.width)), par))

    # test controller
    m = Module('test')
    params = m.copy_params(targ)
    ports = m.copy_sim_ports(targ)
    clk = ports['CLK']
    resetn = ports['RESETN']
    irq = ports[interrupt_name]
    rst = m.Wire('RST')
    rst.assign(Not(resetn))

    # AXI memory model
    if outputfile is None:
        outputfile = os.path.splitext(os.path.basename(__file__))[0] + '.out'

    memimg_name = 'memimg_' + outputfile

    memory = axi.AxiMemoryModel(m, 'memory', clk, rst,
                                datawidth=axi_datawidth,
                                memimg_datawidth=memimg_datawidth,
                                memimg=mem, memimg_name=memimg_name)
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

    num_rep = functools.reduce(lambda x, y: x * y, c.shape[:-1], 1)

    def ctrl():
        for i in range(100):
            pass

        ng.sim.set_global_addrs(_saxi, tmp_addr)

        araddr_irq_ier = ng.control_reg_interrupt_ier * 4
        araddr_irq_isr = ng.control_reg_interrupt_isr * 4
        araddr_irq_iar = ng.control_reg_interrupt_iar * 4
        _saxi.write(araddr_irq_ier , 3) # irq enable

        start_time = time_counter.value
        ng.sim.start(_saxi)

        print('# start')

        # from extern-send
        while irq == 0:
            pass
        irq_stat = _saxi.read(araddr_irq_isr)
        if irq_stat != 2:
            print('# Unexpected irq signal')
            print('# verify: FAILED')
            vthread.finish()
        print('# irq stat = %d' % irq_stat)
        _saxi.write(araddr_irq_iar , 2) # irq acknowledge: just irq stat ack_bit negation

        # from extern-send
        araddr = ng.control_reg_extern_send * 4
        v = _saxi.read(araddr)

        print('# opcode = %d' % v)

        for i in range(num_rep):
            for j in range(c.shape[-1]):
                x_offset = tmp_addr - d.default_global_addr
                y_offset = tmp_addr - e.default_global_addr
                z_offset = tmp_addr - f.default_global_addr
                x = memory.read_word(i * c.aligned_shape[-1] + j,
                                     d.addr + x_offset, c_dtype.width)
                y = memory.read_word(i * c.aligned_shape[-1] + j,
                                     e.addr + y_offset, c_dtype.width)
                z = x + y
                memory.write_word(i * c.aligned_shape[-1] + j,
                                  f.addr + z_offset, z, c_dtype.width)

        # to extern-recv
        awaddr = ng.control_reg_extern_recv * 4
        _saxi.write(awaddr, 1)

        # from extern-send
        while irq == 0:
            pass
        irq_stat = _saxi.read(araddr_irq_isr)
        if irq_stat != 2:
            print('# Unexpected irq signal')
            print('# verify: FAILED')
            vthread.finish()
        print('# irq stat = %d' % irq_stat)
        _saxi.write(araddr_irq_iar , 0xff) # irq acknowledge: all of irq stat nagetion

        # from extern-send
        araddr = ng.control_reg_extern_send * 4
        v = _saxi.read(araddr)

        print('# opcode = %d' % v)

        for i in range(num_rep):
            for j in range(c.shape[-1]):
                x_offset = tmp_addr - g.default_global_addr
                z_offset = tmp_addr - h.default_global_addr
                x = memory.read_word(i * c.aligned_shape[-1] + j,
                                     g.addr + x_offset, c_dtype.width)
                z = x
                memory.write_word(i * c.aligned_shape[-1] + j,
                                  h.addr + z_offset, z, c_dtype.width)

        # to extern-recv
        awaddr = ng.control_reg_extern_recv * 4
        _saxi.write(awaddr, 1)

        # from extern-send
        while irq == 0:
            pass
        irq_stat = _saxi.read(araddr_irq_isr)
        if irq_stat != 1:
            print('# Unexpected irq signal')
            print('# verify: FAILED')
            vthread.finish()
        print('# irq stat = %d' % irq_stat)
        _saxi.write(araddr_irq_iar, 1) # irq acknowledge: irq stat busy_bit negation

        #ng.sim.wait(_saxi)
        end_time = time_counter.value

        print('# end')
        print('# execution cycles: %d' % (end_time - start_time))

        for i in range(100):
            pass

        # verify
        ok = True
        for i in range(num_rep):
            for j in range(c.shape[-1]):
                orig = memory.read_word(i * c.aligned_shape[-1] + j,
                                        c.addr, c_dtype.width)
                check = memory.read_word(i * c.aligned_shape[-1] + j,
                                         check_addr, c_dtype.width)

                if vthread.verilog.NotEql(orig, check):
                    print('NG', i, j, orig, check)
                    ok = False
                # else:
                #    print('OK', i, j, orig, check)

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
