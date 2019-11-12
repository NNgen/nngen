from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng
from veriloggen import *
import veriloggen.thread as vthread
import veriloggen.types.axi as axi


def mkTest(n_input=784, n_classes=10):
    # create target hardware
    x = ng.placeholder(ng.int32, shape=[n_input])

    w1 = ng.variable(ng.int32, shape=(n_input, n_input), name='h1')
    w2 = ng.variable(ng.int32, shape=(n_input, n_input), name='h2')
    w3 = ng.variable(ng.int32, shape=(n_classes, n_input), name='out')

    l1 = ng.matmul(x, w1, transposed_b=True)
    l1 = ng.relu(l1)

    l2 = ng.matmul(l1, w2, transposed_b=True)
    l2 = ng.relu(l2)

    out = ng.matmul(l2, w3, transposed_b=True)

    targ = ng.to_veriloggen([out], 'mlp')
    #targ = ng.to_ipxact([model], 'mlp')

    # test controller
    m = Module('test')
    params = m.copy_params(targ)
    ports = m.copy_sim_ports(targ)
    clk = ports['CLK']
    resetn = ports['RESETN']
    rst = m.Wire('RST')
    rst.assign(Not(resetn))

    # AXI memory model
    memory = axi.AxiMemoryModel(m, 'memory', clk, rst)
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

        start_time = time_counter.value
        ng.sim.start(_saxi)

        print('# start')

        ng.sim.wait(_saxi)
        end_time = time_counter.value

        print('# end')
        print('# execution cycles: %d' % (end_time - start_time))

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

    return m


if __name__ == '__main__':
    test = mkTest()
    verilog = test.to_verilog('tmp.v')
    # print(verilog)

    sim = simulation.Simulator(test)
    rslt = sim.run()
    print(rslt)
