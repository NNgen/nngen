from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import functools
import math

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng
from veriloggen import *
import veriloggen.thread as vthread
import veriloggen.types.axi as axi


def mkTest(ich=3, och=10, ch=64, ksize=3, stride=1, col=28, row=28):
    # create target hardware

    # layer 0: conv2d, max_pool_serial, relu
    input_layer = ng.placeholder(ng.int32, shape=(1, row, col, ich),
                                 name='input_layer')
    w0 = ng.variable(ng.int32, shape=(ch, ksize, ksize, ich), name='w0')
    a0 = ng.conv2d(input_layer, w0, strides=(1, stride, stride, 1))
    a0 = ng.max_pool_serial(a0, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1))
    a0 = ng.relu(a0)

    # layer 1: conv2d, relu, reshape
    w1 = ng.variable(ng.int32,
                     shape=(ch, ksize, ksize, a0.shape[-1]), name='w1')
    a1 = ng.conv2d(a0, w1, strides=(1, stride, stride, 1))
    a1 = ng.relu(a1)
    a1 = ng.reshape(a1, [-1])

    # layer 2: full-connection
    w2 = ng.variable(ng.int32, shape=(16, a1.shape[-1]), name='w2')
    a2 = ng.matmul(a1, w2, transposed_b=True)
    a2 = ng.relu(a2)

    # layer 3: full-connection
    w3 = ng.variable(ng.int32, shape=(och, a2.shape[-1]), name='w3')
    output_layer = ng.matmul(a2, w3, transposed_b=True, name='output_layer')

    targ = ng.to_veriloggen([output_layer], 'cnn')
    #targ = ng.to_ipxact([output_layer], 'cnn')

    # test controller
    m = Module('test')
    params = m.copy_params(targ)
    ports = m.copy_sim_ports(targ)
    clk = ports['CLK']
    resetn = ports['RESETN']
    rst = m.Wire('RST')
    rst.assign(Not(resetn))

    # AXI memory model
    memory = axi.AxiMemoryModel(m, 'memory', clk, rst,
                                mem_addrwidth=23)
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
        Delay(10000000),
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
