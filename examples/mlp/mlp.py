from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import math
import numpy as np

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import nngen as ng
from veriloggen import *
import veriloggen.thread as vthread
import veriloggen.types.axi as axi


def run(act_dtype=ng.int16, weight_dtype=ng.int16,
        par_ich=2, par_och=2,
        chunk_size=64, axi_datawidth=32, silent=False,
        weight_filename='mlp.npz',
        verilog_filename=None,
        sim_filename=None,
        simtype='iverilog',
        # simtype='verilator',
        # simtype=None,  # no RTL simulation
        ):

    # --------------------
    # (1) Represent a DNN model as a dataflow by NNgen operators
    # --------------------

    # input
    input_layer = ng.placeholder(act_dtype, shape=(1, 784),
                                 name='input_layer')

    #w0 = ng.variable(weight_dtype, shape=(1024, 784), name='w0')
    w0 = ng.variable(weight_dtype, shape=(128, 784), name='w0')
    a0 = ng.matmul(input_layer, w0, transposed_b=True, act_func=ng.relu,
                   sum_dtype=ng.dtype_int(64))

    #w1 = ng.variable(weight_dtype, shape=(10, 1024), name='w1')
    w1 = ng.variable(weight_dtype, shape=(10, 128), name='w1')
    output_layer = ng.matmul(a0, w1, transposed_b=True, name='output_layer',
                             sum_dtype=ng.dtype_int(64))

    # --------------------
    # (2) Assign weights to the NNgen operators
    # --------------------

    # In this example, random floating-point values are assigned.
    # In a real case, you should assign actual weight values
    # obtianed by a training on DNN framework.

    # If you don't you NNgen's quantizer, you can assign integer weights to each tensor.

    w0_value = np.random.normal(size=w0.length).reshape(w0.shape)
    w0_value = np.clip(w0_value, -3.0, 3.0)
    w0.set_value(w0_value)

    w1_value = np.random.normal(size=w1.length).reshape(w1.shape)
    w1_value = np.clip(w1_value, -3.0, 3.0)
    w1.set_value(w1_value)

    # Quantizing the floating-point weights by the NNgen quantizer.
    # Alternatively, you can assign integer weights by yourself to each tensor.

    img_mean = np.array([0.485]).astype(np.float32)
    img_std = np.array([0.229]).astype(np.float32)

    if act_dtype.width > 8:
        act_scale_factor = 128
    else:
        act_scale_factor = int(round(2 ** (act_dtype.width - 1) * 0.5))

    input_scale_factors = {'input_layer': act_scale_factor}
    input_means = {'input_layer': img_mean * act_scale_factor}
    input_stds = {'input_layer': img_std * act_scale_factor}

    ng.quantize([output_layer], input_scale_factors, input_means, input_stds)


    # --------------------
    # (3) Assign hardware attributes
    # --------------------

    # conv2d, matmul
    # par_ich: parallelism in input-channel
    # par_och: parallelism in output-channel
    # par_col: parallelism in pixel column
    # par_row: parallelism in pixel row
    # cshamt_out: right shift amount after applying bias/scale

    a0.attribute(par_ich=par_ich, par_och=par_och)
    output_layer.attribute(par_ich=par_ich, par_och=par_och)

    # cshamt_out: right shift amount after applying bias/scale
    # If you assign integer weights by yourself to each tensor,
    # cshamt (constant shift amount) must be assigned to each operator.

    # a0.attribute(cshamt_out=weight_dtype.width + 1)
    # output_layer.attribute(cshamt_out=weight_dtype.width + 1)


    # --------------------
    # (4) Verify the DNN model behavior by executing the NNgen dataflow as a software
    # --------------------

    # In this example, random integer values are assigned.
    # In real case, you should assign actual integer activation values, such as an image.

    input_layer_value = np.random.normal(size=input_layer.length).reshape(input_layer.shape)
    input_layer_value = input_layer_value * img_std + img_mean
    input_layer_value = np.clip(input_layer_value, -5.0, 5.0)
    input_layer_value = input_layer_value * act_scale_factor
    input_layer_value = np.clip(input_layer_value,
                                -1 * 2 ** (act_dtype.width - 1) - 1, 2 ** (act_dtype.width - 1))
    input_layer_value = np.round(input_layer_value).astype(np.int64)

    eval_outs = ng.eval([output_layer], input_layer=input_layer_value)
    output_layer_value = eval_outs[0]

    # print(output_layer_value)
    # breakpoint()

    # --------------------
    # (5) Convert the NNgen dataflow to a hardware description (Verilog HDL and IP-XACT)
    # --------------------

    # to Veriloggen object
    # targ = ng.to_veriloggen([output_layer], 'mlp', silent=silent,
    #                        config={'maxi_datawidth': axi_datawidth})

    # to IP-XACT (the method returns Veriloggen object, as well as to_veriloggen)
    targ = ng.to_ipxact([output_layer], 'mlp', silent=silent,
                        config={'maxi_datawidth': axi_datawidth})

    # to Verilog HDL RTL (the method returns a source code text)
    # rtl = ng.to_verilog([output_layer], 'mlp', silent=silent,
    #                    config={'maxi_datawidth': axi_datawidth})


    # --------------------
    # (6) Save the quantized weights
    # --------------------
    param_data = ng.export_ndarray([output_layer], chunk_size)
    np.savez_compressed(weight_filename, param_data)


    # --------------------
    # (7) Simulate the generated hardware by Veriloggen and Verilog simulator
    # --------------------

    if simtype is None:
        sys.exit()

    param_bytes = len(param_data)

    variable_addr = int(
        math.ceil((input_layer.addr + input_layer.memory_size) / chunk_size)) * chunk_size
    check_addr = int(math.ceil((variable_addr + param_bytes) / chunk_size)) * chunk_size
    tmp_addr = int(math.ceil((check_addr + output_layer.memory_size) / chunk_size)) * chunk_size

    memimg_datawidth = 32
    mem = np.zeros([1024 * 1024 * 256 // (memimg_datawidth // 8)], dtype=np.int64)
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
    if sim_filename is None:
        sim_filename = os.path.splitext(os.path.basename(__file__))[0] + '.out'

    memimg_name = 'memimg_' + sim_filename

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
    if verilog_filename is not None:
        m.to_verilog(verilog_filename)

    # run simulation
    sim = simulation.Simulator(m, sim=simtype)
    rslt = sim.run(outputfile=sim_filename)
    lines = rslt.splitlines()
    if simtype == 'verilator' and lines[-1].startswith('-'):
        rslt = '\n'.join(lines[:-1])
    return rslt


if __name__ == '__main__':
    rslt = run(silent=False, verilog_filename='tmp.v')
    print(rslt)
