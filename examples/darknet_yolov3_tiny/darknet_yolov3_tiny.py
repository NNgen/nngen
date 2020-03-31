from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import math
import numpy as np
import PIL
import urllib

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


def run(act_dtype=ng.int16, weight_dtype=ng.int16,
        bias_dtype=ng.int32, scale_dtype=ng.int16,
        disable_fusion=False,
        conv2d_par_ich=1, conv2d_par_och=1, conv2d_par_col=1, conv2d_par_row=1,
        conv2d_concur_och=None, conv2d_stationary='filter',
        pool_par=1, elem_par=1,
        chunk_size=64, axi_datawidth=32, silent=False,
        filename=None,
        # simtype='iverilog',
        # simtype='verilator',
        simtype=None,  # no RTL simulation
        outputfile=None,

        cfg_filename='yolov3-tiny.cfg',
        weights_filename='yolov3-tiny.weights',
        model_path='yolov3'):

    # input mean and standard deviation
    imagenet_mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225]).astype(np.float32)

    img_size = (416, 416)
    act_shape = (1, img_size[0], img_size[1], 3)

    # pytorch model
    model_url = "https://github.com/ultralytics/yolov3"
    if not os.path.isdir(model_path):
        raise FileNotFoundError("Download the YOLOv3 model using Pytorch, such as "
                                "'%s'. Then extract it, and rename it as '%s'" %
                                (model_url, model_path))

    # Darknet model configuration and pretrained weights
    cfg_url = "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg"
    if not os.path.isfile(cfg_filename):
        urllib.request.urlretrieve(cfg_url, cfg_filename)

    weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
    if not os.path.isfile(weights_filename):
        urllib.request.urlretrieve(weights_url, weights_filename)

    sys.path.insert(0, model_path)
    import models
    models.ONNX_EXPORT = True

    model = models.Darknet(cfg_filename, img_size).to('cpu')
    models.load_darknet_weights(model, weights_filename)

    # Pytorch to ONNX
    onnx_filename = 'yolov3-tiny.onnx'
    dummy_input = torch.randn(*act_shape).transpose(1, 3)
    input_names = ['act']
    output_names = ['scores', 'boxes']
    model.eval()
    torch.onnx.export(model, dummy_input, onnx_filename,
                      input_names=input_names, output_names=output_names)

    # --------------------
    # (1) Represent a DNN model as a dataflow by NNgen operators
    # --------------------

    # ONNX to NNgen
    dtypes = {}
    shapes = {}
    (outputs, placeholders, variables,
     constants, operators) = ng.from_onnx(onnx_filename,
                                          value_dtypes=dtypes,
                                          value_shapes=shapes,
                                          default_placeholder_dtype=act_dtype,
                                          default_variable_dtype=weight_dtype,
                                          default_constant_dtype=weight_dtype,
                                          default_operator_dtype=act_dtype,
                                          default_scale_dtype=scale_dtype,
                                          default_bias_dtype=bias_dtype,
                                          disable_fusion=disable_fusion,
                                          verbose=False)

    # --------------------
    # (2) Assign quantized weights to the NNgen operators
    # --------------------

    if act_dtype.width > 8:
        act_scale_factor = 128
    else:
        act_scale_factor = int(round(2 ** (act_dtype.width - 1) * 0.5))

    input_scale_factors = {'act': act_scale_factor}
    input_means = {'act': imagenet_mean * act_scale_factor}
    input_stds = {'act': imagenet_std * act_scale_factor}

    ng.quantize(outputs, input_scale_factors, input_means, input_stds)

    # --------------------
    # (3) Assign hardware attributes
    # --------------------

    for op in operators.values():
        if isinstance(op, ng.conv2d):
            op.attribute(par_ich=conv2d_par_ich,
                         par_och=conv2d_par_och,
                         par_col=conv2d_par_col,
                         par_row=conv2d_par_row,
                         concur_och=conv2d_concur_och,
                         stationary=conv2d_stationary)

        if isinstance(op, (ng.avg_pool, ng.max_pool,
                           ng.avg_pool_serial, ng.max_pool_serial)):
            op.attribute(par=pool_par)

        if ng.is_elementwise_operator(op):
            op.attribute(par=elem_par)

    # --------------------
    # (4) Verify the DNN model behavior by executing the NNgen dataflow as a software
    # --------------------

    act = placeholders['act']
    outs = (outputs['scores'], outputs['boxes'])

    # verification data
    img = np.array(PIL.Image.open('car416x416.png').convert('RGB')).astype(np.float32)
    img = img.reshape([1] + list(img.shape))

    img = img / 255
    img = (img - imagenet_mean) / imagenet_std

    # execution on pytorch
    model_input = img

    if act.perm is not None:
        model_input = np.transpose(model_input, act.reversed_perm)

    model.eval()
    model_rslts = model(torch.from_numpy(model_input))
    model_outs = [rslt.detach().numpy() for rslt in model_rslts]
    model_outs = [(np.transpose(model_out, act.perm)
                   if act.perm is not None and len(model_out.shape) == len(act.shape)
                   else model_out)
                  for model_out in model_outs]
    scaled_model_outs = [model_out * out.scale_factor
                         for model_out, out in zip(model_outs, outs)]

    # software-based verification
    vact = img * act_scale_factor
    vact = np.clip(vact,
                   -1.0 * (2 ** (act.dtype.width - 1) - 1),
                   1.0 * (2 ** (act.dtype.width - 1) - 1))
    vact = np.round(vact).astype(np.int64)

    # compare outputs of hidden layers
    leaky_relu_ops = [v for k, v in operators.items()
                      if (isinstance(v, ng.conv2d) and
                          isinstance(v.act_func, ng.leaky_relu_base))]
    leaky_relu_ops = list(sorted(set(leaky_relu_ops), key=leaky_relu_ops.index))

    conv2d_ops = [v for k, v in operators.items()
                  if (isinstance(v, ng.conv2d) and v.act_func is None)]
    conv2d_ops = list(sorted(set(conv2d_ops), key=conv2d_ops.index))

    # only 1st output
    sub_ops = leaky_relu_ops[:9] + conv2d_ops[:1]
    sub_outs = ng.eval(sub_ops, act=vact)
    sub_outs = [sub_out.transpose([0, 3, 1, 2]) for sub_out in sub_outs]
    sub_scale_factors = [sub_op.scale_factor for sub_op in sub_ops]

    model.eval()
    mouts = []
    # all Conv2d-LeakyReLU layers before YOLOLayer
    mouts.append(nn.Sequential(model.module_list[0])(
        torch.from_numpy(model_input)).detach().numpy())
    mouts.append(nn.Sequential(*model.module_list[0:3])(
        torch.from_numpy(model_input)).detach().numpy())
    mouts.append(nn.Sequential(*model.module_list[0:5])(
        torch.from_numpy(model_input)).detach().numpy())
    mouts.append(nn.Sequential(*model.module_list[0:7])(
        torch.from_numpy(model_input)).detach().numpy())
    mouts.append(nn.Sequential(*model.module_list[0:9])(
        torch.from_numpy(model_input)).detach().numpy())
    mouts.append(nn.Sequential(*model.module_list[0:11])(
        torch.from_numpy(model_input)).detach().numpy())
    mouts.append(nn.Sequential(*model.module_list[0:13])(
        torch.from_numpy(model_input)).detach().numpy())
    mouts.append(nn.Sequential(*model.module_list[0:14])(
        torch.from_numpy(model_input)).detach().numpy())
    mouts.append(nn.Sequential(*model.module_list[0:15])(
        torch.from_numpy(model_input)).detach().numpy())
    mouts.append(nn.Sequential(*model.module_list[0:16])(
        torch.from_numpy(model_input)).detach().numpy())

    scaled_mouts = [mout * scale_factor
                    for mout, scale_factor in zip(mouts, sub_scale_factors)]

    sub_mean_square_errors = [np.sum((sub_out - mout) ** 2) / sub_out.size
                              for mout, sub_out in zip(scaled_mouts, sub_outs)]
    sub_corrcoefs = [np.corrcoef(mout.reshape([-1]), sub_out.reshape([-1]))
                     for mout, sub_out in zip(mouts, sub_outs)]

    # compare prediction results
    vouts = ng.eval(outs, act=vact)

    mean_square_errors = [np.sum((vout - scaled_model_out) ** 2) / vout.size
                          for vout, scaled_model_out in zip(vouts, scaled_model_outs)]
    corrcoefs = [np.corrcoef(model_out.reshape([-1]), vout.reshape([-1]))
                 for model_out, vout in zip(model_outs, vouts)]

    # breakpoint()

    # --------------------
    # (5) Convert the NNgen dataflow to a hardware description (Verilog HDL and IP-XACT)
    # --------------------

    # to Veriloggen object
    # targ = ng.to_veriloggen([out], 'yolov3tiny', silent=silent,
    #                        config={'maxi_datawidth': axi_datawidth})

    # to IP-XACT (the method returns Veriloggen object, as well as to_veriloggen)
    targ = ng.to_ipxact([out], 'yolov3tiny', silent=silent,
                        config={'maxi_datawidth': axi_datawidth})

    # to Verilog HDL RTL (the method returns a source code text)
    # rtl = ng.to_verilog([out], 'yolov3tiny', silent=silent,
    #                    config={'maxi_datawidth': axi_datawidth})

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
    # mem = np.zeros([1024 * 1024 * 256 // (memimg_datawidth // 8)], dtype=np.int64)
    mem = np.zeros([1024 * 1024 * 1024 // (memimg_datawidth // 8)], dtype=np.int16)
    mem = mem + [100]

    # placeholder
    axi.set_memory(mem, vact, memimg_datawidth,
                   act_dtype.width, act.addr,
                   max(int(math.ceil(axi_datawidth / act_dtype.width)), conv2d_par_ich))

    # parameters (variable and constant)
    axi.set_memory(mem, param_data, memimg_datawidth,
                   8, variable_addr)

    # verification data
    axi.set_memory(mem, vout, memimg_datawidth,
                   act_dtype.width, check_addr,
                   max(int(math.ceil(axi_datawidth / act_dtype.width)), conv2d_par_och))

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
            for x in range(out.shape[1]):
                orig = memory.read_word(bat * out.aligned_shape[1] + x,
                                        out.addr, act_dtype.width)
                check = memory.read_word(bat * out.aligned_shape[1] + x,
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
    lines = rslt.splitlines()
    if simtype == 'verilator' and lines[-1].startswith('-'):
        rslt = '\n'.join(lines[:-1])
    return rslt


if __name__ == '__main__':
    rslt = run(silent=False, filename='tmp.v')
    print(rslt)
