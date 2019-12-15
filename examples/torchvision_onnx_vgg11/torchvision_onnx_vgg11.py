from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import math
import numpy as np
import PIL
import json

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


def run(act_dtype=ng.int8, weight_dtype=ng.int8,
        bias_dtype=ng.int32, scale_dtype=ng.int8,
        with_batchnorm=True, disable_fusion=False,
        conv2d_par_ich=1, conv2d_par_och=1, conv2d_par_col=1, conv2d_par_row=1,
        conv2d_concur_och=None, conv2d_stationary='filter',
        pool_par=1, elem_par=1,
        chunk_size=64, axi_datawidth=32, silent=False,
        filename=None,
        simtype='iverilog',
        # simtype='verilator',
        # simtype=None,  # no RTL simulation
        outputfile=None):

    # input mean and standard deviation
    imagenet_mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225]).astype(np.float32)

    act_shape = (1, 224, 224, 3)

    # pytorch model
    if with_batchnorm:
        model = torchvision.models.vgg11_bn(pretrained=True)
    else:
        model = torchvision.models.vgg11(pretrained=True)

    # Pytorch to ONNX
    onnx_filename = 'vgg11_imagenet.onnx'
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
    out = outputs['out']

    # verification data
    img = np.array(PIL.Image.open('car.png').convert('RGB')).astype(np.float32)
    img = img.reshape([1] + list(img.shape))

    img = img / 255
    img = (img - imagenet_mean) / imagenet_std

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

    # compare behaviors of hidden layers
    # features (conv2d)
    conv2d_ops = [v for k, v in operators.items()
                  if isinstance(v, ng.conv2d) and not isinstance(v, ng.matmul)]
    conv2d_ops = list(sorted(set(conv2d_ops), key=conv2d_ops.index))
    conv2d_outs = ng.eval(conv2d_ops, act=vact)
    conv2d_scale_factors = [op.scale_factor for op in conv2d_ops]

    model_features_relu_layers = [layer
                                  for layer in model.features if isinstance(layer, nn.ReLU)]
    model_features_relu_indexes = [list(model.features).index(layer)
                                   for layer in model_features_relu_layers]
    model_features_seqs = [model.features[:i + 1]
                           for i in model_features_relu_indexes]
    for sub in model_features_seqs:
        sub.eval()
    model_features_outs = [sub(torch.from_numpy(model_input)).detach().numpy()
                           for sub in model_features_seqs]

    scaled_features_outs = [(out * scale_factor)
                            for out, scale_factor in zip(model_features_outs, conv2d_scale_factors)]
    features_error_rates = [(np.sum(np.abs(conv2d_out.transpose([0, 3, 1, 2]) - scaled_features_out)) /
                             np.sum(scaled_features_out))
                            for conv2d_out, scaled_features_out in zip(conv2d_outs, scaled_features_outs)]
    features_max_diffs = [scaled_features_out.max() / conv2d_out.max()
                          for conv2d_out, scaled_features_out in zip(conv2d_outs, scaled_features_outs)]
    features_corrcoefs = [np.corrcoef(model_features_out.reshape([-1]),
                                      conv2d_out.transpose([0, 3, 1, 2]).reshape([-1]))
                          for conv2d_out, model_features_out in zip(conv2d_outs, model_features_outs)]

    # avgpool (avg_pool)
    avg_pool_ops = [v for k, v in operators.items()
                    if isinstance(v, (ng.avg_pool, ng.avg_pool_serial))]
    avg_pool_ops = list(sorted(set(avg_pool_ops), key=avg_pool_ops.index))
    avg_pool_outs = ng.eval(avg_pool_ops, act=vact)
    avg_pool_scale_factors = [op.scale_factor for op in avg_pool_ops]

    model_avgpool_seqs = [nn.Sequential(model.features, model.avgpool)]
    for sub in model_avgpool_seqs:
        sub.eval()
    model_avgpool_outs = [sub(torch.from_numpy(model_input)).detach().numpy()
                          for sub in model_avgpool_seqs]

    scaled_avgpool_outs = [(out * scale_factor)
                           for out, scale_factor in zip(model_avgpool_outs, avg_pool_scale_factors)]
    avgpool_error_rates = [(np.sum(np.abs(avg_pool_out.transpose([0, 3, 1, 2]) - scaled_avgpool_out)) /
                            np.sum(scaled_avgpool_out))
                           for avg_pool_out, scaled_avgpool_out in zip(avg_pool_outs, scaled_avgpool_outs)]
    avgpool_max_diffs = [scaled_avgpool_out.max() / avg_pool_out.max()
                         for avg_pool_out, scaled_avgpool_out in zip(avg_pool_outs, scaled_avgpool_outs)]
    avgpool_corrcoefs = [np.corrcoef(model_avgpool_out.reshape([-1]),
                                     avg_pool_out.transpose([0, 3, 1, 2]).reshape([-1]))
                         for avg_pool_out, model_avgpool_out in zip(avg_pool_outs, model_avgpool_outs)]

    # classifier (matmul)
    matmul_ops = [v for k, v in operators.items() if isinstance(v, ng.matmul)]
    matmul_ops = list(sorted(set(matmul_ops), key=matmul_ops.index))
    matmul_outs = ng.eval(matmul_ops, act=vact)
    matmul_scale_factors = [op.scale_factor for op in matmul_ops]

    class Flatten(nn.Module):
        def forward(self, input):
            return input.view(input.size(0), -1)

    model_classifier_relu_layers = [layer
                                    for layer in model.classifier if isinstance(layer, nn.ReLU)]
    model_classifier_relu_indexes = [list(model.classifier).index(layer)
                                     for layer in model_classifier_relu_layers]
    model_classifier_relu_indexes.append(len(model.classifier))
    model_classifier_seqs = [nn.Sequential(model.features, model.avgpool, Flatten(), model.classifier[:i + 1])
                             for i in model_classifier_relu_indexes]
    for sub in model_classifier_seqs:
        sub.eval()
    model_classifier_outs = [sub(torch.from_numpy(model_input)).detach().numpy()
                             for sub in model_classifier_seqs]

    scaled_classifier_outs = [(out * scale_factor)
                              for out, scale_factor in zip(model_classifier_outs, matmul_scale_factors)]
    classifier_error_rates = [(np.sum(np.abs(matmul_out - scaled_classifier_out)) /
                               np.sum(scaled_classifier_out))
                              for matmul_out, scaled_classifier_out in zip(matmul_outs, scaled_classifier_outs)]
    classifier_max_diffs = [scaled_classifier_out.max() / matmul_out.max()
                            for matmul_out, scaled_classifier_out in zip(matmul_outs, scaled_classifier_outs)]
    classifier_corrcoefs = [np.corrcoef(model_classifier_out.reshape([-1]),
                                        matmul_out.reshape([-1]))
                            for matmul_out, model_classifier_out in zip(matmul_outs, model_classifier_outs)]

    # compare prediction results
    eval_outs = ng.eval([out], act=vact)
    vout = eval_outs[0]

    class_index = json.load(open('imagenet_class_index.json', 'r'))
    labels = {int(key): value for (key, value) in class_index.items()}

    mout = scaled_model_out
    for bat in range(mout.shape[0]):
        for index, value in list(sorted(enumerate(mout[bat]),
                                        key=lambda x: x[1], reverse=True))[:10]:
            print("# mout: %s (%d) = %f" % (str(labels[index]), index, value))
        for index, value in list(sorted(enumerate(vout[bat]),
                                        key=lambda x: x[1], reverse=True))[:10]:
            print("# vout: %s (%d) = %d" % (str(labels[index]), index, value))

    # breakpoint()

    # --------------------
    # (5) Convert the NNgen dataflow to a hardware description (Verilog HDL and IP-XACT)
    # --------------------

    # to Veriloggen object
    # targ = ng.to_veriloggen([out], 'vgg11', silent=silent,
    #                        config={'maxi_datawidth': axi_datawidth})

    # to IP-XACT (the method returns Veriloggen object, as well as to_veriloggen)
    targ = ng.to_ipxact([out], 'vgg11', silent=silent,
                        config={'maxi_datawidth': axi_datawidth})

    # to Verilog HDL RTL (the method returns a source code text)
    # rtl = ng.to_verilog([out], 'vgg11', silent=silent,
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
