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
        onnx_filename='vgg11_imagenet.onnx',
        weight_filename='vgg11_imagenet.npy',
        verilog_filename=None,
        sim_filename=None,
        simtype=None,  # no RTL simulation
        # simtype='iverilog',
        # simtype='verilator'
        ):

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
    model_input = np.broadcast_to(img, act_shape)

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
    vact = np.broadcast_to(vact, act_shape)

    # compare outputs of hidden layers
    features_ops = [v for k, v in operators.items()
                    if isinstance(v, ng.conv2d) and not isinstance(v, ng.matmul)]
    features_ops = list(sorted(set(features_ops), key=features_ops.index))
    avgpool_op = [v for k, v in operators.items()
                  if isinstance(v, (ng.avg_pool, ng.avg_pool_serial))][0]
    classifier_ops = [v for k, v in operators.items() if isinstance(v, ng.matmul)]
    classifier_ops = list(sorted(set(classifier_ops), key=classifier_ops.index))

    sub_ops = features_ops + [avgpool_op] + classifier_ops
    sub_outs = ng.eval(sub_ops, act=vact)
    sub_outs = [sub_out.transpose([0, 3, 1, 2])
                for sub_out in sub_outs[:-len(classifier_ops)]] + sub_outs[-len(classifier_ops):]
    sub_scale_factors = ([features_op.scale_factor for features_op in features_ops]
                         + [avgpool_op.scale_factor]
                         + [classifier_op.scale_factor for classifier_op in classifier_ops])

    model.eval()
    model_features_relu_layers = [layer
                                  for layer in model.features if isinstance(layer, nn.ReLU)]
    model_features_relu_indexes = [list(model.features).index(layer)
                                   for layer in model_features_relu_layers]
    model_features_seqs = [model.features[:i + 1]
                           for i in model_features_relu_indexes]
    model_features_outs = [seq(torch.from_numpy(model_input)).detach().numpy()
                           for seq in model_features_seqs]

    model_avgpool_out = nn.Sequential(model.features,
                                      model.avgpool)(torch.from_numpy(model_input)).detach().numpy()

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
    model_classifier_outs = [seq(torch.from_numpy(model_input)).detach().numpy()
                             for seq in model_classifier_seqs]

    model_outs = model_features_outs + [model_avgpool_out] + model_classifier_outs
    scaled_outs = [model_out * scale_factor
                   for model_out, scale_factor in zip(model_outs, sub_scale_factors)]

    max_diffs = [model_out.max() / sub_out.max()
                 for model_out, sub_out in zip(scaled_outs, sub_outs)]
    overflows = [np.sum(np.abs(sub_out) >= abs(2 ** (sub_op.dtype.width - 1) - 1))
                 for sub_op, sub_out in zip(sub_ops, sub_outs)]
    mean_square_errors = [np.sum((sub_out - model_out) ** 2) / sub_out.size
                          for model_out, sub_out in zip(scaled_outs, sub_outs)]
    corrcoefs = [np.corrcoef(model_out.reshape([-1]), sub_out.reshape([-1]))
                 for model_out, sub_out in zip(model_outs, sub_outs)]

    # compare prediction results
    eval_outs = ng.eval([out], act=vact)
    vout = eval_outs[0]

    mean_square_error = np.sum((vout - scaled_model_out) ** 2) / vout.size
    corrcoef = np.corrcoef(model_out.reshape([-1]), vout.reshape([-1]))

    class_index = json.load(open('imagenet_class_index.json', 'r'))
    labels = {int(key): value for (key, value) in class_index.items()}

    mout = scaled_model_out
    for bat in range(mout.shape[0]):
        m_top10 = list(sorted(enumerate(mout[bat]), key=lambda x: x[1], reverse=True))[:10]
        m_top10_indexes = [index for index, value in m_top10]
        v_top10 = list(sorted(enumerate(vout[bat]), key=lambda x: x[1], reverse=True))[:10]
        v_top10_indexes = [index for index, value in v_top10]
        num_hit = 0
        score = 0
        for index, value in m_top10:
            print("# mout: %s (%d) = %f" % (str(labels[index]), index, value))
        for index, value in v_top10:
            print("# vout: %s (%d) = %d" % (str(labels[index]), index, value))
            if index in m_top10_indexes:
                num_hit += 1
                score += 10 - abs(m_top10_indexes.index(index) - v_top10_indexes.index(index))
        print("# top-10 hit: %d" % num_hit)
        print("# top-10 score: %d" % score)

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
    # (6) Save the quantized weights
    # --------------------
    param_data = ng.export_ndarray([out], chunk_size)
    param_bytes = len(param_data)
    np.save(weight_filename, param_data)

    # --------------------
    # (7) Simulate the generated hardware by Veriloggen and Verilog simulator
    # --------------------

    if simtype is None:
        sys.exit()

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
