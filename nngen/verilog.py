from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import collections
import copy
import inspect
import types

import veriloggen as vg
import veriloggen.types.axi as axi
import veriloggen.thread as vthread
import veriloggen.types.ipxact as ipxact

from . import basic_types as bt
from . import dtype_list
from . import scheduler
from . import version
from . import substreams


default_config = {
    # clock, reset
    'clock_name': 'CLK',
    'reset_name': 'RESETN',
    'reset_polarity': 'low',

    # measure enable main_fsm
    'measurable_main_fsm': True,

    # interrupt
    'interrupt_enable': True,
    'interrupt_name': 'irq',

    # AXI
    'axi_coherent': False,
    'maxi_datawidth': 32,
    'maxi_addrwidth': 32,
    'saxi_datawidth': 32,
    'saxi_addrwidth': 32,
    'req_fifo_addrwidth': 3,

    # address map
    'default_global_addr_offset': 0,
    'use_map_ram': False,  # using a address map RAM instead of AXIS registers
    'use_map_reg': False,  # using dedicated address registers instead of unified register for variable and constant

    # control params
    'use_param_ram': False,  # using a RAM instead of MUX for control params.
    'min_param_ram_len': 0,  # if use_param_ram == True and num_of_params < min_param_ram_len,
                             # a control param RAM is created instead of MUX

    # default parameters
    'default_datawidth': 32,
    'min_onchip_ram_capacity': 32 * 128,
    'offchipram_chunk_bytes': 64,
    'max_parallel_ops': 1,

    # RAM style annotation
    'onchip_ram_style': None,  # '(* ram_style = "block" *)' for Xilinx
    'param_ram_style': None,  # '(* ram_style = "block" *)' for Xilinx
    'map_ram_style': None,  # '(* ram_style = "block" *)' for Xilinx

    'onchip_ram_priority': 'total_size',
    # 'onchip_ram_priority': 'max_size',
    # 'onchip_ram_priority': (lambda cur, width, length, num: cur + width * length * num),

    # for debug
    'fsm_as_module': False,
    'disable_stream_cache': False,
    'disable_control_cache': False,
    'dump_stream': False,
    'dump_stream_base': 10,
}

max_burst_length = 256

num_header_regs = 4
header_reg = 0  # index used in "sim.py"

# control_reg_global_offset must be same as REG_GLOBAL_OFFSET in pynq/nngen_ctrl.py'.
control_reg_global_offset = 32
num_control_regs = control_reg_global_offset + 1 - num_header_regs

num_control_regs = 33 - num_header_regs
control_reg_start = num_header_regs + 0
control_reg_busy = num_header_regs + 1
control_reg_reset = num_header_regs + 2
control_reg_extern_send = num_header_regs + 3
control_reg_extern_recv = num_header_regs + 4

control_reg_interrupt_isr = num_header_regs + 5
control_reg_interrupt_isr_busy = 0  # bit-field index in ISR
control_reg_interrupt_isr_extern = 1  # bit-field index in ISR
control_reg_interrupt_ier = num_header_regs + 6
control_reg_interrupt_iar = num_header_regs + 7

control_reg_count = num_header_regs + 8
control_reg_count_state = num_header_regs + 9
control_reg_count_div = num_header_regs + 10

control_reg_reserved = num_header_regs + 11  # head of reserved region

control_reg_address_amount = control_reg_global_offset - 1
control_reg_global_addr = control_reg_global_offset + 1

# when config['use_map_ram'] is True
num_addr_map_regs = 3
control_reg_load_global_addr_map = num_header_regs + num_control_regs
control_reg_busy_global_addr_map = num_header_regs + num_control_regs + 1
control_reg_addr_global_addr_map = num_header_regs + num_control_regs + 2


def to_veriloggen(objs, name, config=None, silent=False):

    config = load_default_config(config)

    m = _to_veriloggen_module(objs, name, config, silent=silent)

    return m


def to_verilog(objs, name, filename=None, config=None, silent=False):

    config = load_default_config(config)

    m = _to_veriloggen_module(objs, name, config,
                              silent=silent, where_from='to_verilog', output=filename)

    verilog_code = m.to_verilog(filename)

    return verilog_code


def to_ipxact(objs, name, ipname=None, config=None, silent=False):

    if ipname is None:
        ipname = name

    config = load_default_config(config)

    m = _to_veriloggen_module(objs, name, config,
                              silent=silent, where_from='to_ipxact', output=ipname)

    clk_name = config['clock_name']
    rst_name = config['reset_name']
    rst_polarity = ('ACTIVE_LOW'
                    if 'low' in config['reset_polarity'] else
                    'ACTIVE_HIGH')
    irq_name = config['interrupt_name']
    irq_sensitivity = 'LEVEL_HIGH'

    clk_ports = [(clk_name, (rst_name,))]
    rst_ports = [(rst_name, rst_polarity)]
    irq_ports = [(irq_name, irq_sensitivity)] if config['interrupt_enable'] else []

    ipxact.to_ipxact(m, ipname,
                     clk_ports=clk_ports,
                     rst_ports=rst_ports,
                     irq_ports=irq_ports)

    return m


def load_default_config(config=None):
    my_config = copy.copy(default_config)

    if config is not None:
        my_config.update(config)

    config = my_config

    return config


def _to_veriloggen_module(objs, name, config=None,
                          silent=False, where_from=None, output=None):

    if not isinstance(objs, (list, tuple)):
        objs = [objs]

    (objs, num_storages,
     num_input_storages, num_output_storages) = analyze(config, objs)
    m, clk, rst, maxi, saxi = make_module(config, name, objs,
                                          num_storages, num_input_storages,
                                          num_output_storages)

    schedule_table = schedule(config, objs)

    header_info = make_header_addr_map(config, saxi)

    (ram_dict, substrm_dict, ram_set_cache,
     stream_cache, control_cache, main_fsm,
     global_map_info, global_mem_map) = allocate(config, m, clk, rst,
                                                 maxi, saxi, objs, schedule_table)

    reg_map = make_reg_map(config, global_map_info, header_info)

    saxi.register[control_reg_address_amount].initval = address_space_amount(global_mem_map)

    if not silent:
        dump_config(config, where_from, output)
        dump_schedule_table(schedule_table)
        dump_rams(ram_dict)
        dump_substreams(substrm_dict)
        dump_streams(stream_cache)
        dump_main_fsm(main_fsm)
        dump_controls(control_cache, main_fsm)
        dump_register_map(reg_map)
        dump_memory_map(global_mem_map)

    return m


def address_space_amount(mem_map):
    max_gaddr = 0
    min_gaddr = 0
    for start, end in mem_map.keys():
        if start <= end:
            max_gaddr = max(max_gaddr, start, end)
            min_gaddr = min(min_gaddr, start, end)

    num_bytes = max_gaddr - min_gaddr + 1

    return num_bytes


def analyze(config, objs):
    set_output(objs)

    objs = collect_numerics(objs)
    num_storages = count_storages(objs)
    num_input_storages = count_input_storages(objs)
    num_output_storages = count_output_storages(objs)

    set_default_dtype(config, objs)

    return objs, num_storages, num_input_storages, num_output_storages


def set_output(objs):
    for obj in objs:
        obj.set_output()


def collect_numerics(objs):
    ret = set()
    for obj in objs:
        if obj not in ret:
            ret.update(obj._collect_numerics())

    ret = sorted(list(ret), key=lambda x:x.object_id)
    return ret


def count_storages(objs):
    count = 0
    for obj in objs:
        if obj.is_output:
            count += 1
        elif bt.is_storage(obj):
            count += 1
    return count


def count_input_storages(objs):
    count = 0
    for obj in objs:
        if bt.is_input_storage(obj):
            count += 1
    return count


def count_output_storages(objs):
    count = 0
    for obj in objs:
        if obj.is_output:
            count += 1
    return count


def set_default_dtype(config, objs):
    default_datawidth = config['default_datawidth']
    for obj in objs:
        if obj.dtype is None:
            obj.dtype = dtype_list.dtype_info('int', default_datawidth)


def sence_edge(m, clk, wire, rst=None, mode='posedge', name='sence'):
    tmp_reg = m.TmpRegLike(wire, prefix=name, initval=0)
    edge_seq = vg.TmpSeq(m, clk, rst, prefix='sence_edge')
    edge_seq(tmp_reg(wire))

    edge = m.TmpWireLike(wire, prefix=name)
    if mode == 'posedge':
        edge.assign(vg.And(vg.Not(tmp_reg), wire))
    elif mode == 'negedge':
        edge.assign(vg.And(tmp_reg, vg.Not(wire)))
    else:
        raise ValueError(mode)

    return edge


def add_irq(m, clk, rst, saxi, bit_field, target_signal, mode='posedge', name=None):
    irq_target = sence_edge(m, clk, target_signal, rst, mode=mode, name=name)
    saxi.seq.If(irq_target)(
        saxi.register[control_reg_interrupt_isr][bit_field](irq_target)
    )


def make_module(config, name, objs, num_storages, num_input_storages, num_output_storages):
    m = vg.Module(name)

    clk = m.Input(config['clock_name'])

    if 'low' in config['reset_polarity'].lower():
        rst_seq = vg.Seq(m, 'rst_seq', clk)
        rst_x = m.Input(config['reset_name'])
        rst_x_inv = m.Wire(config['reset_name'] + '_inv')
        rst_x_inv.assign(vg.Not(rst_x))
        rst = m.Wire(config['reset_name'] + '_inv_buf')
        rst.assign(rst_seq.Prev(rst_x_inv, 2))

    else:
        rst = m.Input(config['reset_name'])

    if config['interrupt_enable']:
        irq = m.OutputReg(config['interrupt_name'], initval=0)

    datawidth = config['maxi_datawidth']
    addrwidth = config['maxi_addrwidth']
    cache_mode = axi.AxCACHE_COHERENT if config['axi_coherent'] else axi.AxCACHE_NONCOHERENT
    prot_mode = axi.AxPROT_COHERENT if config['axi_coherent'] else axi.AxPROT_NONCOHERENT
    user_mode = axi.AxUSER_COHERENT if config['axi_coherent'] else axi.AxUSER_NONCOHERENT

    maxi = vthread.AXIM(m, 'maxi', clk, rst, datawidth, addrwidth,
                        waddr_cache_mode=cache_mode, raddr_cache_mode=cache_mode,
                        waddr_prot_mode=prot_mode, raddr_prot_mode=prot_mode,
                        waddr_user_mode=user_mode, raddr_user_mode=user_mode,
                        use_global_base_addr=True,
                        req_fifo_addrwidth=config['req_fifo_addrwidth'],
                        fsm_as_module=config['fsm_as_module'])

    datawidth = config['saxi_datawidth']
    addrwidth = config['saxi_addrwidth']

    num_unified_storages = 1
    num_temporal_storages = 1

    if config['use_map_ram']:
        length = num_header_regs + num_control_regs + num_addr_map_regs
    elif not config['use_map_reg']:
        length = (num_header_regs + num_control_regs +
                  num_temporal_storages + num_input_storages + num_output_storages +
                  num_unified_storages)
    else:
        length = (num_header_regs + num_control_regs +
                  num_temporal_storages + num_storages)

    saxi = vthread.AXISLiteRegister(m, 'saxi', clk, rst,
                                    datawidth, addrwidth, length=length,
                                    fsm_as_module=config['fsm_as_module'])

    maxi_idle = m.Wire('maxi_idle')
    maxi_idle.assign(vg.And(maxi.write_idle, maxi.read_idle))

    sw_rst_logic = m.Wire('sw_rst_logic')
    sw_rst_logic.assign(vg.And(maxi_idle, saxi.register[control_reg_reset]))

    rst_logic = m.Wire('rst_logic')
    rst_logic.assign(vg.Or(rst, sw_rst_logic))

    sys_rst_seq = vg.Seq(m, 'sys_rst_seq', clk)
    sys_rst = m.Reg('RST')
    v = rst_logic
    for i in range(2):
        v = vg.Or(v, sys_rst_seq.Prev(rst_logic, i + 1))
    sys_rst_seq(
        sys_rst(v)
    )

    if config['interrupt_enable']:
        irq_tmp = m.TmpWireLike(saxi.register[control_reg_interrupt_isr], prefix="irq")
        irq_tmp.assign(vg.And(saxi.register[control_reg_interrupt_isr],
                              saxi.register[control_reg_interrupt_ier]))
        irq_seq = vg.Seq(m, 'interrupt_seq', clk, rst)
        irq_seq(
            irq(vg.Uor(irq_tmp))
        )

        list(map(lambda ack, stat: saxi.seq.If(ack == 1)(ack(0), stat(0)),
                 saxi.register[control_reg_interrupt_iar], saxi.register[control_reg_interrupt_isr]))

        irq_busy = m.Wire("irq_busy")
        irq_busy.assign(saxi.register[control_reg_busy][0])
        add_irq(m, clk, rst, saxi, control_reg_interrupt_isr_busy,
                irq_busy, mode='negedge', name="irq_busy_edge")

        irq_extern = m.Wire("irq_extern")
        irq_extern.assign(vg.Uor(saxi.register[control_reg_extern_send]))
        add_irq(m, clk, rst, saxi, control_reg_interrupt_isr_extern,
                irq_extern, name="irq_extern_edge")

    for obj in objs:
        obj.set_module_info(m, clk, sys_rst, maxi, saxi)

    return m, clk, sys_rst, maxi, saxi


def schedule(config, objs):
    s = scheduler.OperationScheduler(config)
    s.schedule(objs)
    return s.result


def make_header_addr_map(config, saxi):
    header_info = collections.OrderedDict()
    header_regs = saxi.register[:num_header_regs]

    for i in range(num_header_regs):
        name = 'header%d' % i
        initval = config[name] if name in config else 0
        header_regs[i].initval = initval
        header_info[i] = 'header{} (default: 0x{:08x})'.format(i, initval)

    return header_info


def allocate(config, m, clk, rst, maxi, saxi, objs, schedule_table):
    set_storage_name(objs)
    merge_shared_attrs(objs)

    max_stream_rams = calc_max_stream_rams(config, schedule_table)
    ram_dict = make_rams(config, m, clk, rst, maxi, schedule_table, max_stream_rams)
    ram_set_cache = make_ram_sets(config, schedule_table, ram_dict, max_stream_rams)

    control_param_dict = make_control_params(config, schedule_table)

    substrm_dict = make_substreams(config, m, clk, rst, maxi, schedule_table)
    stream_cache = make_streams(config, schedule_table, ram_dict, substrm_dict)

    (global_addr_map, local_addr_map,
     global_map_info, global_mem_map) = make_addr_map(config, objs, saxi)

    if config['use_map_ram']:
        global_map_ram, local_map_ram = make_addr_map_rams(config, m, clk, rst, maxi,
                                                           global_addr_map, local_addr_map)
    else:
        global_map_ram = None
        local_map_ram = None

    control_cache, main_fsm = make_controls(
        config, m, clk, rst, maxi, saxi,
        schedule_table, control_param_dict,
        global_addr_map, local_addr_map,
        global_map_ram, local_map_ram)

    return (ram_dict, substrm_dict, ram_set_cache, stream_cache, control_cache,
            main_fsm, global_map_info, global_mem_map)


def set_storage_name(objs):
    tmp_input = 0
    tmp_output = 0
    for obj in objs:
        if bt.is_storage(obj) and obj.name is None:
            obj.name = 'input_%d' % tmp_input
            tmp_input += 1
        elif obj.is_output and obj.name is None:
            obj.name = 'output_%s_%d' % (obj.__class__.__name__, tmp_output)
            tmp_output += 1


def merge_shared_attrs(objs):
    repr_obj_dict = collections.OrderedDict()

    for obj in objs:
        if not bt.is_operator(obj):
            continue

        stream_hash = obj.get_stream_hash()
        if stream_hash not in repr_obj_dict:
            obj.merge_shared_attrs(obj)
            repr_obj_dict[stream_hash] = obj
            continue

        orig = repr_obj_dict[stream_hash]
        orig.merge_shared_attrs(obj)


def calc_max_stream_rams(config, schedule_table):
    max_stream_rams = {}  # key: stream_hash, value: max RAM sizes

    for stage, objs in sorted(schedule_table.items(), key=lambda x: x[0]):

        for obj in objs:
            if not bt.is_operator(obj):
                continue

            if (bt.is_output_chainable_operator(obj) and
                    not obj.chain_head):
                continue

            input_rams, output_rams, temp_rams = obj.get_required_rams()
            stream_hash = obj.get_stream_hash()

            if stream_hash not in max_stream_rams:
                max_stream_rams[stream_hash] = (input_rams, output_rams, temp_rams)
            else:
                prev_input_rams, prev_output_rams, prev_temp_rams = max_stream_rams[stream_hash]
                new_input_rams = tuple([max_tuple(prev, cur)
                                        for prev, cur in zip(prev_input_rams, input_rams)])
                new_output_rams = tuple([max_tuple(prev, cur)
                                         for prev, cur in zip(prev_output_rams, output_rams)])
                new_temp_rams = tuple([max_tuple(prev, cur)
                                       for prev, cur in zip(prev_temp_rams, temp_rams)])
                max_stream_rams[stream_hash] = (new_input_rams, new_output_rams, new_temp_rams)

    return max_stream_rams


def make_rams(config, m, clk, rst, maxi, schedule_table, max_stream_rams):
    max_rams = calc_max_rams(config, schedule_table, max_stream_rams)

    ram_dict = collections.defaultdict(list)
    ram_index = collections.defaultdict(int)

    for (width, length), num in reversed(sorted(max_rams.items(),
                                                key=lambda x: x[0][0] * x[0][1])):
        for _ in range(num):
            datawidth = width
            numports = 2
            numbanks = maxi.datawidth // width

            key = (width, length)
            i = ram_index[key]
            ram_index[key] += 1

            name = 'ram_w%d_l%d_id%d' % (width, length, i)

            if numbanks <= 1:
                addrwidth = int(math.ceil(math.log(length, 2)))
                ram = vthread.RAM(m, name, clk, rst,
                                  datawidth, addrwidth, numports=numports,
                                  ram_style=config['onchip_ram_style'])
            else:
                addrwidth = int(math.ceil(math.log(length // numbanks, 2)))
                ram = vthread.MultibankRAM(m, name, clk, rst,
                                           datawidth, addrwidth, numports=numports,
                                           numbanks=numbanks,
                                           ram_style=config['onchip_ram_style'])

            ram_dict[key].append(ram)

    return ram_dict


def calc_max_rams(config, schedule_table, max_stream_rams):
    max_stage_rams = calc_max_stage_rams(config, schedule_table, max_stream_rams)

    priority = collections.Counter()

    if config['onchip_ram_priority'] == 'total_size':
        method = (lambda cur, width, length, num:
                  cur + width * length * num)
    elif config['onchip_ram_priority'] == 'max_size':
        method = (lambda cur, width, length, num:
                  max(cur, width * length))
    elif callable(config['onchip_ram_priority']):
        method = config['onchip_ram_priority']
    else:
        raise ValueError("Unsupported onchip_ram_priority '%s'" %
                         str(config['onchip_ram_priority']))

    for stage, rams in sorted(max_stage_rams.items(), key=lambda x: x[0]):
        for (width, length), num in rams.items():
            priority[stage] = method(priority[stage], width, length, num)

    max_rams = collections.Counter()

    for stage, rams in sorted(max_stage_rams.items(),
                              key=lambda x: priority[x[0]],
                              reverse=True):

        used_count = collections.Counter()

        for (width, length), num in sorted(rams.items(),
                                           key=lambda x: x[0][0] * x[0][1],
                                           reverse=True):

            for (mw, ml), mn in sorted(max_rams.items(),
                                       key=lambda x: x[0][0] * x[0][1],
                                       reverse=True):
                if mw != width:
                    continue

                if ml >= length and used_count[(mw, ml)] + num <= mn:
                    used_count[(mw, ml)] += num
                    break

            else:
                max_rams[(width, length)] += num
                used_count[(width, length)] += num

    return max_rams


def calc_max_stage_rams(config, schedule_table, max_stream_rams):
    max_stage_rams = {}  # key: stage_number, value: RAM counter

    for stage, objs in sorted(schedule_table.items(), key=lambda x: x[0]):
        cnt = collections.Counter()

        for obj in objs:
            if not bt.is_operator(obj):
                continue

            if (bt.is_output_chainable_operator(obj) and
                    not obj.chain_head):
                continue

            stream_hash = obj.get_stream_hash()
            input_rams, output_rams, temp_rams = max_stream_rams[stream_hash]

            for width, length in sorted(input_rams, reverse=True):
                key = to_actual_ram_spec(config, width, length)
                cnt[key] += 1

            for width, length in sorted(output_rams, reverse=True):
                key = to_actual_ram_spec(config, width, length)
                cnt[key] += 1

            for width, length in sorted(temp_rams, reverse=True):
                key = to_actual_ram_spec(config, width, length)
                cnt[key] += 1

        max_stage_rams[stage] = cnt

    return max_stage_rams


def max_tuple(*tuples):
    return tuple([max(*values) for values in zip(*tuples)])


def make_ram_sets(config, schedule_table, ram_dict, max_stream_rams):
    ram_set_cache = collections.defaultdict(list)

    for stage, objs in sorted(schedule_table.items(), key=lambda x: x[0]):
        make_stage_ram_sets(config, schedule_table, ram_dict, max_stream_rams,
                            stage, objs, ram_set_cache)

    return ram_set_cache


def make_stage_ram_sets(config, schedule_table, ram_dict, max_stream_rams,
                        stage, objs, ram_set_cache):

    ram_index_set = collections.defaultdict(set)

    # cache check
    for obj in objs:
        if not bt.is_operator(obj):
            continue

        if bt.is_output_chainable_operator(obj) and not obj.chain_head:
            continue

        stream_hash = obj.get_stream_hash()

        for (input_rams, output_rams,
             temp_rams, used_ram_index_dict) in ram_set_cache[stream_hash]:

            satisfied = obj.check_ram_requirements(input_rams,
                                                   output_rams, temp_rams)

            for key, ram_indexes in used_ram_index_dict.items():
                for ram_index in ram_indexes:
                    if ram_index in ram_index_set[key]:
                        satisfied = False

            # Hit: reuse the existing RAM set
            if satisfied:
                obj.set_rams(input_rams, output_rams, temp_rams)
                obj.cached_ram_set = True

                for key, ram_indexes in used_ram_index_dict.items():
                    for ram_index in ram_indexes:
                        ram_index_set[key].add(ram_index)

    # Miss or Unsatisfied: create a new RAM set
    for obj in objs:
        if not bt.is_operator(obj):
            continue

        if bt.is_output_chainable_operator(obj) and not obj.chain_head:
            continue

        if obj.cached_ram_set:
            continue

        stream_hash = obj.get_stream_hash()
        req_inputs, req_outputs, req_temps = max_stream_rams[stream_hash]
        used_ram_index_dict = collections.defaultdict(list)

        req_rams = []
        req_rams.extend([(width, length, 'input', i)
                         for i, (width, length) in enumerate(req_inputs)])
        req_rams.extend([(width, length, 'output', i)
                         for i, (width, length) in enumerate(req_outputs)])
        req_rams.extend([(width, length, 'temp', i)
                         for i, (width, length) in enumerate(req_temps)])

        input_rams = [None for _ in req_inputs]
        output_rams = [None for _ in req_outputs]
        temp_rams = [None for _ in req_temps]

        # smallest request first
        for width, length, ram_type, pos in sorted(req_rams,
                                                   key=lambda x: (x[0], x[1])):
            width, length = to_actual_ram_spec(config, width, length)
            key = (width, length)

            found = False
            # smallest RAM first
            for ram_key, rams in sorted(ram_dict.items(), key=lambda x: x[0]):
                if found:
                    break

                for i, ram in enumerate(rams):
                    if i in ram_index_set[ram_key]:
                        continue

                    ram_width, ram_length = ram_key

                    if ram_width != width:
                        continue

                    if ram_length >= length:
                        if ram_type == 'input':
                            input_rams[pos] = ram
                        elif ram_type == 'output':
                            output_rams[pos] = ram
                        elif ram_type == 'temp':
                            temp_rams[pos] = ram

                        ram_index_set[ram_key].add(i)
                        used_ram_index_dict[key].append(i)
                        found = True
                        break

        obj.set_rams(input_rams, output_rams, temp_rams)
        ram_set_cache[stream_hash].append((input_rams, output_rams,
                                           temp_rams, used_ram_index_dict))


def to_actual_ram_spec(config, width, length):
    if width == 0:
        width = config['default_datawidth']

    min_capacity = config['min_onchip_ram_capacity']
    maxi_datawidth = config['maxi_datawidth']
    numbanks = maxi_datawidth // width

    if numbanks < 1:
        numbanks = 1

    # while min_length is the total length of a MultibankMemory,
    # min_capacity is the minimum capacity for each bank
    min_length = (min_capacity // width) * numbanks

    if length < min_length:
        length = min_length

    addrwidth = int(math.ceil(math.log(length, 2)))
    real_length = 2 ** addrwidth

    return width, real_length


def make_control_params(config, schedule_table):

    control_param_dict = get_control_param_dict(schedule_table)
    obj_cache = collections.defaultdict(list)

    for stage, objs in sorted(schedule_table.items(), key=lambda x: x[0]):

        for obj in objs:
            if not bt.is_operator(obj):
                continue

            if bt.is_view(obj):
                continue

            if bt.is_removable_reshape(obj):
                continue

            if (bt.is_output_chainable_operator(obj) and
                    not obj.chain_head):
                continue

            key = obj.get_stream_hash()
            control_param_list = control_param_dict[key]
            control_param_len = len(control_param_list)

            if (not config['disable_control_cache'] and
                    obj.control_cachable and len(obj_cache[key]) > 0):
                # hit
                orig = obj_cache[key][0]
                obj.copy_control_params(orig)

            else:
                # miss
                width_dict, signed_dict = calc_control_param_width(control_param_list)
                obj.make_control_params(control_param_len, width_dict, signed_dict,
                                        use_param_ram=config['use_param_ram'],
                                        min_param_ram_len=config['min_param_ram_len'])
                obj.make_control_param_buf(control_param_list,
                                           use_param_ram=config['use_param_ram'],
                                           min_param_ram_len=config['min_param_ram_len'],
                                           ram_style=config['param_ram_style'])

                obj_cache[key].append(obj)

    return control_param_dict


def get_control_param_dict(schedule_table):
    index_dict = collections.defaultdict(int)
    control_param_dict = collections.defaultdict(list)

    for stage, objs in sorted(schedule_table.items(), key=lambda x: x[0]):

        for obj in objs:
            if not bt.is_operator(obj):
                continue

            if bt.is_view(obj):
                continue

            if bt.is_removable_reshape(obj):
                continue

            if (bt.is_output_chainable_operator(obj) and
                    not obj.chain_head):
                continue

            key = obj.get_stream_hash()
            index = index_dict[key]
            obj.set_control_param_index(index)
            index_dict[key] += 1

            values = obj.collect_all_control_param_values()
            control_param_dict[key].append(values)

    return control_param_dict


def calc_control_param_width(control_param_value_list):
    width_dict = collections.OrderedDict()
    signed_dict = collections.OrderedDict()

    for values in control_param_value_list:
        for name, value in values.items():
            if isinstance(value, (tuple, list)):
                if name in width_dict:
                    current_width = width_dict[name]
                else:
                    current_width = [1 for _ in value]
                if name in signed_dict:
                    current_signed = signed_dict[name]
                else:
                    current_signed = [False for _ in value]

                width = []
                signed = []
                for v, cw, cs in zip(value, current_width, current_signed):
                    w = max(abs(v).bit_length(), 1)
                    width.append(max(cw, w))
                    s = v < 0
                    signed.append(cs or s)

                width_dict[name] = width
                signed_dict[name] = signed

            else:
                if name in width_dict:
                    current_width = width_dict[name]
                else:
                    current_width = 1
                if name in signed_dict:
                    current_signed = signed_dict[name]
                else:
                    current_signed = False

                w = max(abs(value).bit_length(), 1)
                width = max(current_width, w)
                s = value < 0
                signed = current_signed or s

                width_dict[name] = width
                signed_dict[name] = signed

    for name, signed in signed_dict.items():
        if isinstance(width_dict[name], (tuple, list)):
            width = []
            for w, s in zip(width_dict[name], signed):
                if s:
                    width.append(w + 1)
                else:
                    width.append(w)
            width_dict[name] = width
        else:
            if signed:
                width_dict[name] += 1

    return width_dict, signed_dict


def make_substreams(config, m, clk, rst, maxi, schedule_table):
    max_substrms = calc_max_substreams(config, schedule_table)

    substrm_dict = collections.defaultdict(list)
    substrm_index = collections.defaultdict(int)

    for key, num in sorted(max_substrms.items(), key=lambda x: x[0]):
        method_name = key[0]
        args = key[1]
        method = getattr(substreams, method_name)

        for _ in range(num):
            i = substrm_index[key]
            substrm_index[key] += 1
            substrm = method(m, clk, rst, *args)
            substrm_dict[key].append(substrm)

    return substrm_dict


def calc_max_substreams(config, schedule_table):
    max_substrms = collections.Counter()

    for stage, objs in sorted(schedule_table.items(), key=lambda x: x[0]):
        cnt = collections.Counter()

        for obj in objs:
            if not bt.is_operator(obj):
                continue

            if (bt.is_output_chainable_operator(obj) and
                    not obj.chain_head):
                continue

            substrms = obj.get_required_substreams()

            for key in substrms:
                cnt[key] += 1

        for key, val in sorted(cnt.items(), key=lambda x: x[0], reverse=True):
            max_substrms[key] = max(max_substrms[key], val)

    return max_substrms


def to_hashable_dict(dct):
    return tuple([(key, val)
                  for key, val in sorted(dct.items(), key=lambda x:x[0])])


def from_hashable_dict(dct):
    return dict(dct)


def make_streams(config, schedule_table, ram_dict, substrm_dict):
    stream_cache = collections.defaultdict(list)

    for stage, objs in sorted(schedule_table.items(), key=lambda x: x[0]):
        make_stage_streams(config, schedule_table, ram_dict, substrm_dict,
                           stage, objs, stream_cache)

    return stream_cache


def make_stage_streams(config, schedule_table, ram_dict, substrm_dict,
                       stage, objs, stream_cache):

    substrm_index_set = collections.defaultdict(set)

    # cache check
    for obj in objs:
        if config['disable_stream_cache']:
            break

        if not bt.is_operator(obj):
            continue

        if bt.is_output_chainable_operator(obj) and not obj.chain_head:
            continue

        if not obj.stream_cachable:
            continue

        stream_hash = obj.get_stream_hash()

        for (strm, used_substrm_index_dict) in stream_cache[stream_hash]:

            satisfied = True

            for key, substrm_indexes in used_substrm_index_dict.items():
                for substrm_index in substrm_indexes:
                    if substrm_index in substrm_index_set:
                        satisfied = False

            # Hit: reuse the existing substream set and main stream
            if satisfied:
                obj.set_stream(strm)
                obj.cached_stream = True

                for key, substrm_indexes in used_substrm_index_dict.items():
                    for substrm_index in substrm_indexes:
                        substrm_index_set[key].add(substrm_index)

    # Miss: create a new substream set and main stream
    for obj in objs:
        if not bt.is_operator(obj):
            continue

        if bt.is_output_chainable_operator(obj) and not obj.chain_head:
            continue

        if obj.cached_stream:
            continue

        req_substrms = obj.get_required_substreams()
        used_substrm_index_dict = collections.defaultdict(list)

        substrms = []
        for key in req_substrms:

            substrm_list = substrm_dict[key]
            for i, substrm in enumerate(substrm_list):
                if i in substrm_index_set[key]:
                    continue

                substrm_index_set[key].add(i)
                used_substrm_index_dict[key].append(i)
                sub = substrm_list[i]
                substrms.append(sub)
                break

        obj.set_substreams(substrms)
        strm = obj.make_stream(datawidth=config['default_datawidth'],
                               fsm_as_module=config['fsm_as_module'],
                               dump=config['dump_stream'],
                               dump_base=config['dump_stream_base'])
        obj.set_stream(strm)

        stream_hash = obj.get_stream_hash()
        stream_cache[stream_hash].append((strm, used_substrm_index_dict))


def make_addr_map(config, objs, saxi):

    chunk_size = config['offchipram_chunk_bytes']

    maxi_datawidth = config['maxi_datawidth']
    if (chunk_size * 8) % maxi_datawidth != 0:
        raise ValueError("'offchipram_chunk_bytes' must be a multiple number of 'maxi_datawidth'.")

    global_addr_offset = config['default_global_addr_offset']

    global_addr_map = collections.OrderedDict()
    local_addr_map = collections.OrderedDict()
    global_map_info = collections.OrderedDict()
    global_mem_map = collections.OrderedDict()  # key: (start, end)

    if not config['use_map_ram']:
        map_regs = saxi.register[num_header_regs + num_control_regs:]

    offset_reg = saxi.register[control_reg_global_offset]
    offset_reg.initval = global_addr_offset

    reg_index = 1
    global_index = 1
    local_index = 1

    storage_used = 0
    unified_storage_used = 0
    temporal_used = 0

    unified_storage_list = []

    local_addr_map[0] = 0

    # output
    for obj in sorted(objs, key=lambda x: x.object_id):
        if obj.is_output and obj.global_index is None:

            orig_obj = obj
            while bt.is_view(obj) or bt.is_removable_reshape(obj):
                obj = obj.args[0]

            width = obj.dtype.width
            length = obj.get_aligned_length()
            space_size = align_space(width, length, chunk_size)
            default_global_addr = storage_used

            obj.set_global_index(global_index)
            obj.set_local_index(0)
            obj.set_default_global_addr(default_global_addr)
            obj.set_default_local_addr(0)

            i = (("output (%s) %s "
                  "(size: %s, dtype: %s, shape: %s, "
                  "alignment: %d words (%d bytes)), "
                  "aligned shape: %s") %
                 (orig_obj.__class__.__name__,
                  "'%s'" % orig_obj.name if orig_obj.name is not None else 'None',
                  size_str(space_size),
                  orig_obj.dtype.to_str() if orig_obj.dtype is not None else 'None',
                  (str(orig_obj.shape)
                   if isinstance(orig_obj.shape, (tuple, list)) else '()'),
                  orig_obj.get_word_alignment(),
                  bt.to_byte(orig_obj.get_word_alignment() * orig_obj.get_ram_width()),
                  (str(tuple(orig_obj.get_aligned_shape()))
                   if isinstance(orig_obj.shape, (tuple, list)) else '()')))

            global_mem_map[(default_global_addr,
                            default_global_addr + space_size - 1)] = i

            if config['use_map_ram']:
                global_addr_map[global_index] = default_global_addr

            if not config['use_map_ram']:
                map_regs[global_index].initval = default_global_addr

            global_map_info[global_index] = i
            global_index += 1

            storage_used += space_size

    # input (placeholder)
    for obj in sorted(objs, key=lambda x: x.object_id):
        if not bt.is_operator(obj):
            continue

        if (bt.is_output_chainable_operator(obj) and
                not obj.chain_head):
            continue

        srcs = obj.collect_sources()

        for src in srcs:
            if src.global_index is not None:
                continue

            if bt.is_view(src):
                continue

            if bt.is_removable_reshape(src):
                continue

            if not bt.is_input_storage(src):
                continue

            # source
            width = src.dtype.width
            length = src.get_aligned_length()
            space_size = align_space(width, length, chunk_size)
            default_global_addr = storage_used

            src.set_global_index(global_index)
            src.set_local_index(0)
            src.set_default_global_addr(default_global_addr)
            src.set_default_local_addr(0)

            i = (("%s %s "
                  "(size: %s, dtype: %s, shape: %s, "
                  "alignment: %d words (%d bytes)), "
                  "aligned shape: %s") %
                 (src.__class__.__name__,
                  "'%s'" % src.name if src.name is not None else 'None',
                  size_str(space_size),
                  src.dtype.to_str() if src.dtype is not None else 'None',
                  (str(src.shape)
                   if isinstance(src.shape, (tuple, list)) else '()'),
                  src.get_word_alignment(),
                  bt.to_byte(src.get_word_alignment()
                             * src.get_ram_width()),
                  (str(tuple(src.get_aligned_shape()))
                   if isinstance(src.shape, (tuple, list)) else '()')))

            global_mem_map[(default_global_addr,
                            default_global_addr + space_size - 1)] = i

            if config['use_map_ram']:
                global_addr_map[global_index] = default_global_addr

            if not config['use_map_ram']:
                map_regs[global_index].initval = default_global_addr

            global_map_info[global_index] = i
            global_index += 1

            storage_used += space_size

    # unified input (variable, constant)
    if not config['use_map_ram'] and not config['use_map_reg']:
        unified_global_index = global_index
        unified_default_global_addr = storage_used
        map_regs[unified_global_index].initval = unified_default_global_addr
        global_index += 1

    for obj in sorted(objs, key=lambda x: x.object_id):
        if not bt.is_operator(obj):
            continue

        if (bt.is_output_chainable_operator(obj) and
                not obj.chain_head):
            continue

        srcs = obj.collect_sources()

        for src in srcs:
            if src.global_index is not None:
                continue

            if bt.is_view(src):
                continue

            if bt.is_removable_reshape(src):
                continue

            if not bt.is_storage(src):
                continue

            # source
            width = src.dtype.width
            length = src.get_aligned_length()
            space_size = align_space(width, length, chunk_size)
            default_global_addr = storage_used

            if not config['use_map_ram'] and not config['use_map_reg']:
                src.set_global_index(unified_global_index)
                src.set_local_index(0)
                src.set_default_global_addr(unified_default_global_addr)
                src.set_default_local_addr(unified_storage_used)
            else:
                src.set_global_index(global_index)
                src.set_local_index(0)
                src.set_default_global_addr(default_global_addr)
                src.set_default_local_addr(0)

            i = (("%s %s "
                  "(size: %s, dtype: %s, shape: %s, "
                  "alignment: %d words (%d bytes)), "
                  "aligned shape: %s") %
                 (src.__class__.__name__,
                  "'%s'" % src.name if src.name is not None else 'None',
                  size_str(space_size),
                  src.dtype.to_str() if src.dtype is not None else 'None',
                  (str(src.shape)
                   if isinstance(src.shape, (tuple, list)) else '()'),
                  src.get_word_alignment(),
                  bt.to_byte(src.get_word_alignment()
                             * src.get_ram_width()),
                  (str(tuple(src.get_aligned_shape()))
                   if isinstance(src.shape, (tuple, list)) else '()')))

            global_mem_map[(default_global_addr,
                            default_global_addr + space_size - 1)] = i

            if config['use_map_ram']:
                global_addr_map[global_index] = default_global_addr

            if not config['use_map_ram'] and config['use_map_reg']:
                map_regs[global_index].initval = default_global_addr

            if config['use_map_ram']:
                global_map_info[global_index] = i

            global_index += 1

            storage_used += space_size
            unified_storage_used += space_size
            unified_storage_list.append(src)

    if (not config['use_map_ram'] and not config['use_map_reg'] and
            unified_storage_used > 0):

        names = ', '.join(["'%s'" % obj.name if obj.name is not None else 'None'
                           for obj in unified_storage_list])
        i = ('variables %s (size: %s)' %
             (names, size_str(unified_storage_used)))
        global_map_info[unified_global_index] = i

    # temporal
    for obj in sorted(objs, key=lambda x: x.object_id):
        if not bt.is_operator(obj):
            continue

        if (bt.is_output_chainable_operator(obj) and
                not obj.chain_head):
            continue

        srcs = obj.collect_sources()

        for src in srcs:
            if src.global_index is not None:
                continue

            if bt.is_view(src):
                continue

            if bt.is_removable_reshape(src):
                continue

            if bt.is_storage(src):
                continue

            # temporal
            width = src.dtype.width
            length = src.get_aligned_length()
            space_size = align_space(width, length, chunk_size)

            default_global_addr = storage_used
            default_local_addr = temporal_used
            local_addr_map[local_index] = default_local_addr

            src.set_global_index(0)
            src.set_local_index(local_index)
            src.set_default_global_addr(default_global_addr)
            src.set_default_local_addr(default_local_addr)

            local_index += 1
            temporal_used += space_size

    default_global_addr = storage_used
    global_addr_map[0] = default_global_addr

    i = ('temporal storages (size: %s)' %
         size_str(temporal_used))

    global_mem_map[(default_global_addr,
                    default_global_addr + temporal_used - 1)] = i

    if not config['use_map_ram']:
        map_regs[0].initval = default_global_addr

    global_map_info[0] = i

    return global_addr_map, local_addr_map, global_map_info, global_mem_map


def align_space(width, length, chunk_size):
    bytes = int(math.ceil((width * length) / 8))
    num_chunks = int(math.ceil(bytes / chunk_size))
    return chunk_size * num_chunks


def size_str(num_bytes):
    if num_bytes >= 1024:
        return '%dKB' % int(math.ceil(num_bytes / 1024))

    return '%dB' % num_bytes


def make_addr_map_rams(config, m, clk, rst, maxi,
                       global_addr_map, local_addr_map):

    datawidth = config['maxi_addrwidth']

    global_name = 'global_map_ram'

    if len(global_addr_map) > 1:
        global_addrwidth = int(math.ceil(math.log(len(global_addr_map), 2)))
    else:
        global_addrwidth = 1

    global_initvals = []
    for index, addr in sorted(global_addr_map.items(), key=lambda x: x[0]):
        global_initvals.append(addr)

    global_map_ram = vthread.RAM(m, global_name, clk, rst,
                                 datawidth, global_addrwidth, numports=2,
                                 initvals=global_initvals,
                                 ram_style=config['map_ram_style'])

    local_name = 'local_map_ram'

    if len(local_addr_map) > 1:
        local_addrwidth = int(math.ceil(math.log(len(local_addr_map), 2)))
    else:
        local_addrwidth = 1

    local_initvals = []
    for index, addr in sorted(local_addr_map.items(), key=lambda x: x[0]):
        local_initvals.append(addr)

    local_map_ram = vthread.RAM(m, local_name, clk, rst,
                                datawidth, local_addrwidth, numports=1,
                                initvals=local_initvals,
                                ram_style=config['map_ram_style'])

    return global_map_ram, local_map_ram


def make_controls(config, m, clk, rst, maxi, saxi,
                  schedule_table, control_param_dict,
                  global_addr_map, local_addr_map,
                  global_map_ram, local_map_ram):

    num_global_vars = len(global_addr_map)

    name = 'main_fsm'
    main_fsm = vg.FSM(m, name, clk, rst, as_module=config['fsm_as_module'])
    state_init = main_fsm.current
    main_fsm.state_list = []

    saxi.seq.If(main_fsm.state == state_init)(
        saxi.register[control_reg_busy](0),
        saxi.register[control_reg_reset](0),
        saxi.register[control_reg_extern_send](0)
    )

    if config['measurable_main_fsm']:
        internal_counter = m.Reg("internal_state_counter", width=32, initval=0)
        saxi.seq.If(main_fsm.state == state_init + 1)(
            internal_counter(0),
            saxi.register[control_reg_count](0)
        ).Elif(main_fsm.state == saxi.register[control_reg_count_state])(
            vg.If(internal_counter == saxi.register[control_reg_count_div])(
                internal_counter(0),
                saxi.register[control_reg_count](saxi.register[control_reg_count] + 1)
            ).Else(
                internal_counter(internal_counter + 1)
            )
        )

    if not config['use_map_ram']:
        map_regs = saxi.register[num_header_regs + num_control_regs:]

    offset_reg = saxi.register[control_reg_global_offset]
    maxi.seq(
        maxi.global_base_addr(offset_reg)
    )

    if config['use_map_ram']:
        start = saxi.register[control_reg_start]
        load_global_addr_map = saxi.register[control_reg_load_global_addr_map]
        global_addr = saxi.register[control_reg_addr_global_addr_map]

        def wait_start():
            while True:
                if load_global_addr_map:
                    saxi.write(control_reg_busy_global_addr_map, 1)
                    saxi.write(control_reg_load_global_addr_map, 0)

                    maxi.dma_read(global_map_ram, 0, global_addr,
                                  num_global_vars, port=1)

                    saxi.write(control_reg_busy_global_addr_map, 0)

                elif start:
                    saxi.write(control_reg_busy, 1)
                    saxi.write(control_reg_start, 0)
                    break

        main_fsm = vthread.embed_thread(main_fsm, wait_start)

    else:
        saxi.wait(main_fsm, control_reg_start, 0, polarity=False)
        saxi.write(main_fsm, control_reg_busy, 1)
        saxi.write(main_fsm, control_reg_start, 0)

    control_cache = collections.defaultdict(list)

    for stage, objs in sorted(schedule_table.items(), key=lambda x: x[0]):

        for obj in objs:
            if not bt.is_operator(obj):
                continue

            if bt.is_view(obj):
                continue

            if bt.is_removable_reshape(obj):
                continue

            if (bt.is_output_chainable_operator(obj) and
                    not obj.chain_head):
                continue

            param_key = obj.get_stream_hash()
            control_param_list = control_param_dict[param_key]
            control_param_len = len(control_param_list)

            key = obj.get_control_hash()

            if (not config['disable_control_cache'] and
                    obj.control_cachable and len(control_cache[key]) > 0):
                # hit
                control, orig = control_cache[key][0]
                if control.stream_ram_hash != key:
                    raise ValueError("hash mismatch: '%x' != '%x'" %
                                     (control.stream_ram_hash, key))

                obj.copy_control(orig)

            else:
                # miss
                obj.make_objaddr()
                obj.make_arg_objaddrs()

                control = obj.make_control(fsm_as_module=config['fsm_as_module'])
                control.stream_ram_hash = key
                control_cache[key].append((control, obj))

            # bind address
            if config['use_map_ram']:
                gaddr = global_map_ram.read(main_fsm, obj.global_index)
                main_fsm.set_index(main_fsm.current - 1)
                laddr = local_map_ram.read(main_fsm, obj.local_index)
            else:
                gaddr = map_regs[obj.global_index]
                laddr = obj.default_local_addr

            if laddr != 0:
                addr = gaddr + laddr
            else:
                addr = gaddr

            main_fsm(
                obj.objaddr(addr)
            )
            main_fsm.goto_next()

            arg_global_indexes = obj.get_arg_global_indexes()
            arg_local_indexes = obj.get_arg_local_indexes()
            arg_default_global_addrs = obj.get_arg_default_global_addrs()
            arg_default_local_addrs = obj.get_arg_default_local_addrs()

            for (arg_objaddr, arg_global_index, arg_local_index,
                 arg_default_global_addr, arg_default_local_addr) in zip(
                     obj.arg_objaddrs, arg_global_indexes, arg_local_indexes,
                     arg_default_global_addrs, arg_default_local_addrs):

                if config['use_map_ram']:
                    gaddr = global_map_ram.read(main_fsm, arg_global_index)
                    main_fsm.set_index(main_fsm.current - 1)
                    laddr = local_map_ram.read(main_fsm, arg_local_index)
                else:
                    gaddr = map_regs[arg_global_index]
                    laddr = arg_default_local_addr

                if laddr != 0:
                    addr = gaddr + laddr
                else:
                    addr = gaddr

                main_fsm(
                    arg_objaddr(addr)
                )
                main_fsm.goto_next()

            # bind parameter parameters
            obj.set_control_params(main_fsm, control_param_len,
                                   use_param_ram=config['use_param_ram'],
                                   min_param_ram_len=config['min_param_ram_len'])

            obj.run_control(main_fsm)

        ret_start_state = main_fsm.current
        main_fsm.goto_next()

        for obj in objs:
            if not bt.is_operator(obj):
                continue

            if (bt.is_output_chainable_operator(obj) and
                    not obj.chain_head):
                continue

            obj.join_control(main_fsm)
            obj.reset_control(main_fsm)

        ret_end_state = main_fsm.current

        if not bt.is_operator(obj):
            control_name = 'None'
        elif bt.is_view(obj):
            control_name = 'None'
        elif bt.is_removable_reshape(obj):
            control_name = 'None'
        elif (bt.is_output_chainable_operator(obj) and
                not obj.chain_head):
            control_name = 'None'
        else:
            control_name = control_cache[obj.get_control_hash()][0][0].name

        main_fsm.state_list.append(
            (ret_start_state, ret_end_state, obj.name, control_name))

        main_fsm.goto_next()

    # finalize
    main_fsm.goto_next()

    saxi.write(main_fsm, control_reg_busy, 0)

    main_fsm.goto_next()
    main_fsm.goto_next()
    main_fsm.goto_init()

    return control_cache, main_fsm


def make_reg_map(config, global_map_info, header_info):
    reg_map = collections.OrderedDict()
    reg_type = {'rw': 'RW', 'r': 'R ', 'w': ' W', 'x': '  '}

    for i in range(num_header_regs):
        index = index_to_bytes(i)
        reg_map[index] = (reg_type['r'], header_info[i])

    index = index_to_bytes(control_reg_start)
    reg_map[index] = (reg_type['w'], "Start (set '1' to run)")

    index = index_to_bytes(control_reg_busy)
    reg_map[index] = (reg_type['r'], "Busy (returns '1' when running)")

    index = index_to_bytes(control_reg_reset)
    reg_map[index] = (reg_type['w'], "Reset (set '1' to initialize internal logic)")

    index = index_to_bytes(control_reg_extern_send)
    reg_map[index] = (reg_type['r'], "Opcode from extern objects to SW (returns '0' when idle)")

    index = index_to_bytes(control_reg_extern_recv)
    reg_map[index] = (reg_type['w'], "Resume extern objects (set '1' to resume)")

    if config['interrupt_enable']:
        index = index_to_bytes(control_reg_interrupt_isr)
        reg_map[index] = (reg_type['r'], "Interrupt Status Register")

        index = index_to_bytes(control_reg_interrupt_ier)
        reg_map[index] = (reg_type['w'], "Interrupt Enable Register")

        index = index_to_bytes(control_reg_interrupt_iar)
        reg_map[index] = (reg_type['w'], "Interrupt Acknowledge Register")

    if config['use_map_ram']:
        control_reg_reserved_start = control_reg_addr_global_addr_map + 1
    elif config['interrupt_enable']:
        control_reg_reserved_start = control_reg_reserved
    else:
        control_reg_reserved_start = control_reg_interrupt_isr

    if config['measurable_main_fsm']:
        index = index_to_bytes(control_reg_count)
        reg_map[index] = (reg_type['r'], "State Counter")

        index = index_to_bytes(control_reg_count_state)
        reg_map[index] = (reg_type['w'], "Count Target")

        index = index_to_bytes(control_reg_count_div)
        reg_map[index] = (reg_type['w'], "Count Divider")
    else:
        control_reg_reserved_start = control_reg_count
    index = index_to_bytes(control_reg_reserved_start)
    reg_map[index] = (reg_type['x'], "Reserved ...")

    index = index_to_bytes(control_reg_address_amount - 1)
    reg_map[index] = (reg_type['x'], "... Reserved")

    index = index_to_bytes(control_reg_address_amount)
    reg_map[index] = (reg_type['r'], "Address space amount")

    index = index_to_bytes(control_reg_global_offset)
    default_global_addr_offset = config['default_global_addr_offset']
    reg_map[index] = (reg_type['rw'], 'Global address offset (default: %d)' %
                      default_global_addr_offset)

    if config['use_map_ram']:
        index = index_to_bytes(control_reg_load_global_addr_map)
        reg_map[index] = (reg_type['w'], "Load global address map (set '1' to load)")

        index = index_to_bytes(control_reg_busy_global_addr_map)
        reg_map[index] = (
            reg_type['r'], "Busy loading global address map (returns '1' when loading)")

        index = index_to_bytes(control_reg_addr_global_addr_map)
        reg_map[index] = (reg_type['w'], "Head address of global address map")

    else:
        for gindex, info in sorted(global_map_info.items(), key=lambda x: x[0]):
            index = index_to_bytes(control_reg_global_addr + gindex)
            reg_map[index] = (reg_type['rw'], 'Address of ' + info)

    return reg_map


def index_to_bytes(index, wordsize=4):
    return index * wordsize


def dump_main_fsm(main_fsm):
    s = []
    s.append('[State IDs in main_fsm]')

    for value in main_fsm.state_list:
        s.append("  %s" % str(value))

    print('\n'.join(s))


def dump_config(config, where_from=None, output=None):
    s = []
    s.append('NNgen: Neural Network Accelerator Generator (version %s)' %
             version.__version__)

    if where_from == 'to_verilog':
        s.append('[Verilog HDL]')
        s.append('  Output: %s' % output)
    elif where_from == 'to_ipxact':
        s.append('[IP-XACT]')
        s.append('  Output: %s' % output)

    s.append('[Configuration]')
    s.append('(AXI Master Interface)')
    s.append('  Data width   : %d' % config['maxi_datawidth'])
    s.append('  Address width: %d' % config['maxi_addrwidth'])
    s.append('(AXI Slave Interface)')
    s.append('  Data width   : %d' % config['saxi_datawidth'])
    s.append('  Address width: %d' % config['saxi_addrwidth'])

    no_print = ('maxi_datawidth', 'maxi_addrwidth',
                'saxi_datawidth', 'saxi_addrwidth')

    has_other_option = False
    for key, value in sorted(config.items(), key=lambda x: x[0]):
        if key in no_print:
            continue
        if key in default_config and default_config[key] != value:
            if not has_other_option:
                s.append('(Other Options)')
                has_other_option = True
            s.append('  %s: %s' % (key, str(value)))

    print('\n'.join(s))


def dump_schedule_table(schedule_table):
    s = []
    s.append('[Schedule Table]')

    for stage, objs in sorted(schedule_table.items(), key=lambda x: x[0]):
        s.append('(Stage %d)' % stage)
        for obj in objs:
            if bt.is_storage(obj):
                continue

            if bt.is_output_chainable_operator(obj) and not obj.chain_head:
                continue

            s.append('  %s' % str(obj))
            s.extend(dump_sources(obj, prefix='  | '))

    print('\n'.join(s))


def dump_sources(obj, prefix='  '):
    s = []
    srcs = obj.args
    for i, src in enumerate(srcs):
        s.append('%s%s' % (prefix, str(src)))
        if bt.is_output_chainable_operator(src) and not src.chain_head:
            next_prefix = prefix + '| '
            s.extend(dump_sources(src, prefix=next_prefix))
    return s


def dump_rams(ram_dict):
    s = []
    s.append('[RAM (spec: num)]')

    for (width, length), rams in sorted(ram_dict.items(), key=lambda x: x[0], reverse=True):
        num_rams = len(rams)
        if isinstance(rams[0], vthread.MultibankRAM):
            bank = rams[0].numbanks
            ports = rams[0].numports
        else:
            bank = 1
            ports = rams[0].numports

        s.append('  %d-bit %d-entry %d-port %d-bank RAM: %d' %
                 (width, length, ports, bank, num_rams))

    print('\n'.join(s))


def dump_substreams(substrm_dict):
    s = []
    s.append('[Substream (spec: num)]')

    for hsh, lst in substrm_dict.items():
        s.append('  %s: %d' % (str(hsh), len(lst)))

    print('\n'.join(s))


def dump_streams(stream_cache):
    s = []
    s.append('[Stream (spec: num)]')

    for hsh, lst in stream_cache.items():
        s.append('  %s: %d' % (str(hsh), len(lst)))

    print('\n'.join(s))


def dump_controls(control_cache, main_fsm):
    s = []
    s.append('[Control (name (# states: num))]')

    indexes = set(main_fsm.body.keys())
    indexes.update(set(main_fsm.jump.keys()))
    num_states = len(indexes)
    s.append('  %s (# states: %d)' % (main_fsm.name, num_states))

    for lst in control_cache.values():
        for control, obj in lst:
            indexes = set(control.fsm.body.keys())
            indexes.update(set(control.fsm.jump.keys()))
            num_states = len(indexes)
            s.append('  %s (# states: %d)' % (control.name, num_states))

    print('\n'.join(s))


def dump_register_map(reg_map):
    s = []
    s.append('[Register Map]')

    maximum = sorted(reg_map.items(), key=lambda x: x[0], reverse=True)[0][0]
    num_digits = max(int(math.ceil(math.log(maximum, 10))), 1)
    fmt = ''.join(('  %', '%d' % num_digits, 'd (%s): %s'))

    for i, (direction, desc) in sorted(reg_map.items(), key=lambda x: x[0]):
        s.append(fmt % (i, direction, desc))

    print('\n'.join(s))


def dump_memory_map(mem_map):
    max_gaddr = 0
    min_gaddr = 0
    for start, end in mem_map.keys():
        if start <= end:
            max_gaddr = max(max_gaddr, start, end)
            min_gaddr = min(min_gaddr, start, end)

    s = []
    num_bytes = max_gaddr - min_gaddr + 1
    s.append('[Default Memory Map (start - end)] (entire range: [%d - %d], size: %s)' %
             (min_gaddr, max_gaddr, size_str(num_bytes)))

    num_digits = max(int(math.ceil(math.log(max_gaddr, 10))), 1)
    fmt = ''.join(('  [%', '%d' % num_digits, 'd - %', '%d' % num_digits, 'd]: %s'))

    for (start, end), desc in sorted(mem_map.items(), key=lambda x: x[0]):
        if start <= end:
            s.append(fmt % (start, end, desc))

    print('\n'.join(s))
