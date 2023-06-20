from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from . import verilog


__intrinsics__ = ('set_header', 'get_header',
                  'set_global_offset', 'set_global_addrs',
                  'set_global_addr_map', 'write_global_addr_map', 'load_global_addr_map',
                  'start', 'wait', 'sw_rst')


def set_header(fsm, saxi, index, header, wordsize=4):
    awaddr = (verilog.header_reg + index) * wordsize
    saxi.write(fsm, awaddr, header)


def get_header(fsm, saxi, index, wordsize=4):
    araddr = (verilog.header_reg + index) * wordsize
    h = saxi.read(fsm, araddr)
    return h


def set_global_offset(fsm, saxi, addr, **opt):
    wordsize = opt['wordsize'] if 'wordsize' in opt else 4

    awaddr = verilog.control_reg_global_offset * wordsize
    saxi.write(fsm, awaddr, addr)


def set_global_addrs(fsm, saxi, *addrs, **opt):
    wordsize = opt['wordsize'] if 'wordsize' in opt else 4
    offset = opt['offset'] if 'offset' in opt else 0

    awaddr = (offset + verilog.control_reg_global_addr) * wordsize
    for addr in addrs:
        saxi.write(fsm, awaddr, addr)
        awaddr += wordsize


def set_global_addr_map(fsm, saxi, memory, map_addr, *addrs, **opt):
    write_global_addr_map(fsm, memory, map_addr, *addrs, **opt)
    load_global_addr_map(fsm, saxi, map_addr, **opt)


def write_global_addr_map(fsm, memory, map_addr, *addrs, **opt):
    wordsize = opt['wordsize'] if 'wordsize' in opt else 4
    offset = opt['offset'] if 'offset' in opt else 0

    for i, addr in enumerate(addrs):
        memory.write_word(fsm, i + offset, map_addr, addr, wordsize * 8)


def load_global_addr_map(fsm, saxi, map_addr, **opt):
    wordsize = opt['wordsize'] if 'wordsize' in opt else 4

    awaddr = verilog.control_reg_addr_global_addr_map * wordsize
    saxi.write(fsm, awaddr, map_addr)

    awaddr = verilog.control_reg_load_global_addr_map * wordsize
    saxi.write(fsm, awaddr, 1)

    araddr = verilog.control_reg_load_global_addr_map * wordsize
    b = fsm.current
    v = saxi.read(fsm, araddr)
    fsm.If(v != 0).goto(b)
    fsm.If(v == 0).goto_next()

    araddr = verilog.control_reg_busy_global_addr_map * wordsize
    b = fsm.current
    v = saxi.read(fsm, araddr)
    fsm.If(v != 0).goto(b)
    fsm.If(v == 0).goto_next()


def start(fsm, saxi, wordsize=4):
    awaddr = verilog.control_reg_start * wordsize
    saxi.write(fsm, awaddr, 1)

    araddr = verilog.control_reg_start * wordsize
    b = fsm.current
    v = saxi.read(fsm, araddr)
    fsm.If(v != 0).goto(b)
    fsm.If(v == 0).goto_next()


def wait(fsm, saxi, wordsize=4):
    araddr = verilog.control_reg_busy * wordsize
    b = fsm.current
    v = saxi.read(fsm, araddr)
    fsm.If(v != 0).goto(b)
    fsm.If(v == 0).goto_next()


def sw_rst(fsm, saxi, wordsize=4):
    awaddr = verilog.control_reg_reset * wordsize
    saxi.write(fsm, awaddr, 1)

    araddr = verilog.control_reg_busy * wordsize
    b = fsm.current
    v = saxi.read(fsm, araddr)
    fsm.If(v != 0).goto(b)
    fsm.If(v == 0).goto_next()
