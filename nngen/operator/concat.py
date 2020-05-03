from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import functools
from collections import OrderedDict

import veriloggen as vg

import nngen.basic_types as bt
import nngen.util as util


class concat(bt._Operator):
    input_chainable = False
    output_chainable = False

    def __sub_str__(self):
        axis = ' axis:%d' % self.axis
        buffered = (' buffered'
                    if hasattr(self, 'buffered_value') and self.buffered_value else '')
        return ''.join([axis, buffered])

    def __init__(self, values, axis, dtype=None, name=None):
        rank = bt.get_rank(values[0].shape)
        _dtype = values[0].dtype

        for value in values:
            r = bt.get_rank(value.shape)
            if r != rank:
                raise ValueError('all values must have a same rank: %d != %d' %
                                 (r, rank))
            rank = r
            d = value.dtype
            if d != _dtype:
                raise ValueError('all values must have a same dtype: %s != %s' %
                                 (d, _dtype))
            _dtype = d

        if isinstance(axis, (tuple, list)):
            raise TypeError('axis must be int, not tuple or list.')

        axis = util.to_axis(axis, rank)[0]

        shape = []
        for i in range(rank):
            size = 0
            for value in values:
                if i == axis:
                    size += value.shape[i]
                else:
                    if size != 0 and size != value.shape[i]:
                        raise ValueError(
                            'all values must have a same shape, excluding axis: %d != %d' %
                            (size, value.shape[i]))
                    size = max(size, value.shape[i])

            shape.append(size)

        shape = tuple(shape)

        bt._Operator.__init__(self, *values,
                              dtype=dtype, shape=shape, name=name)
        self.axis = axis

    def attribute(self):
        pass

    def get_required_rams(self):
        arg_width = 0
        arg_len = 0
        for arg in self.args:
            arg_width = max(arg_width, arg.get_ram_width())
            arg_len = max(arg_len, arg.shape[-1])

        inputs = [(arg_width, arg_len)]
        outputs = [(self.get_ram_width(), self.shape[-1])]
        temps = []
        return inputs, outputs, temps

    def get_stream_func(self):
        def func(strm):
            datawidth = self.args[0].get_op_width()
            src = strm.source(datawidth=datawidth)
            strm.sink(src)

        return func

    def get_control_param_values(self):
        buffered = False

        for arg in self.args:
            if self.dtype != arg.dtype:
                buffered = True

        if self.axis == bt.get_rank(self.shape) - 1:
            for arg in self.args:
                if arg.shape[-1] != arg.get_aligned_shape()[-1]:
                    buffered = True

        # for __str__
        self.buffered_value = buffered

        aligned_shape = self.get_aligned_shape()
        aligned_length = self.get_aligned_length()

        arg_read_sizes = [arg.shape[-1] for arg in self.args]
        arg_addr_incs = [bt.to_byte(bt.align_word(arg.shape[-1], arg.get_word_alignment()) *
                                    arg.get_ram_width())
                         for arg in self.args]

        arg_chunk_sizes = [functools.reduce(lambda x, y: x * y, arg.shape[self.axis:-1], 1)
                           for arg in self.args]

        out_write_size = aligned_shape[-1]
        out_addr_inc = bt.to_byte(bt.align_word(self.shape[-1], self.get_word_alignment()) *
                                  self.get_ram_width())

        num_steps = int(math.ceil(aligned_length / out_write_size))
        if not buffered:
            num_steps *= len(self.args)

        return OrderedDict([('buffered', buffered),
                            ('arg_read_sizes', arg_read_sizes),
                            ('arg_addr_incs', arg_addr_incs),
                            ('arg_chunk_sizes', arg_chunk_sizes),
                            ('out_write_size', out_write_size),
                            ('out_addr_inc', out_addr_inc),
                            ('num_steps', num_steps)])

    def control_sequence(self, fsm):
        arg_gaddrs = [self.m.Reg(self._name('arg_gaddr_%d' % i),
                                 self.maxi.addrwidth, initval=0)
                      for i, _ in enumerate(self.arg_objaddrs)]
        out_gaddr = self.m.Reg(self._name('out_gaddr'),
                               self.maxi.addrwidth, initval=0)

        arg_laddr = self.m.Reg(self._name('arg_laddr'),
                               self.maxi.addrwidth, initval=0)
        copy_laddr = self.m.Reg(self._name('copy_laddr'),
                                self.maxi.addrwidth, initval=0)
        copy_size = self.m.Reg(self._name('copy_size'),
                               self.maxi.addrwidth, initval=0)
        sum_read_sizes = self.m.Wire(self._name('sum_read_sizes'),
                                     max([i.width for i in self.arg_read_sizes]) +
                                     int(math.ceil(math.log2(len(self.arg_read_sizes)))))
        sum_read_sizes.assign(vg.Add(*self.arg_read_sizes))

        out_addr_inc_unbuffered = self.m.Reg(self._name('out_addr_inc_unbuffered'),
                                             self.maxi.addrwidth, initval=0)

        arg_select = self.m.Reg(self._name('arg_select'),
                                int(max(math.ceil(math.log(len(self.args), 2)), 1)),
                                initval=0)
        arg_chunk_count = self.m.Reg(self._name('arg_chunk_count'),
                                     self.maxi.addrwidth + 1, initval=0)
        out_count = self.m.Reg(self._name('out_count'),
                               self.maxi.addrwidth + 1, initval=0)

        # --------------------
        # initialization phase
        # --------------------
        fsm(
            [arg_gaddr(0) for arg_gaddr in arg_gaddrs],
            out_gaddr(0),
            arg_laddr(0),
            copy_laddr(0),
            copy_size(0),
            out_addr_inc_unbuffered(0),
            arg_select(0),
            arg_chunk_count(0),
            out_count(0)
        )

        fsm.goto_next()

        # --------------------
        # Read phase
        # --------------------
        state_read = fsm.current

        fsm.inc()

        state_read_begin_list = []
        state_read_end_list = []

        for (arg,
             arg_objaddr, arg_gaddr, arg_addr_inc,
             arg_read_size, arg_chunk_size) in zip(
                 self.args,
                 self.arg_objaddrs, arg_gaddrs, self.arg_addr_incs,
                 self.arg_read_sizes, self.arg_chunk_sizes):

            b = fsm.current
            state_read_begin_list.append(b)

            # normal
            laddr = arg_laddr
            gaddr = arg_objaddr + arg_gaddr

            bt.bus_lock(self.maxi, fsm)
            bt.dma_read(self.maxi, fsm, self.input_rams[0], laddr, gaddr, arg_read_size)
            bt.bus_unlock(self.maxi, fsm)

            fsm(
                arg_gaddr.add(arg_addr_inc),
                arg_laddr.add(arg_read_size),
                arg_chunk_count.inc(),
                copy_size(arg_read_size),
                out_addr_inc_unbuffered(arg_addr_inc)
            )
            fsm.If(arg_chunk_count == arg_chunk_size - 1)(
                arg_chunk_count(0),
                arg_select.inc()
            )
            fsm.If(arg_chunk_count == arg_chunk_size - 1,
                   arg_select == len(self.args) - 1)(
                arg_select(0)
            )

            e = fsm.current
            state_read_end_list.append(e)

            fsm.inc()

        state_read_end = fsm.current

        for i, b in enumerate(state_read_begin_list):
            fsm.If(arg_select == i).goto_from(state_read, b)

        for i, e in enumerate(state_read_end_list):
            fsm.goto_from(e, state_read_end)

        # --------------------
        # Copy phase
        # --------------------
        state_copy = fsm.current

        name = list(self.stream.sources.keys())[0]
        self.stream.set_source(fsm, name, self.input_rams[0], 0, copy_size)
        fsm.set_index(fsm.current - 1)

        name = list(self.stream.sinks.keys())[0]
        self.stream.set_sink(fsm, name, self.output_rams[0], copy_laddr, copy_size)
        self.stream.run(fsm)
        self.stream.join(fsm)

        fsm(
            arg_laddr(0),
            copy_laddr.add(copy_size)
        )
        fsm.goto_next()

        fsm.If(copy_laddr < sum_read_sizes).goto(state_read)
        fsm.If(copy_laddr >= sum_read_sizes).goto_next()

        state_copy_end = fsm.current
        fsm.If(vg.Not(self.buffered)).goto_from(state_copy, state_copy_end)

        # --------------------
        # Write phase
        # --------------------
        state_write = fsm.current
        fsm.inc()

        # Case with Copy
        state_write_buffered = fsm.current

        laddr = 0
        gaddr = self.objaddr + out_gaddr
        bt.bus_lock(self.maxi, fsm)
        bt.dma_write(self.maxi, fsm,
                     self.output_rams[0], laddr, gaddr, self.out_write_size)
        bt.bus_unlock(self.maxi, fsm)

        fsm(
            copy_laddr(0),
            out_gaddr.add(self.out_addr_inc),
            out_count.inc()
        )

        state_write_end_buffered = fsm.current
        fsm.inc()

        # Case without Copy
        state_write_unbuffered = fsm.current

        laddr = 0
        gaddr = self.objaddr + out_gaddr
        bt.bus_lock(self.maxi, fsm)
        bt.dma_write(self.maxi, fsm,
                     self.input_rams[0], laddr, gaddr, copy_size)
        bt.bus_unlock(self.maxi, fsm)

        fsm(
            arg_laddr(0),
            out_gaddr.add(out_addr_inc_unbuffered),
            out_count.inc()
        )

        state_write_end_unbuffered = fsm.current
        fsm.inc()

        state_write_end = fsm.current

        fsm.If(self.buffered).goto_from(state_write, state_write_buffered)
        fsm.If(vg.Not(self.buffered)).goto_from(state_write, state_write_unbuffered)

        fsm.goto_from(state_write_end_buffered, state_write_end)
        fsm.goto_from(state_write_end_unbuffered, state_write_end)

        # --------------------
        # update for next iteration
        # --------------------
        fsm.If(out_count < self.num_steps).goto(state_read)
        fsm.If(out_count == self.num_steps).goto_next()

    def eval(self, memo, input_dict, **kwargs):
        if id(self) in memo:
            return memo[id(self)]

        import nngen.verify as verify

        name = self.__class__.__name__

        values = [arg.eval(memo, input_dict)
                  for arg in self.args]

        kwargs['axis'] = self.axis
        kwargs['dtype'] = self.dtype
        kwargs['name'] = self.name

        method = self.get_eval_method()
        ret = method(values, **kwargs)
        memo[id(self)] = ret

        return ret
