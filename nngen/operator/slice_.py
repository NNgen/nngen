from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import functools
import math
import numpy as np
from collections import OrderedDict

import veriloggen as vg

import nngen.basic_types as bt
import nngen.util as util


class slice_(bt._Operator):
    """
    Create a sliced tensor with a similar API to the numpy slice.
    """
    input_chainable = False
    output_chainable = False

    def __sub_str__(self):
        begins = str(self.begins)
        ends = str(self.ends)
        strides = str(self.strides)
        par = ' par:%d' % self.par if self.par > 1 else ''

        value_ram_size = (' value_ram_size:%d' % self.value_ram_size
                          if self.value_ram_size is not None else '')
        out_ram_size = (' out_ram_size:%d' % self.out_ram_size
                        if self.out_ram_size is not None else '')

        return (' begins:%s ends:%s strides:%s %s%s%s' %
                (begins, ends, strides, par, value_ram_size, out_ram_size))

    def __init__(self, value, begins, ends, strides,
                 dtype=None, name=None, par=1,
                 value_ram_size=None, out_ram_size=None):

        if not isinstance(begins, (tuple, list)):
            raise TypeError('begins must be tuple or list.')

        if not isinstance(ends, (tuple, list)):
            raise TypeError('ends must be tuple or list.')

        if not isinstance(strides, (tuple, list)):
            raise TypeError('strides must be tuple or list.')

        if len(value.shape) != len(begins):
            raise ValueError('length mismatch between value.shape and begins: %d != %d' %
                             (len(value.shape), len(begins)))

        if len(value.shape) != len(ends):
            raise ValueError('length mismatch between value.shape and ends: %d != %d' %
                             (len(value.shape), len(ends)))

        if len(value.shape) != len(strides):
            raise ValueError('length mismatch between value.shape and strides: %d != %d' %
                             (len(value.shape), len(strides)))

        for begin in begins:
            begin = int(begin)
            if not isinstance(begin, int):
                raise TypeError('values of begins must be int, not %s' % str(type(begin)))

        for end in ends:
            end = int(end)
            if not isinstance(end, int):
                raise TypeError('values of ends must be int, not %s' % str(type(end)))

        for stride in strides:
            stride = int(stride)
            if not isinstance(stride, int):
                raise TypeError('values of strides must be int, not %s' % str(type(stride)))

        if strides[-1] != 1 and par != 1:
            raise ValueError("par must be 1 when strides[-1] is not 1")

        if value_ram_size is not None and value_ram_size < 1:
            raise ValueError('value_ram_size must be greater than 0')

        if out_ram_size is not None and out_ram_size < 1:
            raise ValueError('out_ram_size must be greater than 0')

        # delegate a shape calculation to numpy
        slices = to_slices(begins, ends, strides)
        shape = np.zeros(value.shape)[slices].shape

        bt._Operator.__init__(self, value,
                              dtype=dtype, shape=shape, name=name, par=par)

        self.begins = tuple(begins)
        self.ends = tuple(ends)
        self.strides = tuple(strides)

        # attribute
        self.value_ram_size = value_ram_size
        self.out_ram_size = out_ram_size
        slice_.attribute(self, par, value_ram_size, out_ram_size)

    def attribute(self, par=None, value_ram_size=None, out_ram_size=None):
        if par is not None:
            if (par - 1) & par != 0:
                raise ValueError('par must be power of 2.')

            self.par = par

            for arg in self.args:
                arg.add_alignment_request(self.par)

            self.add_alignment_request(self.par)

        if value_ram_size is not None:
            if value_ram_size < 1:
                raise ValueError('value_ram_size must be greater than 0')

            self.value_ram_size = value_ram_size

        if out_ram_size is not None:
            if out_ram_size < 1:
                raise ValueError('out_ram_size must be greater than 0')

            self.out_ram_size = out_ram_size

    def get_required_rams(self):
        act = self.args[0]
        act_shape = act.get_aligned_shape()
        out_shape = self.get_aligned_shape()

        input_min_size = ((act_shape[-1] // self.par) *
                          (act_shape[-2] if len(act_shape) > 1 else 1) * 2)
        if self.value_ram_size is not None and input_min_size < self.value_ram_size:
            input_min_size = self.value_ram_size
        input_width = act.get_ram_width() * self.par

        output_min_size = ((out_shape[-1] // self.par) *
                           (out_shape[-2] if len(out_shape) > 1 else 1) * 2)
        if self.out_ram_size is not None and output_min_size < self.out_ram_size:
            output_min_size = self.out_ram_size
        output_width = self.get_ram_width() * self.par

        inputs = []
        inputs.append((input_width, input_min_size))

        outputs = []
        outputs.append((output_width, output_min_size))

        temps = []

        return inputs, outputs, temps

    def get_stream_hash(self):
        base = bt._Operator.get_stream_hash(self)
        rank = len(self.shape)
        return (base, rank, self.par)

    def get_stream_func(self):
        def func(strm):
            arg = self.args[0]
            datawidth = arg.get_op_width()
            vec_datawidth = datawidth * self.par
            point = arg.get_op_point()
            signed = arg.get_signed()
            vec_act_var = strm.source(datawidth=vec_datawidth, signed=False)
            strm.sink(vec_act_var)

        return func

    def get_control_param_values(self):
        act = self.args[0]

        act_shape = act.get_aligned_shape()
        act_num_ch = act_shape[-1]

        out_shape = self.get_aligned_shape()
        out_num_ch = out_shape[-1]

        act_offset_base = bt.to_byte(act_num_ch * act.get_ram_width())

        act_offset_begins = []
        act_offset_strides = []
        for i, (begin, stride) in enumerate(zip(reversed(self.begins[:-2]), reversed(self.strides[:-2]))):
            mul = functools.reduce(lambda x, y: x * y, act_shape[-i - 2:-1], 1)
            act_offset_begin = act_offset_base * mul * begin
            act_offset_begins.append(act_offset_begin)
            act_offset_stride = act_offset_base * mul * stride
            act_offset_strides.append(act_offset_stride)

        act_offset_begins.reverse()
        act_offset_strides.reverse()

        act_read_size = ((act_num_ch // self.par) *
                         (act_shape[-2] if len(act_shape) > 1 else 1))

        out_offset_base = bt.to_byte(out_num_ch * self.get_ram_width())

        out_offset_strides = []
        for i in range(len(out_shape) - 2):
            mul = functools.reduce(lambda x, y: x * y, out_shape[-i - 2:-1], 1)
            out_offset_stride = out_offset_base * mul
            out_offset_strides.append(out_offset_stride)

        out_offset_strides.reverse()

        out_write_size = ((out_num_ch // self.par) *
                          (out_shape[-2] if len(out_shape) > 1 else 1))

        stream_size = out_num_ch // self.par
        if len(self.strides) > 1:
            stream_stride = self.strides[-2] * (act_num_ch // self.par)
            stream_local = self.begins[-2] * (act_num_ch // self.par) + self.begins[-1]
        else:
            stream_stride = 0
            stream_local = 0

        return OrderedDict([('act_shape', act_shape),
                            ('out_shape', out_shape),
                            ('act_begins', self.begins),
                            ('act_strides', self.strides),
                            ('act_offset_begins', act_offset_begins),
                            ('act_offset_strides', act_offset_strides),
                            ('act_read_size', act_read_size),
                            ('out_offset_strides', out_offset_strides),
                            ('out_write_size', out_write_size),
                            ('stream_size', stream_size),
                            ('stream_stride', stream_stride),
                            ('stream_local', stream_local)])

    def control_sequence(self, fsm):
        act_ram = self.input_rams[0]
        out_ram = self.output_rams[0]

        act_base_offset = self.m.Wire(self._name('act_base_offset'),
                                      self.maxi.addrwidth, signed=True)

        act_offsets = [self.m.Reg(self._name('act_offset_%d' % i),
                                  self.maxi.addrwidth, initval=0, signed=True)
                       for i, _ in enumerate(self.act_shape[:-2])]

        if act_offsets:
            v = act_offsets[0]
            for act_offset in act_offsets[1:]:
                v += act_offset
            act_base_offset.assign(v)
        else:
            act_base_offset.assign(0)

        out_base_offset = self.m.Wire(self._name('out_base_offset'),
                                      self.maxi.addrwidth, signed=True)

        out_offsets = [self.m.Reg(self._name('out_offset_%d' % i),
                                  self.maxi.addrwidth, initval=0, signed=True)
                       for i, _ in enumerate(self.out_shape[:-2])]

        if out_offsets:
            v = out_offsets[0]
            for out_offset in out_offsets[1:]:
                v += out_offset
            out_base_offset.assign(v)
        else:
            out_base_offset.assign(0)

        counts = [self.m.Reg(self._name('count_%d' % i),
                             self.maxi.addrwidth, initval=0)
                  for i, _ in enumerate(self.act_shape[:-2])]

        prev_counts = [self.m.Reg(self._name('prev_count_%d' % i),
                                  self.maxi.addrwidth, initval=0)
                       for i, _ in enumerate(self.act_shape[:-2])]

        stream_act_local = self.m.Reg(self._name('stream_act_local'),
                                      self.maxi.addrwidth, initval=0)
        stream_out_local = self.m.Reg(self._name('stream_out_local'),
                                      self.maxi.addrwidth, initval=0)

        comp_count = self.m.Reg(self._name('comp_count'),
                                self.maxi.addrwidth, initval=0)
        out_count = self.m.Reg(self._name('out_count'),
                               self.maxi.addrwidth, initval=0)

        act_page = self.m.Reg(self._name('act_page'), initval=0)
        act_page_comp_offset = self.m.Reg(self._name('act_page_comp_offset'),
                                          self.maxi.addrwidth, initval=0)
        act_page_dma_offset = self.m.Reg(self._name('act_page_dma_offset'),
                                         self.maxi.addrwidth, initval=0)

        out_page = self.m.Reg(self._name('out_page'), initval=0)
        out_page_comp_offset = self.m.Reg(self._name('out_page_comp_offset'),
                                          self.maxi.addrwidth, initval=0)
        out_page_dma_offset = self.m.Reg(self._name('out_page_dma_offset'),
                                         self.maxi.addrwidth, initval=0)

        act_page_size = act_ram.length // 2
        out_page_size = out_ram.length // 2

        skip_read_act = self.m.Reg(self._name('skip_read_act'), initval=0)
        skip_comp = self.m.Reg(self._name('skip_comp'), initval=0)
        skip_write_out = self.m.Reg(self._name('skip_write_out'), initval=0)

        # --------------------
        # initialization phase
        # --------------------
        # ReadAct: offset
        for act_offset, act_offset_begin in zip(act_offsets, self.act_offset_begins):
            fsm(
                act_offset(act_offset_begin)
            )

        # ReadAct: double buffer control
        fsm(
            act_page(0),
            act_page_comp_offset(0),
            act_page_dma_offset(0)
        )

        # WriteOutput: offset
        for out_offset in out_offsets:
            fsm(
                out_offset(0)
            )

        # WriteOutput: double buffer control
        fsm(
            out_page(0),
            out_page_comp_offset(0),
            out_page_dma_offset(0)
        )

        # counter
        fsm(
            [count(0) for count in counts],
            [prev_count(0) for prev_count in prev_counts]
        )

        # double buffer control
        fsm(
            skip_read_act(0),
            skip_comp(0),
            skip_write_out(1)
        )

        fsm(
            out_count(0)
        )

        state_init = fsm.current

        fsm.goto_next()

        # --------------------
        # ReadAct phase
        # --------------------
        state_read_act = fsm.current

        act_gaddr = self.arg_objaddrs[0] + act_base_offset

        bt.bus_lock(self.maxi, fsm)

        act_laddr = act_page_dma_offset

        begin_state_read = fsm.current
        fsm.goto_next()

        bt.dma_read(self.maxi, fsm, act_ram, act_laddr,
                    act_gaddr, self.act_read_size, port=1)

        end_state_read = fsm.current

        # --------------------
        # Comp phase
        # --------------------
        state_comp = fsm.current

        # Stream Control FSM
        comp_fsm = vg.FSM(self.m, self._name('comp_fsm'), self.clk, self.rst)

        comp_state_init = comp_fsm.current
        comp_fsm.If(fsm.state == state_comp, vg.Not(skip_comp)).goto_next()

        fsm.If(comp_fsm.state == comp_state_init).goto_next()

        # local address
        comp_fsm(
            stream_act_local(self.stream_local),
            stream_out_local(0)
        )

        act_page_comp_offset_buf = self.m.Reg(self._name('act_page_comp_offset_buf'),
                                              self.maxi.addrwidth, initval=0)
        out_page_comp_offset_buf = self.m.Reg(self._name('out_page_comp_offset_buf'),
                                              self.maxi.addrwidth, initval=0)

        comp_fsm(
            act_page_comp_offset_buf(act_page_comp_offset),
            out_page_comp_offset_buf(out_page_comp_offset)
        )

        comp_fsm.goto_next()

        # busy check
        self.stream.source_join(comp_fsm)

        # set_source
        name = list(self.stream.sources.keys())[0]
        local = stream_act_local + act_page_comp_offset_buf

        if len(self.out_shape) > 1:
            pat = ((self.stream_size, self.act_strides[-1]),
                   (self.out_shape[-2], self.stream_stride))
        else:
            pat = ((self.stream_size, self.act_strides[-1]),)

        self.stream.set_source_pattern(comp_fsm, name, act_ram,
                                       local, pat)

        comp_fsm.set_index(comp_fsm.current - 1)

        # set_sink
        name = list(self.stream.sinks.keys())[0]
        local = stream_out_local + out_page_comp_offset_buf

        if len(self.out_shape) > 1:
            pat = ((self.stream_size, 1),
                   (self.out_shape[-2], self.stream_size))
        else:
            pat = ((self.stream_size, 1),)

        self.stream.set_sink_pattern(comp_fsm, name, out_ram,
                                     local, pat)

        # stream run (async)
        self.stream.run(comp_fsm)

        comp_fsm.goto_init()

        # sync with WriteOut control
        comp_fsm.seq.If(fsm.state == state_init)(
            comp_count(0)
        )
        comp_fsm.seq.If(self.stream.end_flag)(
            comp_count.inc()
        )

        # --------------------
        # WriteOut phase
        # --------------------
        state_write_out = fsm.current

        # sync with Comp control
        fsm.If(comp_count > out_count).goto_next()

        out_laddr = out_page_dma_offset
        out_gaddr = self.objaddr + out_offset

        bt.bus_lock(self.maxi, fsm)

        bt.dma_write(self.maxi, fsm, out_ram, out_laddr,
                     out_gaddr, self.out_write_size, port=1, use_async=True)

        bt.bus_unlock(self.maxi, fsm)

        fsm(
            out_count.inc()
        )

        fsm.goto_next()

        state_write_out_end = fsm.current
        fsm.If(skip_write_out).goto_from(state_write_out, state_write_out_end)

        # --------------------
        # update for next iteration
        # --------------------
        # ReadAct: count
        cond = None
        for size, count in zip(reversed(self.out_shape[:-2]), reversed(counts)):

            fsm.If(cond)(
                count.inc()
            )
            fsm.If(cond, count >= size - 1)(
                count(0)
            )
            if cond is not None:
                cond = vg.Ands(cond, count >= size - 1)
            else:
                cond = count >= size - 1

        # ReadAct: offset
        cond = None
        for size, count, act_offset, act_offset_stride in zip(
                reversed(self.out_shape[:-2]), reversed(counts),
                reversed(act_offsets), reversed(self.act_offset_strides)):

            fsm.If(cond)(
                act_offset.add(act_offset_stride)
            )
            fsm.If(cond, count >= size - 1)(
                act_offset(0)
            )
            if cond is not None:
                cond = vg.Ands(cond, count >= size - 1)
            else:
                cond = count >= size - 1

        # ReadAct and Comp: double buffer
        fsm.If(vg.Not(act_page))(
            act_page_comp_offset(act_page_size),
            act_page_dma_offset(act_page_size),
            act_page(1)
        )
        fsm.If(act_page)(
            act_page_comp_offset(0),
            act_page_dma_offset(0),
            act_page(0)
        )

        # WriteOut: offset
        cond = vg.Not(skip_write_out)
        for size, prev_count, out_offset, out_offset_stride in zip(
                reversed(self.out_shape[:-2]), reversed(prev_counts),
                reversed(out_offsets), reversed(self.out_offset_strides)):

            fsm.If(cond)(
                out_offset.add(out_offset_stride)
            )
            fsm.If(cond, prev_count >= size - 1)(
                out_offset(0)
            )
            cond = vg.Ands(cond, prev_count >= size - 1)

        # WriteOut and Comp: double buffer
        fsm.If(vg.Not(out_page))(
            out_page_comp_offset(out_page_size),
            out_page_dma_offset(0),
            out_page(1)
        )
        fsm.If(out_page)(
            out_page_comp_offset(0),
            out_page_dma_offset(out_page_size),
            out_page(0)
        )

        # ReadAct and WriteOut: prev
        for count, prev_count in zip(counts, prev_counts):
            fsm(
                prev_count(count)
            )

        # ReadAct, Comp, WriteOut: skip
        cond_skip_read_act = None
        cond_skip_comp = None
        for size, count in zip(reversed(self.out_shape[:-2]), reversed(counts)):
            if cond_skip_read_act is not None:
                cond_skip_read_act = vg.Ands(cond_skip_read_act, count >= size - 1)
            else:
                cond_skip_read_act = count >= size - 1

        cond_skip_comp = cond_skip_read_act

        cond_cancel_write_out = None
        for size, prev_count in zip(reversed(self.out_shape[:-2]), reversed(prev_counts)):
            if cond_cancel_write_out is not None:
                cond_cancel_write_out = vg.Ands(cond_cancel_write_out, prev_count == 0)
            else:
                cond_cancel_write_out = prev_count == 0

        cond_done = None
        for size, prev_count in zip(reversed(self.out_shape[:-2]), reversed(prev_counts)):
            if cond_done is not None:
                cond_done = vg.Ands(cond_done, prev_count >= size - 1)
            else:
                cond_done = prev_count >= size - 1

        fsm.If(cond_skip_read_act)(
            skip_read_act(1)
        )
        fsm.If(cond_skip_comp)(
            skip_comp(1)
        )
        fsm.If(skip_write_out,
               cond_cancel_write_out)(
            skip_write_out(0)
        )

        fsm.goto(state_read_act)
        fsm.If(vg.Not(skip_write_out), cond_done).goto_next()

        # wait for last DMA write
        bt.dma_wait_write(self.maxi, fsm)

    def eval(self, memo, input_dict, **kwargs):
        if id(self) in memo:
            return memo[id(self)]

        import nngen.verify as verify

        name = self.__class__.__name__
        method = getattr(verify, name, None)

        args = [arg.eval(memo, input_dict)
                for arg in self.args]

        value = args[0]

        kwargs['begins'] = self.begins
        kwargs['ends'] = self.ends
        kwargs['strides'] = self.strides
        kwargs['dtype'] = self.dtype
        kwargs['name'] = self.name
        kwargs['par'] = self.par

        ret = method(value, **kwargs)
        memo[id(self)] = ret

        return ret


def to_slices(begins, ends, strides):
    slices = []

    for begin, end, stride in zip(begins, ends, strides):
        slices.append(slice(begin, end, stride))

    return tuple(slices)
