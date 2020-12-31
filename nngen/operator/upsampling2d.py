from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
from collections import OrderedDict

import veriloggen as vg

import nngen.basic_types as bt
import nngen.util as util


class upsampling2d(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = False

    @staticmethod
    def op(strm, *args, **kwargs):
        return args[0]

    def __init__(self, value, factors, dtype=None, name=None, par=1):

        if len(value.shape) != 4:
            raise ValueError('rank of value.shape must be 4.')

        if len(factors) != 4:
            raise ValueError('rank of factors must be 4.')

        if factors[0] != 1 or factors[3] != 1:
            raise ValueError('factors[0] and [3] must be 1')

        shape = [s * f for s, f in zip(value.shape, factors)]

        bt._ElementwiseOperator.__init__(self, value,
                                         dtype=dtype, shape=shape, name=name, par=par)
        self.factors = factors

    def get_control_param_values(self):
        orig_shape = self.args[0].shape

        num_words = self.get_word_alignment()

        aligned_shape = []
        for s in orig_shape[:-1]:
            aligned_shape.append(s)

        res = num_words - orig_shape[-1] % num_words

        if res == num_words:
            res = 0

        aligned_shape.append(orig_shape[-1] + res)

        aligned_length = bt.shape_to_length(aligned_shape)

        total_size = int(math.ceil(aligned_length / self.par))
        dma_size = int(math.ceil(aligned_shape[-1] / self.par))
        num_comp = int(math.ceil(total_size / dma_size))

        base = bt.to_byte(bt.align_word(orig_shape[-1], self.get_word_alignment()) *
                          self.get_ram_width())

        factor_col = self.factors[2]
        factor_row = self.factors[1]

        out_col_step = base
        out_row_step = base * (self.shape[-2] - (factor_col - 1))
        max_out_pos_col = factor_col - 1
        max_out_pos_row = factor_row - 1

        out_col_inc = base * factor_col
        out_row_inc = out_col_inc + base * self.shape[-2] * (factor_row - 1)
        max_out_col_count = orig_shape[-2] - 1

        sources = self.collect_sources()

        arg_addr_incs = []
        wrap_modes = []
        wrap_sizes = []
        for arg in sources:
            arg_addr_inc = bt.to_byte(bt.align_word(arg.shape[-1], arg.get_word_alignment()) *
                                      arg.get_ram_width())
            if tuple(arg.shape) == tuple(orig_shape):
                wrap_mode = 0
                wrap_size = 0
            elif len(arg.shape) == 1 and arg.shape[-1] == 1:
                # stride-0
                wrap_mode = 2
                wrap_size = bt.get_wrap_size(orig_shape, arg.shape)
            else:
                # repeat
                wrap_mode = 1
                wrap_size = bt.get_wrap_size(orig_shape, arg.shape)
            arg_addr_incs.append(arg_addr_inc)
            wrap_modes.append(wrap_mode)
            wrap_sizes.append(wrap_size)

        return OrderedDict([('dma_size', dma_size),
                            ('num_comp', num_comp),
                            ('out_col_step', out_col_step),
                            ('out_row_step', out_row_step),
                            ('max_out_pos_col', max_out_pos_col),
                            ('max_out_pos_row', max_out_pos_row),
                            ('out_col_inc', out_col_inc),
                            ('out_row_inc', out_row_inc),
                            ('max_out_col_count', max_out_col_count),
                            ('arg_addr_incs', arg_addr_incs),
                            ('wrap_modes', wrap_modes),
                            ('wrap_sizes', wrap_sizes)])

    def control_sequence(self, fsm):
        sources = self.collect_sources()

        arg_gaddrs = [self.m.Reg(self._name('arg_gaddr_%d' % i),
                                 self.maxi.addrwidth, initval=0)
                      for i, _ in enumerate(self.arg_objaddrs)]
        out_gaddr = self.m.Reg(self._name('out_gaddr'),
                               self.maxi.addrwidth, initval=0)
        out_gaddr_offset = self.m.Reg(self._name('out_gaddr_offset'),
                                      self.maxi.addrwidth, initval=0)
        out_pos_col = self.m.Reg(self._name('out_pos_col'),
                                 self.maxi.addrwidth + 1, initval=0)
        out_pos_row = self.m.Reg(self._name('out_pos_row'),
                                 self.maxi.addrwidth + 1, initval=0)
        out_col_count = self.m.Reg(self._name('out_col_count'),
                                   self.maxi.addrwidth + 1, initval=0)
        comp_count = self.m.Reg(self._name('comp_count'),
                                self.maxi.addrwidth + 1, initval=0)
        wrap_counts = [self.m.Reg(self._name('wrap_count_%d' % i),
                                  self.maxi.addrwidth + 1, initval=0)
                       for i, arg in enumerate(sources)]

        arg_pages = [self.m.Reg(self._name('arg_page_%d' % i), initval=0)
                     for i, _ in enumerate(self.arg_objaddrs)]
        arg_page_comp_offsets = [self.m.Reg(self._name('arg_page_comp_offset_%d' % i),
                                            self.maxi.addrwidth, initval=0)
                                 for i, _ in enumerate(self.arg_objaddrs)]
        arg_page_dma_offsets = [self.m.Reg(self._name('arg_page_dma_offset_%d' % i),
                                           self.maxi.addrwidth, initval=0)
                                for i, _ in enumerate(self.arg_objaddrs)]

        out_page = self.m.Reg(self._name('out_page'), initval=0)
        out_page_comp_offset = self.m.Reg(self._name('out_page_comp_offset'),
                                          self.maxi.addrwidth, initval=0)
        out_page_dma_offset = self.m.Reg(self._name('out_page_dma_offset'),
                                         self.maxi.addrwidth, initval=0)

        arg_page_size = self.output_rams[0].length // 2
        out_page_size = self.output_rams[0].length // 2

        skip_read = self.m.Reg(self._name('skip_read'), initval=0)
        skip_comp = self.m.Reg(self._name('skip_comp'), initval=0)
        skip_write = self.m.Reg(self._name('skip_write'), initval=0)

        # --------------------
        # initialization phase
        # --------------------
        fsm(
            [arg_gaddr(0) for arg_gaddr in arg_gaddrs]
        )

        fsm(
            comp_count(0),
            out_gaddr(0),
            out_gaddr_offset(0),
            out_pos_col(0),
            out_pos_row(0),
            out_col_count(0),
            [wrap_count(0) for wrap_count in wrap_counts]
        )

        fsm(
            [arg_page(0) for arg_page in arg_pages],
            [arg_page_comp_offset(0)
             for arg_page_comp_offset in arg_page_comp_offsets],
            [arg_page_dma_offset(0)
             for arg_page_dma_offset in arg_page_dma_offsets]
        )

        fsm(
            out_page(0),
            out_page_comp_offset(0),
            out_page_dma_offset(out_page_size)
        )

        fsm(
            skip_read(0),
            skip_comp(0),
            skip_write(1)
        )

        fsm.goto_next()

        # --------------------
        # Read phase
        # --------------------
        state_read = fsm.current

        # DMA read -> Stream run -> Stream wait -> DMA write
        for (ram, arg_objaddr,
             arg_gaddr, arg_page_dma_offset,
             wrap_mode, wrap_count, arg) in zip(self.input_rams, self.arg_objaddrs,
                                                arg_gaddrs, arg_page_dma_offsets,
                                                self.wrap_modes, wrap_counts, sources):

            b = fsm.current
            fsm.goto_next()

            # normal
            laddr = arg_page_dma_offset
            gaddr = arg_objaddr + arg_gaddr
            bt.bus_lock(self.maxi, fsm)
            bt.dma_read(self.maxi, fsm, ram, laddr, gaddr, self.dma_size)
            bt.bus_unlock(self.maxi, fsm)
            fsm.goto_next()

            b_stride0 = fsm.current
            fsm.goto_next()

            # stride-0
            bt.bus_lock(self.maxi, fsm)
            bt.dma_read(self.maxi, fsm, ram, laddr, gaddr, 1)
            bt.bus_unlock(self.maxi, fsm)
            fsm.goto_next()

            # for reuse
            e = fsm.current
            fsm.If(wrap_mode == 2, wrap_count > 0).goto_from(b, e)
            fsm.If(wrap_mode == 2, wrap_count == 0).goto_from(b, b_stride0)
            fsm.If(wrap_mode != 2).goto_from(b_stride0, e)

        state_read_end = fsm.current
        fsm.If(skip_read).goto_from(state_read, state_read_end)

        # --------------------
        # Comp phase
        # --------------------
        state_comp = fsm.current

        self.stream.source_join(fsm)

        # set_source, set_parameter (dup)
        for (source_name, dup_name,
             arg_page_comp_offset,
             ram, wrap_mode) in zip(self.stream.sources.keys(),
                                    self.stream.parameters.keys(),
                                    arg_page_comp_offsets,
                                    self.input_rams, self.wrap_modes):
            read_laddr = arg_page_comp_offset
            read_size = self.dma_size
            stride = vg.Mux(wrap_mode == 2, 0, 1)
            dup = vg.Mux(wrap_mode == 2, 1, 0)
            self.stream.set_parameter(fsm, dup_name, dup)
            fsm.set_index(fsm.current - 1)
            self.stream.set_source(fsm, source_name, ram,
                                   read_laddr, read_size, stride)
            fsm.set_index(fsm.current - 1)

        # set_sink
        write_laddr = out_page_comp_offset
        write_size = self.dma_size

        for name, ram in zip(self.stream.sinks.keys(), self.output_rams):
            self.stream.set_sink(fsm, name, ram, write_laddr, write_size)
            fsm.set_index(fsm.current - 1)

        fsm.goto_next()

        self.stream.run(fsm)

        state_comp_end = fsm.current

        self.stream.join(fsm)

        state_comp_end_join = fsm.current

        fsm.If(skip_comp).goto_from(state_comp, state_comp_end)
        fsm.If(vg.Not(skip_comp)).goto_from(
            state_comp_end, state_comp_end_join)

        # --------------------
        # Write phase
        # --------------------
        state_write = fsm.current

        laddr = out_page_dma_offset
        gaddr_base = self.objaddr + out_gaddr

        bt.bus_lock(self.maxi, fsm)

        b = fsm.current

        gaddr = gaddr_base + out_gaddr_offset
        bt.dma_write(self.maxi, fsm,
                     self.output_rams[0], laddr, gaddr, self.dma_size, use_async=True)

        fsm(
            out_pos_col.inc(),
            out_gaddr_offset.add(self.out_col_step),
        )
        fsm.If(out_pos_col == self.max_out_pos_col)(
            out_pos_col(0),
            out_pos_row.inc(),
            out_gaddr_offset.add(self.out_row_step)
        )

        fsm.goto(b)
        fsm.If(out_pos_col == self.max_out_pos_col,
               out_pos_row == self.max_out_pos_row).goto_next()

        bt.bus_unlock(self.maxi, fsm)

        fsm.goto_next()

        state_write_end = fsm.current
        fsm.If(skip_write).goto_from(state_write, state_write_end)

        # --------------------
        # update for next iteration
        # --------------------
        fsm(
            comp_count.inc()
        )

        fsm(
            out_gaddr_offset(0),
            out_pos_col(0),
            out_pos_row(0)
        )

        fsm.If(vg.Not(skip_write))(
            out_gaddr.add(self.out_col_inc),
            out_col_count.inc(),
        )
        fsm.If(vg.Not(skip_write), out_col_count == self.max_out_col_count)(
            out_gaddr.add(self.out_row_inc),
            out_col_count(0),
        )

        for (arg_gaddr, arg_addr_inc,
             arg_page, arg_page_comp_offset, arg_page_dma_offset,
             wrap_mode, wrap_size,
             wrap_count, arg) in zip(arg_gaddrs, self.arg_addr_incs,
                                     arg_pages, arg_page_comp_offsets,
                                     arg_page_dma_offsets,
                                     self.wrap_modes, self.wrap_sizes,
                                     wrap_counts, sources):

            fsm.If(wrap_mode == 2)(
                wrap_count(1)
            )

            fsm.If(wrap_mode == 1)(
                arg_gaddr.add(arg_addr_inc),
                wrap_count.inc()
            )
            fsm.If(wrap_mode == 1, wrap_count == wrap_size - 1)(
                arg_gaddr(0),
                wrap_count(0)
            )

            fsm.If(wrap_mode == 0)(
                arg_gaddr.add(arg_addr_inc)
            )

            fsm.If(vg.Not(arg_page), wrap_mode != 2)(
                arg_page_comp_offset(arg_page_size),
                arg_page_dma_offset(out_page_size),
                arg_page(1)
            )
            fsm.If(arg_page, wrap_mode != 2)(
                arg_page_comp_offset(0),
                arg_page_dma_offset(0),
                arg_page(0)
            )

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

        fsm(
            skip_write(0)
        )
        fsm.If(comp_count == self.num_comp - 1)(
            skip_read(1),
            skip_comp(1)
        )

        fsm.If(comp_count < self.num_comp).goto(state_read)
        fsm.If(comp_count == self.num_comp).goto_next()

        # wait for last DMA write
        bt.dma_wait_write(self.maxi, fsm)

    def eval(self, memo, input_dict, **kwargs):
        if id(self) in memo:
            return memo[id(self)]

        import nngen.verify as verify

        name = self.__class__.__name__

        args = [arg.eval(memo, input_dict)
                for arg in self.args]

        value = args[0]
        factors = self.factors

        kwargs['factors'] = self.factors
        kwargs['dtype'] = self.dtype
        kwargs['name'] = self.name
        kwargs['par'] = self.par

        method = self.get_eval_method()
        ret = method(value, **kwargs)
        memo[id(self)] = ret

        return ret
