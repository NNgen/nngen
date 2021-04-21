from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
from collections import OrderedDict

import veriloggen as vg

import nngen.basic_types as bt
import nngen.util as util


class _pool(bt._Operator):
    # shape order
    # - value:  'NHWC' (batch, height, width, channel)
    # - ksize:  'NHWC' (batch, height, width, channel)

    input_chainable = False
    output_chainable = False

    control_param_custom_width = {'act_offset_values': bt.get_maxi_addrwidth}
    control_param_custom_signed = {'act_offset_values': True}

    def __sub_str__(self):
        ksize = str(self.ksize)
        strides = str(self.strides)
        if hasattr(self, 'pad_col_left') and self.padding == 'SAME':
            padding = ("'%s'-(%d, %d, %d, %d)" %
                       (self.padding,
                        self.pad_row_top_value, self.pad_row_bottom_value,
                        self.pad_col_left_value, self.pad_col_right_value))
        else:
            padding = self.padding

        par = ' par:%d' % self.par if self.par > 1 else ''

        value_ram_size = (' value_ram_size:%d' % self.value_ram_size
                          if self.value_ram_size is not None else '')
        out_ram_size = (' out_ram_size:%d' % self.out_ram_size
                        if self.out_ram_size is not None else '')

        return (' ksize:%s strides:%s padding:%s%s%s%s' %
                (ksize, strides, padding, par, value_ram_size, out_ram_size))

    def __init__(self, value, ksize, strides, padding='SAME',
                 dtype=None, name=None, par=1,
                 value_ram_size=None, out_ram_size=None):

        if isinstance(padding, str) and padding != 'SAME' and padding != 'VALID':
            raise ValueError("padding options must be 'SAME', 'VALID', int, tuple, or list.")
        elif isinstance(padding, (tuple, list)) and len(padding) != 4:
            raise ValueError('padding rank must be 4.')

        if bt.get_rank(value.shape) != 4:
            raise ValueError('rank of value must be 4.')

        if len(ksize) != 4:
            raise ValueError('rank of ksize must be 4.')

        if len(strides) != 4:
            raise ValueError('rank of strides must be 4.')

        if ksize[0] != 1 or ksize[3] != 1:
            raise ValueError('ksize[0] and [3] must be 1')

        if strides[0] != 1 or strides[3] != 1:
            raise ValueError('strides[0] and [3] must be 1')

        if isinstance(padding, str) and (padding == 'SAME' or padding == 'VALID'):
            shape = []
            shape.append(int(math.ceil(value.shape[0] / strides[0])))
            for sh, st, fs in list(zip(value.shape, strides, ksize))[1:-1]:
                shape.append(util.pix_size(sh, fs, st, padding))
            shape.append(int(math.ceil(value.shape[3] / strides[3])))
        elif isinstance(padding, int):
            shape = []
            shape.append(int(math.ceil(value.shape[0] / strides[0])))
            for sh, st, fs in list(zip(value.shape, strides, ksize))[1:-1]:
                shape.append(util.pix_size(sh + padding * 2, fs, st, 'VALID'))
            shape.append(int(math.ceil(value.shape[3] / strides[3])))
        elif isinstance(padding, (tuple, list)):
            shape = []
            shape.append(int(math.ceil(value.shape[0] / strides[0])))
            for i, (sh, st, fs) in enumerate(
                    list(zip(value.shape, strides, ksize))[1:-1]):
                pd0 = padding[i * 2]
                pd1 = padding[i * 2 + 1]
                shape.append(util.pix_size(sh + pd0 + pd1, fs, st, 'VALID'))
            shape.append(int(math.ceil(value.shape[3] / strides[3])))

        shape = tuple(shape)

        if value_ram_size is not None and value_ram_size < 1:
            raise ValueError('value_ram_size must be greater than 0')

        if out_ram_size is not None and out_ram_size < 1:
            raise ValueError('out_ram_size must be greater than 0')

        bt._Operator.__init__(self, value,
                              dtype=dtype, shape=shape, name=name, par=par)

        self.ksize = tuple(ksize)
        self.strides = tuple(strides)
        self.padding = padding

        # attribute
        self.value_ram_size = value_ram_size
        self.out_ram_size = out_ram_size
        _pool.attribute(self, par, value_ram_size, out_ram_size)

    def get_pad_value(self, strm):
        raise NotImplementedError('not implemented')

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
        ksize_ch = self.ksize[-1]
        ksize_col = self.ksize[-2]
        ksize_row = self.ksize[-3]
        ksize_bat = self.ksize[-4]

        act = self.args[0]
        act_shape = act.get_aligned_shape()
        act_num_ch = act_shape[-1]
        act_num_col = act_shape[-2]
        act_num_row = act_shape[-3]
        act_num_bat = act_shape[-4]

        out_shape = self.get_aligned_shape()
        out_num_ch = out_shape[-1]
        out_num_col = out_shape[-2]
        out_num_row = out_shape[-3]
        out_num_bat = out_shape[-4]

        input_min_size = (int(math.ceil(act_num_ch / self.par)) *
                          int(math.ceil(act_num_col / ksize_col)) * 2)
        if self.value_ram_size is not None and input_min_size < self.value_ram_size:
            input_min_size = self.value_ram_size
        input_width = act.get_ram_width() * self.par

        output_min_size = (int(math.ceil(out_num_ch / self.par)) *
                           out_num_col * 2)
        if self.out_ram_size is not None and output_min_size < self.out_ram_size:
            output_min_size = self.out_ram_size
        output_width = self.get_ram_width() * self.par

        inputs = []
        inputs.extend([(input_width, input_min_size)] * ksize_col * ksize_row)

        outputs = []
        outputs.append((output_width, output_min_size))

        temps = []

        return inputs, outputs, temps

    def get_stream_hash(self):
        base = bt._Operator.get_stream_hash(self)
        ksize_col = self.ksize[-2]
        ksize_row = self.ksize[-3]
        return (base, ksize_col, ksize_row, self.par)

    def get_stream_func(self):

        def func(strm):
            ksize_col = self.ksize[-2]
            ksize_row = self.ksize[-3]
            len_act_rams = len(self.input_rams)

            mask = strm.constant(datawidth=len_act_rams, signed=False)

            act_rams = self.input_rams
            out_ram = self.output_rams[0]

            # vec_act
            arg = self.args[0]
            datawidth = arg.get_op_width()
            vec_datawidth = datawidth * self.par
            point = arg.get_op_point()
            signed = arg.get_signed()

            vec_act_vars = []
            for act_ram in act_rams:
                vec_act_var = strm.source(datawidth=vec_datawidth, signed=False)
                vec_act_vars.append(vec_act_var)

            # vec_act -> act
            act_vars_list = []
            split_vec_act_vars = [strm.Split(vec_act_var,
                                             datawidth, point, signed, reverse=True)
                                  for vec_act_var in vec_act_vars]

            for i in range(self.par):
                if self.par == 1:
                    act_vars = [strm.ReinterpretCast(vec_act_var, datawidth, point, signed)
                                for vec_act_var in vec_act_vars]
                else:
                    act_vars = [split_vec_act_var[i]
                                for split_vec_act_var in split_vec_act_vars]

                act_vars_list.append(act_vars)

            if len(vec_act_vars) > mask.bit_length():
                raise ValueError('Not enough mask bits.')

            pad_value = self.get_pad_value(strm)

            # vector parallel processing
            out_vars = []

            for i, act_vars in enumerate(act_vars_list):
                masked_vars = []
                for act_var, pmask in zip(act_vars, mask):
                    masked_var = strm.Mux(pmask, pad_value, act_var)
                    masked_vars.append(masked_var)

                out_var = self.pool_op(strm, i, *masked_vars)

                width = self.get_op_width()
                point = self.get_op_point()
                signed = self.get_signed()

                out_var = bt.out_rcast(strm, out_var, width, point, signed)
                out_vars.append(out_var)

            if self.par == 1:
                vec_out_var = out_vars[0]
            else:
                vec_out_var = strm.Cat(*reversed(out_vars))

            strm.sink(vec_out_var)

        return func

    def pool_op(self, strm, *vars):
        # return value
        raise NotImplementedError('not implemented')

    def get_control_param_values(self):
        act = self.args[0]

        ksize_ch = self.ksize[-1]
        ksize_col = self.ksize[-2]
        ksize_row = self.ksize[-3]
        ksize_bat = self.ksize[-4]

        act_shape = act.get_aligned_shape()
        act_num_ch = act_shape[-1]
        act_num_col = act_shape[-2]
        act_num_row = act_shape[-3]
        act_num_bat = act_shape[-4]

        # stride_ch = self.strides[-1]  # always 1
        stride_col = self.strides[-2]  # width
        stride_row = self.strides[-3]  # height
        stride_bat = self.strides[-4]  # always 1

        out_shape = self.get_aligned_shape()
        out_num_ch = out_shape[-1]
        out_num_col = out_shape[-2]
        out_num_row = out_shape[-3]
        out_num_bat = out_shape[-4]

        if isinstance(self.padding, str) and self.padding == 'SAME':
            pad_col, pad_col_left, pad_col_right = util.pad_size_split(
                act_num_col, ksize_col, stride_col)
            pad_row, pad_row_top, pad_row_bottom = util.pad_size_split(
                act_num_row, ksize_row, stride_row)
        elif isinstance(self.padding, int):
            pad_col = self.padding * 2
            pad_col_left = self.padding
            pad_col_right = self.padding
            pad_row = self.padding * 2
            pad_row_top = self.padding
            pad_row_bottom = self.padding
        elif isinstance(self.padding, (tuple, list)):
            pad_col = self.padding[2] + self.padding[3]
            pad_col_left = self.padding[2]
            pad_col_right = self.padding[3]
            pad_row = self.padding[0] + self.padding[1]
            pad_row_top = self.padding[0]
            pad_row_bottom = self.padding[1]
        else:
            pad_col = 0
            pad_col_left = 0
            pad_col_right = 0
            pad_row = 0
            pad_row_top = 0
            pad_row_bottom = 0

        # for __str__
        self.pad_col_left_value = pad_col_left
        self.pad_col_right_value = pad_col_right
        self.pad_row_top_value = pad_row_top
        self.pad_row_bottom_value = pad_row_bottom

        max_col_count = act_num_col + pad_col + 1 - ksize_col - stride_col
        if max_col_count < 0:
            max_col_count = 0

        max_row_count = act_num_row + pad_row + 1 - ksize_row - stride_row
        if max_row_count < 0:
            max_row_count = 0

        max_bat_count = act_num_bat - stride_bat
        if max_bat_count < 0:
            max_bat_count = 0

        dma_flag_conds = []
        for row_select in range(ksize_row):
            v = False
            for i in range(stride_row):
                v = v or (row_select == (i % ksize_row))

            dma_flag_conds.append(v)

        aligned_act_num_ch = bt.align_word(act_num_ch,
                                           act.get_word_alignment())

        act_step = bt.to_byte(aligned_act_num_ch * act.get_ram_width())

        act_offset_values = []
        for y in range(ksize_row):
            v = act_num_col * (y - pad_row_top) * act_step
            act_offset_values.append(v)

        act_row_step = act_step * act_num_col * stride_row
        act_bat_step = act_step * act_num_col * act_num_row

        act_read_size = (int(math.ceil(aligned_act_num_ch / self.par)) *
                         act_num_col)
        act_read_block = int(math.ceil(aligned_act_num_ch / self.par))

        out_step = bt.to_byte(bt.align_word(out_num_ch, self.get_word_alignment()) *
                              self.get_ram_width())

        out_row_step = out_step * out_num_col
        out_bat_step = out_step * out_num_col * out_num_row

        out_write_size = (int(math.ceil(out_num_ch / self.par)) *
                          out_num_col)

        stream_size = int(math.ceil(act_num_ch / self.par))

        if pad_col_left == 0:
            col_select_initval = 0
        else:
            col_select_initval = (ksize_col - pad_col_left) % ksize_col

        stride_col_mod_ksize = stride_col % ksize_col
        ksize_col_minus_stride_col_mod = ksize_col - stride_col_mod_ksize

        inc_act_laddr_conds = []
        for y in range(ksize_row):
            for x in range(ksize_col):
                for col_select in range(ksize_col):
                    v = False
                    for i in range(stride_col_mod_ksize):
                        v = v or (col_select == ((x + ksize_col - i) %
                                                 ksize_col))

                    inc_act_laddr_conds.append(v)

        inc_act_laddr_small = (int(math.floor(stride_col / ksize_col)) *
                               act_read_block)
        inc_act_laddr_large = (int(math.ceil(stride_col / ksize_col)) *
                               act_read_block)
        inc_out_laddr = int(math.ceil(out_num_ch / self.par))

        stream_act_local_small_offset = (-1 * int(math.floor(pad_col_left / ksize_col)) *
                                         act_read_block)
        stream_act_local_large_offset = (-1 * int(math.ceil(pad_col_left / ksize_col)) *
                                         act_read_block)

        stream_act_local_small_flags = []
        stream_act_local_large_flags = []
        for x in range(ksize_col):
            s = (ksize_col - x) <= pad_col_left
            l = (ksize_col - x) <= (pad_col_left % ksize_col)
            stream_act_local_small_flags.append(s)
            stream_act_local_large_flags.append(s and l)

        return OrderedDict([('act_num_col', act_num_col),
                            ('act_num_row', act_num_row),
                            ('stride_col', stride_col),
                            ('stride_row', stride_row),
                            ('out_num_col', out_num_col),
                            ('out_num_row', out_num_row),
                            ('pad_col_left', pad_col_left),
                            ('pad_row_top', pad_row_top),
                            ('max_col_count', max_col_count),
                            ('max_row_count', max_row_count),
                            ('max_bat_count', max_bat_count),
                            ('dma_flag_conds', dma_flag_conds),
                            ('act_offset_values', act_offset_values),
                            ('act_row_step', act_row_step),
                            ('act_bat_step', act_bat_step),
                            ('act_read_size', act_read_size),
                            ('act_read_block', act_read_block),
                            ('out_row_step', out_row_step),
                            ('out_bat_step', out_bat_step),
                            ('out_write_size', out_write_size),
                            ('stream_size', stream_size),
                            ('col_select_initval', col_select_initval),
                            ('stride_col_mod_ksize', stride_col_mod_ksize),
                            ('ksize_col_minus_stride_col_mod', ksize_col_minus_stride_col_mod),
                            ('inc_act_laddr_conds', inc_act_laddr_conds),
                            ('inc_act_laddr_small', inc_act_laddr_small),
                            ('inc_act_laddr_large', inc_act_laddr_large),
                            ('inc_out_laddr', inc_out_laddr),
                            ('stream_act_local_small_offset', stream_act_local_small_offset),
                            ('stream_act_local_large_offset', stream_act_local_large_offset),
                            ('stream_act_local_small_flags', stream_act_local_small_flags),
                            ('stream_act_local_large_flags', stream_act_local_large_flags)])

    def control_sequence(self, fsm):
        ksize_ch = self.ksize[-1]
        ksize_col = self.ksize[-2]
        ksize_row = self.ksize[-3]
        ksize_bat = self.ksize[-4]

        self.stride_bat = 1

        act_rams = self.input_rams
        out_ram = self.output_rams[0]

        act_base_offset = self.m.Wire(self._name('act_base_offset'),
                                      self.maxi.addrwidth, signed=True)
        act_base_offset_row = self.m.Reg(self._name('act_base_offset_row'),
                                         self.maxi.addrwidth, initval=0, signed=True)
        act_base_offset_bat = self.m.Reg(self._name('act_base_offset_bat'),
                                         self.maxi.addrwidth, initval=0, signed=True)

        act_base_offset.assign(act_base_offset_row +
                               act_base_offset_bat)

        out_base_offset = self.m.Wire(self._name('out_base_offset'),
                                      self.maxi.addrwidth, signed=True)
        out_base_offset_row = self.m.Reg(self._name('out_base_offset_row'),
                                         self.maxi.addrwidth, initval=0, signed=True)
        out_base_offset_bat = self.m.Reg(self._name('out_base_offset_bat'),
                                         self.maxi.addrwidth, initval=0, signed=True)

        out_base_offset.assign(out_base_offset_row +
                               out_base_offset_bat)

        dma_flags = [self.m.Reg(self._name('dma_flag_%d' % i), initval=0)
                     for i in range(ksize_row)]

        col_count = self.m.Reg(self._name('col_count'),
                               self.maxi.addrwidth, initval=0)
        row_count = self.m.Reg(self._name('row_count'),
                               self.maxi.addrwidth, initval=0)
        bat_count = self.m.Reg(self._name('bat_count'),
                               self.maxi.addrwidth, initval=0)
        col_select = self.m.Reg(self._name('col_select'),
                                bt.log_width(ksize_col),
                                initval=0)
        row_select = self.m.Reg(self._name('row_select'),
                                bt.log_width(ksize_row),
                                initval=0)

        prev_row_count = self.m.Reg(self._name('prev_row_count'),
                                    self.maxi.addrwidth, initval=0)
        prev_bat_count = self.m.Reg(self._name('prev_bat_count'),
                                    self.maxi.addrwidth, initval=0)
        prev_row_select = self.m.Reg(self._name('prev_row_select'),
                                     bt.log_width(ksize_row),
                                     initval=0)

        stream_act_locals = [self.m.Reg(self._name('stream_act_local_%d' % i),
                                        self.maxi.addrwidth, initval=0)
                             for i in range(len(act_rams))]
        stream_out_local = self.m.Reg(self._name('stream_out_local'),
                                      self.maxi.addrwidth, initval=0)

        # double buffer control
        act_pages = [self.m.Reg(self._name('act_page_%d' % i), initval=0)
                     for i in range(ksize_row)]
        act_page_comp_offsets = [self.m.Reg(self._name('act_page_comp_offset_%d' % i),
                                            self.maxi.addrwidth, initval=0)
                                 for i in range(ksize_row)]
        act_page_dma_offsets = [self.m.Reg(self._name('act_page_dma_offset_%d' % i),
                                           self.maxi.addrwidth, initval=0)
                                for i in range(ksize_row)]

        out_page = self.m.Reg(self._name('out_page'), initval=0)
        out_page_comp_offset = self.m.Reg(self._name('out_page_comp_offset'),
                                          self.maxi.addrwidth, initval=0)
        out_page_dma_offset = self.m.Reg(self._name('out_page_dma_offset'),
                                         self.maxi.addrwidth, initval=0)

        act_page_size = act_rams[0].length // 2
        out_page_size = out_ram.length // 2

        skip_read_act = self.m.Reg(self._name('skip_read_act'), initval=0)
        skip_comp = self.m.Reg(self._name('skip_comp'), initval=0)
        skip_write_out = self.m.Reg(self._name('skip_write_out'), initval=0)

        comp_count = self.m.Reg(self._name('comp_count'),
                                self.maxi.addrwidth, initval=0)
        out_count = self.m.Reg(self._name('out_count'),
                               self.maxi.addrwidth, initval=0)

        # --------------------
        # initialization phase
        # --------------------
        # ReadAct: offset
        fsm(
            act_base_offset_row(0),
            act_base_offset_bat(0)
        )

        act_offsets = []
        for v in self.act_offset_values:
            act_offset = act_base_offset + v
            act_offsets.append(act_offset)

        # ReadAct: DMA flag
        for y, dma_flag in enumerate(dma_flags):
            fsm(
                dma_flag(1)
            )

        dma_pad_masks = []

        for y in range(ksize_row):
            v = vg.Ors((row_count + y < self.pad_row_top),
                       (row_count + y >= self.act_num_row + self.pad_row_top))
            dma_pad_mask = self.m.Wire(
                self._name('dma_pad_mask_%d' % y))
            dma_pad_mask.assign(v)
            dma_pad_masks.append(dma_pad_mask)

        # ReadAct: double buffer control
        fsm(
            [act_page(0) for act_page in act_pages],
            [act_page_comp_offset(0)
             for act_page_comp_offset in act_page_comp_offsets],
            [act_page_dma_offset(0)
             for act_page_dma_offset in act_page_dma_offsets]
        )

        # WriteOutput: offset
        fsm(
            out_base_offset_row(0),
            out_base_offset_bat(0)
        )

        out_offset = out_base_offset

        # WriteOut: double buffer control
        fsm(
            out_page(0),
            out_page_comp_offset(0),
            out_page_dma_offset(0)
        )

        # counter
        fsm(
            row_count(0),
            bat_count(0),
            row_select(0),
            prev_row_count(0),
            prev_bat_count(0),
            prev_row_select(0)
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

        act_gaddrs = []
        for act_offset in act_offsets:
            act_gaddr = self.arg_objaddrs[0] + act_offset
            act_gaddrs.append(act_gaddr)

        act_rams_2d = line_to_2d(act_rams, ksize_col)
        mux_act_gaddr_values = mux_1d(act_gaddrs, row_select, ksize_row)
        mux_act_gaddrs = []
        for i, mux_act_gaddr_value in enumerate(mux_act_gaddr_values):
            mux_act_gaddr = self.m.Wire(self._name('mux_act_gaddr_%d' % i),
                                        self.maxi.addrwidth)
            mux_act_gaddr.assign(mux_act_gaddr_value)
            mux_act_gaddrs.append(mux_act_gaddr)

        mux_dma_pad_mask_values = mux_1d(dma_pad_masks, row_select, ksize_row)
        mux_dma_pad_masks = []
        for i, mux_dma_pad_mask_value in enumerate(mux_dma_pad_mask_values):
            mux_dma_pad_mask = self.m.Wire(
                self._name('mux_dma_pad_mask_%d' % i))
            mux_dma_pad_mask.assign(mux_dma_pad_mask_value)
            mux_dma_pad_masks.append(mux_dma_pad_mask)

        # determined at the previous phase
        mux_dma_flag_values = mux_1d(dma_flags, prev_row_select, ksize_row)
        mux_dma_flags = []
        for i, mux_dma_flag_value in enumerate(mux_dma_flag_values):
            mux_dma_flag = self.m.Wire(self._name('mux_dma_flag_%d' % i))
            mux_dma_flag.assign(mux_dma_flag_value)
            mux_dma_flags.append(mux_dma_flag)

        bt.bus_lock(self.maxi, fsm)

        for (act_rams_row, act_gaddr, act_page_dma_offset,
             dma_pad_mask, dma_flag) in zip(act_rams_2d, mux_act_gaddrs,
                                            act_page_dma_offsets,
                                            mux_dma_pad_masks, mux_dma_flags):
            act_laddr = act_page_dma_offset

            begin_state_read = fsm.current
            fsm.goto_next()

            if len(act_rams_row) == 1:
                bt.dma_read(self.maxi, fsm, act_rams_row[0], act_laddr,
                            act_gaddr, self.act_read_size, port=1)
            else:
                bt.dma_read_block(self.maxi, fsm, act_rams_row, act_laddr,
                                  act_gaddr, self.act_read_size,
                                  self.act_read_block, port=1)

            end_state_read = fsm.current

            fsm.If(vg.Ors(dma_pad_mask,
                          vg.Not(dma_flag))).goto_from(begin_state_read, end_state_read)

        bt.bus_unlock(self.maxi, fsm)

        fsm.goto_next()
        state_read_act_end = fsm.current
        fsm.If(skip_read_act).goto_from(state_read_act, state_read_act_end)

        # --------------------
        # Comp phase
        # --------------------
        state_comp = fsm.current

        # Stream Control FSM
        comp_fsm = vg.FSM(self.m, self._name('comp_fsm'), self.clk, self.rst)

        comp_state_init = comp_fsm.current
        comp_fsm.If(fsm.state == state_comp, vg.Not(skip_comp)).goto_next()

        fsm.If(comp_fsm.state == comp_state_init).goto_next()

        # waiting for previous DMA write completion
        bt.dma_wait_write_idle(self.maxi, comp_fsm)

        # local address
        stream_act_locals_2d = line_to_2d(stream_act_locals, ksize_col)
        for y, stream_act_locals_row in enumerate(stream_act_locals_2d):
            for x, stream_act_local in enumerate(stream_act_locals_row):
                comp_fsm(
                    stream_act_local(0)
                )
                comp_fsm.If(self.stream_act_local_small_flags[x])(
                    stream_act_local(self.stream_act_local_small_offset)
                )
                comp_fsm.If(self.stream_act_local_large_flags[x])(
                    stream_act_local(self.stream_act_local_large_offset)
                )

        comp_fsm(
            stream_out_local(0)
        )

        # count and sel
        comp_fsm(
            col_count(0)
        )
        comp_fsm(
            col_select(self.col_select_initval)
        )

        act_page_comp_offset_bufs = [self.m.Reg(self._name('act_page_comp_offset_buf_%d' % i),
                                                self.maxi.addrwidth, initval=0)
                                     for i in range(ksize_row)]
        out_page_comp_offset_buf = self.m.Reg(self._name('out_page_comp_offset_buf'),
                                              self.maxi.addrwidth, initval=0)
        row_count_buf = self.m.Reg(self._name('row_count_buf'),
                                   self.maxi.addrwidth, initval=0)
        row_select_buf = self.m.Reg(self._name('row_select_buf'),
                                    bt.log_width(ksize_row),
                                    initval=0)
        comp_fsm(
            [act_page_comp_offset_buf(act_page_comp_offset)
             for act_page_comp_offset_buf, act_page_comp_offset in zip(
                act_page_comp_offset_bufs, act_page_comp_offsets)],
            out_page_comp_offset_buf(out_page_comp_offset),
            row_count_buf(row_count),
            row_select_buf(row_select)
        )

        comp_fsm.goto_next()

        # repeat comp
        comp_state_rep = comp_fsm.current

        # pad_mask
        stream_pad_masks = []

        for y in range(ksize_row):
            for x in range(ksize_col):
                stream_col_count = col_count + x
                stream_row_count = row_count_buf + y
                v = vg.Ors((stream_col_count < self.pad_col_left),
                           (stream_col_count >= self.act_num_col + self.pad_col_left),
                           (stream_row_count < self.pad_row_top),
                           (stream_row_count >= self.act_num_row + self.pad_row_top))
                stream_pad_mask = self.m.Wire(
                    self._name('stream_pad_mask_%d_%d' % (y, x)))
                stream_pad_mask.assign(v)
                stream_pad_masks.append(stream_pad_mask)

        stream_pad_mask_2d = line_to_2d(stream_pad_masks, ksize_col)
        stream_pad_mask_2d_mux = mux_2d(stream_pad_mask_2d,
                                        col_select, row_select_buf,
                                        ksize_col, ksize_row)
        stream_pad_masks = [flatten for inner in stream_pad_mask_2d_mux
                            for flatten in inner]

        stream_pad_masks_reg = self.m.Reg(self._name('stream_pad_masks'),
                                          len(stream_pad_masks), initval=0)
        comp_fsm(
            stream_pad_masks_reg(vg.Cat(*reversed(stream_pad_masks)))
        )
        comp_fsm.goto_next()

        # busy check
        self.stream.source_join(comp_fsm)

        stream_masks = stream_pad_masks_reg

        # set_constant
        name = list(self.stream.constants.keys())[0]
        self.stream.set_constant(comp_fsm, name, stream_masks)
        comp_fsm.set_index(comp_fsm.current - 1)

        # set_source
        act_page_comp_offset_bufs_dup = []
        for act_page_comp_offset_buf in act_page_comp_offset_bufs:
            act_page_comp_offset_bufs_dup.extend(
                [act_page_comp_offset_buf] * ksize_col)

        for name, ram, stream_act_local, act_page_comp_offset_buf in zip(
                self.stream.sources.keys(), act_rams,
                stream_act_locals, act_page_comp_offset_bufs_dup):
            local = stream_act_local + act_page_comp_offset_buf
            self.stream.set_source(comp_fsm, name, ram,
                                   local, self.stream_size)
            comp_fsm.set_index(comp_fsm.current - 1)

        # set_sink
        name = list(self.stream.sinks.keys())[0]
        local = stream_out_local + out_page_comp_offset_buf
        self.stream.set_sink(comp_fsm, name, out_ram, local, self.stream_size)

        # stream run (async)
        self.stream.run(comp_fsm)

        # stream_act_local
        stream_act_locals_2d = line_to_2d(stream_act_locals, ksize_col)

        i = 0
        for y, stream_act_locals_row in enumerate(stream_act_locals_2d):
            for x, stream_act_local in enumerate(stream_act_locals_row):
                patterns = []
                for col in range(ksize_col):
                    val = self.inc_act_laddr_conds[i]
                    i += 1
                    pat = (col_select == col, val)
                    patterns.append(pat)

                patterns.append((None, 0))
                v = vg.PatternMux(*patterns)

                comp_fsm.If(vg.Not(v))(
                    stream_act_local.add(self.inc_act_laddr_small)
                )
                comp_fsm.If(v)(
                    stream_act_local.add(self.inc_act_laddr_large)
                )

                comp_fsm.If(col_count >= self.max_col_count)(
                    stream_act_local(0)
                )
                comp_fsm.If(col_count >= self.max_col_count,
                            self.stream_act_local_small_flags[x])(
                    stream_act_local(self.stream_act_local_small_offset)
                )
                comp_fsm.If(col_count >= self.max_col_count,
                            self.stream_act_local_large_flags[x])(
                    stream_act_local(self.stream_act_local_large_offset)
                )

        # stream_out_local
        comp_fsm(
            stream_out_local.add(self.inc_out_laddr)
        )
        comp_fsm.If(col_count >= self.max_col_count)(
            stream_out_local(0)
        )

        # counter
        comp_fsm(
            col_count.add(self.stride_col)
        )
        comp_fsm.If(col_count >= self.max_col_count)(
            col_count(0)
        )

        comp_fsm(
            col_select.add(self.stride_col_mod_ksize)
        )
        comp_fsm.If(col_select + self.stride_col_mod_ksize >= ksize_col)(
            col_select.sub(self.ksize_col_minus_stride_col_mod)
        )

        comp_fsm.If(col_count >= self.max_col_count)(
            col_select(self.col_select_initval),
        )

        # repeat
        comp_fsm.goto(comp_state_rep)
        comp_fsm.If(col_count >= self.max_col_count).goto_init()

        # sync with WriteOut control
        comp_fsm.seq.If(fsm.state == state_init)(
            comp_count(0)
        )
        comp_fsm.seq.If(self.stream.end_flag)(
            comp_count.add(self.inc_out_laddr)
        )

        # --------------------
        # WriteOut phase
        # --------------------
        state_write_out = fsm.current

        # sync with Comp control
        fsm.If(comp_count >= out_count + self.out_write_size).goto_next()

        out_laddr = out_page_dma_offset
        out_gaddr = self.objaddr + out_offset

        bt.bus_lock(self.maxi, fsm)

        bt.dma_write(self.maxi, fsm, out_ram, out_laddr,
                     out_gaddr, self.out_write_size, port=1, use_async=True)

        bt.bus_unlock(self.maxi, fsm)

        fsm(
            out_count.add(self.out_write_size)
        )

        fsm.goto_next()

        state_write_out_end = fsm.current
        fsm.If(skip_write_out).goto_from(state_write_out, state_write_out_end)

        # --------------------
        # update for next iteration
        # --------------------
        # ReadAct: offset
        fsm(
            act_base_offset_row.add(self.act_row_step)
        )
        fsm.If(row_count >= self.max_row_count)(
            act_base_offset_row(0),
            act_base_offset_bat.add(self.act_bat_step)
        )
        fsm.If(row_count >= self.max_row_count,
               bat_count >= self.max_bat_count)(
            act_base_offset_bat(0)
        )

        # ReadAct: DMA flag
        next_dma_flags = []
        for dma_flag, dma_flag_cond in zip(dma_flags, self.dma_flag_conds):
            fsm(
                dma_flag(dma_flag_cond)
            )

            fsm.If(row_count >= self.max_row_count)(
                dma_flag(1)
            )

            next_dma_flags.append(
                vg.Mux(row_count >= self.max_row_count, 1, dma_flag_cond))

        # ReadAct: counter
        fsm(
            row_count.add(self.stride_row)
        )
        fsm.If(row_count >= self.max_row_count)(
            row_count(0),
            bat_count.add(self.stride_bat)
        )
        fsm.If(row_count >= self.max_row_count,
               bat_count >= self.max_bat_count)(
            bat_count(0)
        )

        fsm.If(self.stride_row < ksize_row)(
            row_select.add(self.stride_row),
            prev_row_select(row_select)
        )
        fsm.If(self.stride_row < ksize_row,
               row_select + self.stride_row >= ksize_row)(
            row_select(row_select - (vg.Int(ksize_row) - self.stride_row)),
            prev_row_select(row_select)
        )
        fsm.If(vg.Not(self.stride_row < ksize_row))(
            row_select(0),
            prev_row_select(0)
        )

        fsm.If(row_count >= self.max_row_count)(
            row_select(0),
            prev_row_select(0)
        )

        # ReadAct and Comp: double buffer
        mux_next_dma_flag_values = mux_1d(
            next_dma_flags, row_select, ksize_row)
        mux_next_dma_flags = []
        for i, mux_next_dma_flag_value in enumerate(mux_next_dma_flag_values):
            mux_next_dma_flag = self.m.Wire(
                self._name('mux_next_dma_flag_%d' % i))
            mux_next_dma_flag.assign(mux_next_dma_flag_value)
            mux_next_dma_flags.append(mux_next_dma_flag)

        for (act_page, act_page_comp_offset,
             act_page_dma_offset, mux_next_dma_flag) in zip(
                 act_pages, act_page_comp_offsets,
                 act_page_dma_offsets, mux_next_dma_flags):

            fsm.If(vg.Not(act_page), mux_next_dma_flag)(
                act_page_comp_offset(act_page_size),
                act_page_dma_offset(act_page_size),
                act_page(1)
            )
            fsm.If(act_page, mux_next_dma_flag)(
                act_page_comp_offset(0),
                act_page_dma_offset(0),
                act_page(0)
            )

        # WriteOut: counter
        fsm.If(vg.Not(skip_write_out))(
            out_base_offset_row.add(self.out_row_step)
        )
        fsm.If(vg.Not(skip_write_out),
               prev_row_count >= self.max_row_count)(
            out_base_offset_row(0),
            out_base_offset_bat.add(self.out_bat_step)
        )
        fsm.If(vg.Not(skip_write_out),
               prev_row_count >= self.max_row_count,
               prev_bat_count >= self.max_bat_count)(
            out_base_offset_bat(0)
        )

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
        fsm(
            prev_row_count(row_count),
            prev_bat_count(bat_count)
        )

        # ReadAct, Comp, WriteOut: skip
        fsm.If(row_count >= self.max_row_count,
               bat_count >= self.max_bat_count)(
            skip_read_act(1)
        )

        fsm.If(row_count >= self.max_row_count,
               bat_count >= self.max_bat_count)(
            skip_comp(1)
        )

        fsm.If(skip_write_out,
               prev_row_count == 0,
               prev_bat_count == 0)(
            skip_write_out(0)
        )

        fsm.goto(state_read_act)
        fsm.If(vg.Not(skip_write_out),
               prev_row_count >= self.max_row_count,
               prev_bat_count >= self.max_bat_count).goto_next()

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
        ksize = self.ksize
        strides = self.strides

        kwargs['padding'] = self.padding
        kwargs['dtype'] = self.dtype
        kwargs['name'] = self.name
        kwargs['par'] = self.par
        kwargs['value_dtype'] = self.args[0].dtype

        method = self.get_eval_method()
        ret = method(value, ksize, strides, **kwargs)
        memo[id(self)] = ret

        return ret


class avg_pool(_pool):

    def __sub_str__(self):
        sum_dtype = (' sum_dtype:%s' % self.sum_dtype.to_str()
                     if self.sum_dtype is not None else '')
        force_div = ' force_div' if self.force_div else ''

        return ''.join([_pool.__sub_str__(self), sum_dtype, force_div])

    def __init__(self, value, ksize, strides, padding='SAME',
                 dtype=None, sum_dtype=None, name=None, par=1,
                 force_div=False,
                 value_ram_size=None, out_ram_size=None):

        self.sum_dtype = sum_dtype
        self.force_div = force_div
        _pool.__init__(self, value, ksize, strides,
                       padding, dtype, name, par,
                       value_ram_size, out_ram_size)

    def get_pad_value(self, strm):
        return strm.Int(0)

    def get_required_substreams(self):
        ksize_col = self.ksize[-2]
        ksize_row = self.ksize[-3]
        num_vars = ksize_col * ksize_row

        act = self.args[0]
        if self.sum_dtype is not None:
            x_datawidth = self.sum_dtype.width
        else:
            x_datawidth = max(self.get_op_width(), act.get_op_width())
        x_point = act.get_op_point()
        x_signed = act.get_signed()

        substrms = [('add_tree_rshift_round',
                     (x_datawidth, x_point, x_signed, num_vars))] * self.par

        if self.force_div or num_vars & (num_vars - 1) != 0:
            y_datawidth = max(num_vars.bit_length() + 1, 2)
            y_point = 0
            y_signed = True
            substrms.extend([('div_const_frac',
                              (x_datawidth, x_point, x_signed,
                               y_datawidth, y_point, y_signed))] * self.par)

        return substrms

    def get_stream_hash(self):
        base = _pool.get_stream_hash(self)
        return (base, self.sum_dtype, self.force_div)

    def pool_op(self, strm, index, *vars):
        num_vars = len(vars)

        addtree = strm.substream(self.substreams[index])

        for i, var in enumerate(vars):
            addtree.to_source('var%d' % i, var)

        sum = addtree.from_sink('sum')

        if (self.sum_dtype is not None and
            (sum.point != self.sum_dtype.point or
             sum.signed != self.sum_dtype.signed)):
            sum = strm.Cast(sum, self.sum_dtype.width, self.sum_dtype.point,
                            self.sum_dtype.signed)

        if self.force_div or num_vars & (num_vars - 1) != 0:
            addtree.to_constant('rshift', 0)
            div = strm.substream(self.substreams[self.par + index])

            frac = num_vars//2
            div.to_constant('frac', frac)

            div.to_source('x', sum)
            div.to_constant('y', num_vars)
            return div.from_sink('z')

        rshift = int(math.log(num_vars, 2))
        addtree.to_constant('rshift', rshift)

        return sum


class max_pool(_pool):

    def get_pad_value(self, strm):
        arg = self.args[0]
        datawidth = arg.get_op_width()
        value = (-1) * (1 << (datawidth - 1))
        return strm.Int(value)

    def get_required_substreams(self):
        ksize_col = self.ksize[-2]
        ksize_row = self.ksize[-3]
        num_vars = ksize_col * ksize_row

        act = self.args[0]
        width = act.get_op_width()
        point = act.get_op_point()
        signed = act.get_signed()

        substrms = [('_max',
                     (width, point, signed, num_vars))] * self.par

        return substrms

    def pool_op(self, strm, index, *vars):
        maxtree = strm.substream(self.substreams[index])

        for i, var in enumerate(vars):
            maxtree.to_source('var%d' % i, var)

        return maxtree.from_sink('val')


def line_to_2d(lst, kx):
    row = []
    col = []

    for l in lst:
        col.append(l)
        if len(col) == kx:
            row.append(col)
            col = []

    if len(col) != 0:
        raise ValueError('not all objects are not utilized.')

    return row


def transpose_2d(mat):
    return list(zip(*mat))


def mux_2d(mat, col_select, row_select, col_size, row_size, width=1):
    ret_list = []
    for line in mat:
        for j in range(col_size):
            ret = vg.Int(0, width=width)
            for i in reversed(range(len(line))):
                ret = vg.Mux(col_select == i,
                             line[(j + col_size - i) % col_size], ret)
            ret_list.append(ret)

    mat = transpose_2d(line_to_2d(ret_list, col_size))

    ret_list = []
    for line in mat:
        for j in range(row_size):
            ret = vg.Int(0, width=width)
            for i in reversed(range(len(line))):
                ret = vg.Mux(row_select == i,
                             line[(j + row_size - i) % row_size], ret)
            ret_list.append(ret)

    return transpose_2d(line_to_2d(ret_list, row_size))


def mux_1d(line, select, size, width=1):
    ret_list = []
    for j in range(size):
        ret = vg.Int(0, width=width)
        for i in reversed(range(len(line))):
            ret = vg.Mux(select == i,
                         line[(j + size - i) % size], ret)
        ret_list.append(ret)

    return ret_list
