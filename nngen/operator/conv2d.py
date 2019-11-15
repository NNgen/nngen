from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
from collections import OrderedDict

import veriloggen as vg

import nngen.basic_types as bt
import nngen.util as util


STATIONARY_FILETER = 0
STATIONARY_INPUT = 1


class conv2d(bt._Operator):
    """
    Computes a 2-D convolution given 4-D input and filter tensors.

    Parameters
    ----------
    input :
        ``NHWC`` (batch, height, width, channel)

    filter :
       ``OHWI`` (outchannel, height, width, inchannel)

    stride :
        ``NHWC`` (N and C are always 1)

    bias : optional
        Tensor for bias addition to outputs.

    scale : optional
        Tensor for scaling to outputs.

    rshift_mul : optional
        Arithmetic shift right can be performed on the result of \
        activation and kernel multiplication as needed. \
        Designation of shift amount can select constant and vector. \
        The constants are register generated and \
        the vector is configured in RAM.

    rshift_sum : optional
        Arithmetic shift right can be performed on the result of \
        accumulation as needed. \
        Designation of shift amount can select constant and vector. \
        The constants are register generated and \
        the vector is configured in RAM.

    rshift_out : optional
        Arithmetic shift right can be performed on the output value \
        per channel as needed. \
        Designation of shift amount can select constant and vector. \
        The constants are register generated and \
        the vector is configured in RAM.

    act_func : optional
        The output value of conv2d can be input to the activation function \
        before writing to memory. \
        The activation function that can be specified is the operator \
        that inherited the element-wise class.

    padding : optional
        A string (Only 'SAME' padding is supported). \
        The type of padding algorithm to use. \
        The detailed algorithm conforms to Tensorflow.

    dtype : optional
        Output data type.

    mul_dtype : optional
        Data type of register that stores the result of multiplication \
        of activation and kernel parameter.

    sum_dtype : optional
        Data type of register that stores accumulation result.

    name : optional
        A name for the operation (optional).

    par_ich : optional
        Specifies the degree of operation parallelism \
        in the input channel direction.

    par_och : optional
        Specifies the degree of operation parallelism \
        in the output channel direction.

    par_col : optional
        Specifies the degree of operation parallelism \
        in the column direction.

    par_row : optional
        Specifies the degree of operation parallelism \
        in the row direction.

    concur_och : optional
        Specifies how many channels of kernel parameters to read simultaneously. \
        If this value is large enough, the efficiency of burst transfer \
        in writing the output result is maximized. \
        If not specified, it is calculated automatically from the relationship \
        between the number of words of memory for kernel parameters \
        and the number of words of memory for output.

    stationary : optional
        Designate data flow to MAC. \
        In 'weight' mode, the weight data supplied to the MAC is fixed, \
        while in 'act' mode, the input data is fixed.

    input_ram_size : optional
        Specify the word length of the input data RAM. \
        If set to less than the minimum required word length \
        (depends on the input data size), the set value is ignored.

    filter_ram_size : optional
        Specifies the word length of the kernel parameter RAM. \
        If set to less than the minimum required word length \
        (depends on the kernel parameter size), the set value is ignored.

    bias_ram_size : optional
        Specify the word length of the bias parameter RAM. \
        If set to less than the minimum required word length \
        (depends on bias parameter size), the set value is ignored.

    scale_ram_size : optional
        Specify the word length of scaling parameter RAM. \
        If set to less than the minimum required word length \
        (depends on the scaling parameter size), the set value is ignored.

    vshamt_mul_ram_size : optional
        Specifies the word length of the RAM that stores the arithmetic right shift value \
        for the result of activation and multiplication of kernel parameters. \
        The setting value is ignored if it is set to the minimum required word length or less.

    vshamt_sum_ram_size : optional
        Specifies the word length of the RAM that stores the arithmetic right shift value \
        for the accumulation result. \
        The setting value is ignored if it is set to the minimum required word length or less.

    vshamt_out_ram_size : optional
        Specifies the word length of the RAM that stores the arithmetic right shift value \
        for the output result. \
        The setting value is ignored if it is set to the minimum required word length or less.

    out_ram_size : optional
        Specifies the word length of the output data RAM. \
        If set to less than the minimum required word length \
        (depends on the output data size), the set value is ignored.

    disable_keep_input : optional
        If this parameter is set to True, data will be reread from the main memory \
        even if the required amount of input data is full on the on-chip RAM.

    Notes
    --------
    Note that the original order of tensorflow's conv2d is ``HWIO``
    (height, width, inchannel, outchannel)

    """

    input_chainable = False
    output_chainable = False

    control_param_custom_width = {'act_offset_values': bt.get_maxi_addrwidth}
    control_param_custom_signed = {'act_offset_values': True}

    shared_attr_names = ('act_func',)

    def __sub_str__(self):
        strides = str(self.strides)

        bias = (' bias:%s' % str(self.args[self.args_dict['bias']].shape)
                if 'bias' in self.args_dict else '')
        scale = (' scale:%s' % str(self.args[self.args_dict['scale']].shape)
                 if 'scale' in self.args_dict else '')

        vshamt_mul = (' vshamt_mul:%s' % str(self.args[self.args_dict['vshamt_mul']].shape)
                      if 'vshamt_mul' in self.args_dict else '')
        vshamt_sum = (' vshamt_sum:%s' % str(self.args[self.args_dict['vshamt_sum']].shape)
                      if 'vshamt_sum' in self.args_dict else '')
        vshamt_out = (' vshamt_out:%s' % str(self.args[self.args_dict['vshamt_out']].shape)
                      if 'vshamt_out' in self.args_dict else '')

        cshamt_mul = (' cshamt_mul:%s' % self.cshamt_mul
                      if self.cshamt_mul is not None else '')
        cshamt_sum = (' cshamt_sum:%s' % self.cshamt_sum
                      if self.cshamt_sum is not None else '')
        cshamt_out = (' cshamt_out:%s' % self.cshamt_out
                      if self.cshamt_out is not None else '')

        act_func = (' act_func:%s' % str(self.act_func.__name__)
                    if self.act_func is not None else '')
        mul_dtype = (' mul_dtype:%s' % self.mul_dtype.to_str()
                     if self.mul_dtype is not None else '')
        sum_dtype = (' sum_dtype:%s' % self.sum_dtype.to_str()
                     if self.sum_dtype is not None else '')

        if hasattr(self, 'pad_col_left_value') and self.padding == 'SAME':
            padding = ("'%s'-(%d, %d, %d, %d)" %
                       (self.padding,
                        self.pad_col_left_value, self.pad_col_right_value,
                        self.pad_row_top_value, self.pad_row_bottom_value))
        else:
            padding = self.padding

        par_ich = ' par_ich:%d' % self.par_ich if self.par_ich > 1 else ''
        par_och = ' par_och:%d' % self.par_och if self.par_och > 1 else ''
        par_col = ' par_col:%d' % self.par_col if self.par_col > 1 else ''
        par_row = ' par_row:%d' % self.par_row if self.par_row > 1 else ''
        if hasattr(self, 'concur_och_value'):
            concur_och = ' concur_och:%d' % self.concur_och_value
        else:
            concur_och = ' concur_och:%s' % str(self.concur_och)
        stationary = ' stationary:%s' % self.stationary

        input_ram_size = (' input_ram_size:%d' % self.input_ram_size
                          if self.input_ram_size is not None else '')
        filter_ram_size = (' filter_ram_size:%d' % self.filter_ram_size
                           if self.filter_ram_size is not None else '')
        bias_ram_size = (' bias_ram_size:%d' % self.bias_ram_size
                         if self.bias_ram_size is not None else '')
        scale_ram_size = (' scale_ram_size:%d' % self.scale_ram_size
                          if self.scale_ram_size is not None else '')
        vshamt_mul_ram_size = (' vshamt_mul_ram_size:%d' % self.vshamt_mul_ram_size
                               if self.vshamt_mul_ram_size is not None else '')
        vshamt_sum_ram_size = (' vshamt_sum_ram_size:%d' % self.vshamt_sum_ram_size
                               if self.vshamt_sum_ram_size is not None else '')
        vshamt_out_ram_size = (' vshamt_out_ram_size:%d' % self.vshamt_out_ram_size
                               if self.vshamt_out_ram_size is not None else '')
        out_ram_size = (' out_ram_size:%d' % self.out_ram_size
                        if self.out_ram_size is not None else '')

        if hasattr(self, 'keep_input_value'):
            keep_input = ' keep_input' if self.keep_input_value else ''
        else:
            keep_input = ''
        if hasattr(self, 'keep_filter_value'):
            keep_filter = ' keep_filter' if self.keep_filter_value else ''
        else:
            keep_filter = ''

        ret = [' strides:%s padding:%s' % (strides, padding)]
        ret.extend([str(s)
                    for s in (bias, scale,
                              vshamt_mul, vshamt_sum, vshamt_out,
                              cshamt_mul, cshamt_sum, cshamt_out,
                              act_func, mul_dtype, sum_dtype,
                              par_ich, par_och, par_col, par_row,
                              concur_och, stationary,
                              input_ram_size, filter_ram_size,
                              bias_ram_size, scale_ram_size,
                              vshamt_mul_ram_size, vshamt_sum_ram_size, vshamt_out_ram_size,
                              out_ram_size,
                              keep_input, keep_filter)])
        return ''.join(ret)

    def __init__(self, input, filter, strides,
                 bias=None, scale=None,
                 rshift_mul=None, rshift_sum=None, rshift_out=None,
                 act_func=None, padding='SAME',
                 dtype=None, mul_dtype=None, sum_dtype=None,
                 name=None,

                 # attribute
                 par_ich=1, par_och=1, par_col=1, par_row=1,
                 concur_och=None, stationary='filter',
                 input_ram_size=None, filter_ram_size=None,
                 bias_ram_size=None, scale_ram_size=None,
                 vshamt_mul_ram_size=None, vshamt_sum_ram_size=None, vshamt_out_ram_size=None,
                 out_ram_size=None,
                 disable_keep_input=False,

                 # for matmul
                 input_shape=None, filter_shape=None, out_shape=None):

        if isinstance(padding, str) and padding != 'SAME' and padding != 'VALID':
            raise ValueError("padding options must be 'SAME', 'VALID', int, tuple, or list.")
        elif isinstance(padding, (tuple, list)) and len(padding) != 4:
            raise ValueError('padding rank must be 4.')

        if input_shape is not None:
            if input_shape[-1] != input.shape[-1]:
                raise ValueError("""external input_shape[-1] must have"""
                                 """a same size as original shape[-1]:"""
                                 """ '%d' != '%d'""" % (input_shape[-1], input.shape[-1]))
            if bt.shape_to_length(input_shape[:-1]) != bt.shape_to_length(input.shape[:-1]):
                raise ValueError("""external input_shape must have"""
                                 """a same total length as original shape:"""
                                 """ '%d' != '%d'""" % (bt.shape_to_length(input_shape[:-1]),
                                                        bt.shape_to_length(input.shape[:-1])))
        else:
            input_shape = input.shape

        if filter_shape is not None:
            if filter_shape[-1] != filter.shape[-1]:
                raise ValueError("""external filter_shape[-1] must have"""
                                 """a same size as original shape[-1]:"""
                                 """ '%d' != '%d'""" % (filter_shape[-1], filter.shape[-1]))
            if bt.shape_to_length(filter_shape[:-1]) != bt.shape_to_length(filter.shape[:-1]):
                raise ValueError("""external filter_shape must have"""
                                 """a same total length as original shape:"""
                                 """ '%d' != '%d'""" % (bt.shape_to_length(filter_shape[:-1]),
                                                        bt.shape_to_length(filter.shape[:-1])))
        else:
            filter_shape = filter.shape

        if bt.get_rank(input_shape) != 4 or bt.get_rank(filter_shape) != 4:
            raise ValueError('input and filter ranks must be 4.')

        if len(strides) != 4:
            raise ValueError('rank of strides must be 4.')

        if strides[0] != 1 or strides[3] != 1:
            raise ValueError('strides[0] and [3] must be 1')

        if input_shape[-1] != filter_shape[-1]:
            raise ValueError("""input and filter must have """
                             """a same input channel length as shape[3]: """
                             """'%d' != '%d'""" % (input_shape[-1], filter_shape[-1]))

        if isinstance(padding, str) and (padding == 'SAME' or padding == 'VALID'):
            shape = []
            shape.append(int(math.ceil(input_shape[0] / strides[0])))
            for sh, st, fs in list(zip(input_shape, strides, filter_shape))[1:-1]:
                shape.append(util.pix_size(sh, fs, st, padding))
            shape.append(filter_shape[0])
        elif isinstance(padding, int):
            shape = []
            shape.append(int(math.ceil(input_shape[0] / strides[0])))
            for sh, st, fs in list(zip(input_shape, strides, filter_shape))[1:-1]:
                shape.append(util.pix_size(sh + padding * 2, fs, st, 'VALID'))
            shape.append(filter_shape[0])
        elif isinstance(padding, (tuple, list)):
            shape = []
            shape.append(int(math.ceil(input_shape[0] / strides[0])))
            for i, (sh, st, fs) in enumerate(
                    list(zip(input_shape, strides, filter_shape))[1:-1]):
                pd0 = padding[i * 2]
                pd1 = padding[i * 2 + 1]
                shape.append(util.pix_size(sh + pd0 + pd1, fs, st, 'VALID'))
            shape.append(filter_shape[0])

        orig_shape = tuple(shape)

        if out_shape is not None:
            if out_shape[-1] != orig_shape[-1]:
                raise ValueError("""external out_shape[-1] must have"""
                                 """a same size as original shape[-1]:"""
                                 """ '%d' != '%d'""" % (out_shape[-1], orig_shape[-1]))
            if bt.shape_to_length(out_shape[:-1]) != bt.shape_to_length(orig_shape[:-1]):
                raise ValueError("""external out_shape must have"""
                                 """a same total length as original shape:"""
                                 """ '%d' != '%d'""" % (bt.shape_to_length(out_shape[:-1]),
                                                        bt.shape_to_length(orig_shape[:-1])))
            shape = out_shape
        else:
            shape = orig_shape

        if bias is not None:
            if (bt.get_rank(bias.shape) > 1 or
                    (bias.shape[-1] != 1 and bias.shape[-1] != shape[-1])):
                raise ValueError("shape of bias must be (1,) or (num_och,)")

        if scale is not None:
            if (bt.get_rank(scale.shape) > 1 or
                    (scale.shape[-1] != 1 and scale.shape[-1] != shape[-1])):
                raise ValueError("shape of scale must be (1,) or (num_och,)")

        if rshift_mul is None:
            vshamt_mul = None
            cshamt_mul = None
        elif isinstance(rshift_mul, int):
            vshamt_mul = None
            cshamt_mul = rshift_mul
        else:
            vshamt_mul = rshift_mul
            cshamt_mul = None

        if rshift_sum is None:
            vshamt_sum = None
            cshamt_sum = None
        elif isinstance(rshift_sum, int):
            vshamt_sum = None
            cshamt_sum = rshift_sum
        else:
            vshamt_sum = rshift_sum
            cshamt_sum = None

        if rshift_out is None:
            vshamt_out = None
            cshamt_out = None
        elif isinstance(rshift_out, int):
            vshamt_out = None
            cshamt_out = rshift_out
        else:
            vshamt_out = rshift_out
            cshamt_out = None

        if vshamt_mul is not None:
            if (bt.get_rank(vshamt_mul.shape) > 1 or
                    (vshamt_mul.shape[-1] != 1 and vshamt_mul.shape[-1] != shape[-1])):
                raise ValueError("shape of vshamt_mul must be (1,) or (num_och,)")

        if vshamt_sum is not None:
            if (bt.get_rank(vshamt_sum.shape) > 1 or
                    (vshamt_sum.shape[-1] != 1 and vshamt_sum.shape[-1] != shape[-1])):
                raise ValueError("shape of vshamt_sum must be (1,) or (num_och,)")

        if vshamt_out is not None:
            if (bt.get_rank(vshamt_out.shape) > 1 or
                    (vshamt_out.shape[-1] != 1 and vshamt_out.shape[-1] != shape[-1])):
                raise ValueError("shape of vshamt_out must be (1,) or (num_och,)")

        if cshamt_mul is not None:
            if not isinstance(cshamt_mul, int):
                raise ValueError("cshamt_mul must be int, not '%s'" %
                                 str(type(cshamt_mul)))

        if cshamt_sum is not None:
            if not isinstance(cshamt_sum, int):
                raise ValueError("cshamt_sum must be int, not '%s'" %
                                 str(type(cshamt_sum)))

        if cshamt_out is not None:
            if not isinstance(cshamt_out, int):
                raise ValueError("cshamt_out must be int, not '%s'" %
                                 str(type(cshamt_out)))

        if (act_func is not None and
                not issubclass(act_func, bt._ElementwiseOperator)):
            raise TypeError('act_func must be _ElementwiseOperator class.')

        args = [input, filter]
        if bias is not None:
            args.append(bias)
            self.has_bias = True
        else:
            self.has_bias = False

        if scale is not None:
            args.append(scale)
            self.has_scale = True
        else:
            self.has_scale = False

        if vshamt_mul is not None:
            args.append(vshamt_mul)
            self.has_vshamt_mul = True
        else:
            self.has_vshamt_mul = False

        if vshamt_sum is not None:
            args.append(vshamt_sum)
            self.has_vshamt_sum = True
        else:
            self.has_vshamt_sum = False

        if vshamt_out is not None:
            args.append(vshamt_out)
            self.has_vshamt_out = True
        else:
            self.has_vshamt_out = False

        self.cshamt_mul = cshamt_mul
        self.cshamt_sum = cshamt_sum
        self.cshamt_out = cshamt_out

        args_count = 2
        self.args_dict = OrderedDict()

        if self.has_bias:
            self.args_dict['bias'] = args_count
            args_count += 1

        if self.has_scale:
            self.args_dict['scale'] = args_count
            args_count += 1

        if self.has_vshamt_mul:
            self.args_dict['vshamt_mul'] = args_count
            args_count += 1

        if self.has_vshamt_sum:
            self.args_dict['vshamt_sum'] = args_count
            args_count += 1

        if self.has_vshamt_out:
            self.args_dict['vshamt_out'] = args_count
            args_count += 1

        bt._Operator.__init__(self, *args,
                              dtype=dtype, shape=shape, name=name)

        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.orig_shape = orig_shape

        self.strides = tuple(strides)
        self.padding = padding
        self.mul_dtype = mul_dtype
        self.sum_dtype = sum_dtype

        self.act_func = act_func

        # attribute
        self.par_ich = par_ich
        self.par_och = par_och
        self.par_col = par_col
        self.par_row = par_row
        self.concur_och = concur_och
        self.stationary = stationary
        self.input_ram_size = input_ram_size
        self.filter_ram_size = filter_ram_size
        self.bias_ram_size = bias_ram_size
        self.scale_ram_size = scale_ram_size
        self.vshamt_mul_ram_size = vshamt_mul_ram_size
        self.vshamt_sum_ram_size = vshamt_sum_ram_size
        self.vshamt_out_ram_size = vshamt_out_ram_size
        self.out_ram_size = out_ram_size
        self.disable_keep_input = disable_keep_input
        conv2d.attribute(self, None, None, None,
                         par_ich, par_och, par_col, par_row,
                         concur_och,
                         stationary,
                         input_ram_size, filter_ram_size,
                         bias_ram_size, scale_ram_size,
                         vshamt_mul_ram_size, vshamt_sum_ram_size, vshamt_out_ram_size,
                         out_ram_size,
                         disable_keep_input)

    def attribute(self, cshamt_mul=None, cshamt_sum=None, cshamt_out=None,
                  par_ich=None, par_och=None, par_col=None, par_row=None,
                  concur_och=None,
                  stationary=None,
                  input_ram_size=None, filter_ram_size=None,
                  bias_ram_size=None, scale_ram_size=None,
                  vshamt_mul_ram_size=None, vshamt_sum_ram_size=None, vshamt_out_ram_size=None,
                  out_ram_size=None,
                  disable_keep_input=None):

        if cshamt_mul is not None:
            self.cshamt_mul = cshamt_mul

        if cshamt_sum is not None:
            self.cshamt_sum = cshamt_sum

        if cshamt_out is not None:
            self.cshamt_out = cshamt_out

        if par_ich is not None:
            if par_ich < 1:
                raise ValueError('par_ich must be greater than 0')

            if (par_ich - 1) & par_ich != 0:
                raise ValueError('par_ich must be power of 2')

            self.par_ich = par_ich

            for arg in self.args[:2]:
                arg.add_alignment_request(self.par_ich)

        if par_och is not None:
            if par_och < 1:
                raise ValueError('par_och must be greater than 0')

            if (par_och - 1) & par_och != 0:
                raise ValueError('par_och must be power of 2')

            self.par_och = par_och

            for arg in self.args[2:]:
                arg.add_alignment_request(self.par_och)

            self.add_alignment_request(self.par_och)

        if par_col is not None:
            if par_col < 1:
                raise ValueError('par_col must be greater than 0')

            self.par_col = par_col

        if par_row is not None:
            if par_row < 1:
                raise ValueError('par_row must be greater than 0')

            self.par_row = par_row

        if concur_och is not None:
            if concur_och is not None and concur_och < 1:
                raise ValueError('concur_och must be greater than 0')

            self.concur_och = concur_och

        if self.concur_och is not None and self.concur_och % self.par_och != 0:
            raise ValueError('concur_och must be a multiple of par_och')

        if stationary is not None:
            if stationary == 'weight':
                stationary = 'filter'

            if stationary == 'act':
                stationary = 'input'

            if stationary != 'filter' and stationary != 'input':
                raise ValueError("stationary must be 'filter' or 'input'.")

            self.stationary = stationary

        if input_ram_size is not None:
            if input_ram_size is not None and input_ram_size < 1:
                raise ValueError('input_ram_size must be greater than 0')

            self.input_ram_size = input_ram_size

        if filter_ram_size is not None:
            if filter_ram_size is not None and filter_ram_size < 1:
                raise ValueError('filter_ram_size must be greater than 0')

            self.filter_ram_size = filter_ram_size

        if bias_ram_size is not None:
            if bias_ram_size is not None and bias_ram_size < 1:
                raise ValueError('bias_ram_size must be greater than 0')

            self.bias_ram_size = bias_ram_size

        if scale_ram_size is not None:
            if scale_ram_size is not None and scale_ram_size < 1:
                raise ValueError('scale_ram_size must be greater than 0')

            self.scale_ram_size = scale_ram_size

        if vshamt_mul_ram_size is not None:
            if vshamt_mul_ram_size is not None and vshamt_mul_ram_size < 1:
                raise ValueError('vshamt_mul_ram_size must be greater than 0')

            self.vshamt_mul_ram_size = vshamt_mul_ram_size

        if vshamt_sum_ram_size is not None:
            if vshamt_sum_ram_size is not None and vshamt_sum_ram_size < 1:
                raise ValueError('vshamt_sum_ram_size must be greater than 0')

            self.vshamt_sum_ram_size = vshamt_sum_ram_size

        if vshamt_out_ram_size is not None:
            if vshamt_out_ram_size is not None and vshamt_out_ram_size < 1:
                raise ValueError('vshamt_out_ram_size must be greater than 0')

            self.vshamt_out_ram_size = vshamt_out_ram_size

        if out_ram_size is not None:
            if out_ram_size is not None and out_ram_size < 1:
                raise ValueError('out_ram_size must be greater than 0')

            self.out_ram_size = out_ram_size

        if disable_keep_input is not None:
            self.disable_keep_input = disable_keep_input

    def get_required_rams(self):
        arg_input = self.args[0]
        arg_filter = self.args[1]
        arg_bias = (self.args[self.args_dict['bias']]
                    if 'bias' in self.args_dict else None)
        arg_scale = (self.args[self.args_dict['scale']]
                     if 'scale' in self.args_dict else None)
        arg_vshamt_mul = (self.args[self.args_dict['vshamt_mul']]
                          if 'vshamt_mul' in self.args_dict else None)
        arg_vshamt_sum = (self.args[self.args_dict['vshamt_sum']]
                          if 'vshamt_sum' in self.args_dict else None)
        arg_vshamt_out = (self.args[self.args_dict['vshamt_out']]
                          if 'vshamt_out' in self.args_dict else None)

        act = arg_input
        act_shape = to_aligned_shape(act, self.input_shape)
        act_num_ch = act_shape[-1]
        act_num_col = act_shape[-2]
        act_num_row = act_shape[-3]
        act_num_bat = act_shape[-4]

        filter = arg_filter
        filter_shape = to_aligned_shape(filter, self.filter_shape)
        filter_num_col = filter_shape[-2]
        filter_num_row = filter_shape[-3]
        filter_num_och = filter_shape[-4]

        out_shape = to_aligned_shape(self, self.orig_shape)
        out_num_ch = out_shape[-1]
        out_num_col = out_shape[-2]
        out_num_row = out_shape[-3]
        out_num_bat = out_shape[-4]

        stride_col = self.strides[-2]  # width
        stride_row = self.strides[-3]  # height
        src_num_col = filter_num_col + stride_col * (self.par_col - 1)
        src_num_row = filter_num_row + stride_row * (self.par_row - 1)

        req_concur_och = self.get_req_concur_och()

        # in set_thread_args, keep_input is determined precisely
        keep_input = (not self.disable_keep_input and
                      self.stationary == 'filter' and
                      act_num_row <= filter_num_row and
                      act_num_bat == 1)
        input_double = 2 if not keep_input else 1

        keep_filter = out_num_ch <= req_concur_och
        filter_double = 2 if not keep_filter else 1

        aligned_act_num_ch = bt.align_word(act_num_ch,
                                           act.get_word_alignment())
        input_min_size = (int(math.ceil(aligned_act_num_ch / self.par_ich)) *
                          int(math.ceil(act_num_col / src_num_col)) * input_double)
        if self.input_ram_size is not None and input_min_size < self.input_ram_size:
            input_min_size = self.input_ram_size
        input_width = act.get_ram_width() * self.par_ich

        aligned_filter_num_ich = bt.align_word(act_num_ch,
                                               filter.get_word_alignment())

        num_filter_blocks = (req_concur_och // self.par_och
                             if self.par_och is not None else req_concur_och)

        filter_min_size = (int(math.ceil(aligned_filter_num_ich / self.par_ich)) *
                           num_filter_blocks * filter_double)
        if self.filter_ram_size is not None and filter_min_size < self.filter_ram_size:
            filter_min_size = self.filter_ram_size
        filter_width = filter.get_ram_width() * self.par_ich

        if arg_bias is not None:
            bias_min_size = arg_bias.get_aligned_shape()[-1]
            if self.bias_ram_size is not None and bias_min_size < self.bias_ram_size:
                bias_min_size = self.bias_ram_size
            bias_width = arg_bias.get_ram_width() * self.par_och
        else:
            bias_min_size = None
            bias_width = None

        if arg_scale is not None:
            scale_min_size = arg_scale.get_aligned_shape()[-1]
            if self.scale_ram_size is not None and scale_min_size < self.scale_ram_size:
                scale_min_size = self.scale_ram_size
            scale_width = arg_scale.get_ram_width() * self.par_och
        else:
            scale_min_size = None
            scale_width = None

        if arg_vshamt_mul is not None:
            vshamt_mul_min_size = arg_vshamt_mul.get_aligned_shape()[-1]
            if self.vshamt_mul_ram_size is not None and vshamt_mul_min_size < self.vshamt_mul_ram_size:
                vshamt_mul_min_size = self.vshamt_mul_ram_size
            vshamt_mul_width = arg_vshamt_mul.get_ram_width() * self.par_och
        else:
            vshamt_mul_min_size = None
            vshamt_mul_width = None

        if arg_vshamt_sum is not None:
            vshamt_sum_min_size = arg_vshamt_sum.get_aligned_shape()[-1]
            if self.vshamt_sum_ram_size is not None and vshamt_sum_min_size < self.vshamt_sum_ram_size:
                vshamt_sum_min_size = self.vshamt_sum_ram_size
            vshamt_sum_width = arg_vshamt_sum.get_ram_width() * self.par_och
        else:
            vshamt_sum_min_size = None
            vshamt_sum_width = None

        if arg_vshamt_out is not None:
            vshamt_out_min_size = arg_vshamt_out.get_aligned_shape()[-1]
            if self.vshamt_out_ram_size is not None and vshamt_out_min_size < self.vshamt_out_ram_size:
                vshamt_out_min_size = self.vshamt_out_ram_size
            vshamt_out_width = arg_vshamt_out.get_ram_width() * self.par_och
        else:
            vshamt_out_min_size = None
            vshamt_out_width = None

        if self.stationary == 'filter':
            output_min_size = num_filter_blocks * out_num_col * 2
        else:
            och_max = max(out_num_ch, req_concur_och)
            output_min_size = (int(math.ceil(och_max / self.par_och)) *
                               out_num_col * 2)
        if self.out_ram_size is not None and output_min_size < self.out_ram_size:
            output_min_size = self.out_ram_size
        output_width = self.get_ram_width() * self.par_och

        stride_col = self.strides[-2]  # width
        stride_row = self.strides[-3]  # height

        inputs = []
        # act
        inputs.extend([(input_width, input_min_size)] *
                      src_num_col * src_num_row)

        # weight
        inputs.extend([(filter_width, filter_min_size)] *
                      filter_num_col * filter_num_row * self.par_och)

        # bias
        if bias_min_size is not None:
            inputs.append((bias_width, bias_min_size))

        # scale
        if scale_min_size is not None:
            inputs.append((scale_width, scale_min_size))

        # vshamt_mul
        if vshamt_mul_min_size is not None:
            inputs.append((vshamt_mul_width, vshamt_mul_min_size))

        # vshamt_sum
        if vshamt_sum_min_size is not None:
            inputs.append((vshamt_sum_width, vshamt_sum_min_size))

        # vshamt_out
        if vshamt_out_min_size is not None:
            inputs.append((vshamt_out_width, vshamt_out_min_size))

        outputs = []
        # out
        outputs.extend([(output_width, output_min_size)] *
                       self.par_row * self.par_col)

        temps = []

        return inputs, outputs, temps

    def get_min_concur_och(self):
        if self.maxi.datawidth < self.get_ram_width():
            min_concur_och = 1
        else:
            min_concur_och = self.maxi.datawidth // self.get_ram_width()

        return min_concur_och

    def get_req_concur_och(self):
        min_concur_och = self.get_min_concur_och()

        req_concur_och = 1
        if self.par_och is not None:
            req_concur_och = self.par_och

        if self.concur_och is not None:
            req_concur_och = max(req_concur_och, self.concur_och)

        req_concur_och = int(math.ceil(req_concur_och / min_concur_och)) * min_concur_och
        return req_concur_och

    def get_required_substreams(self):
        arg_scale = (self.args[self.args_dict['scale']]
                     if 'scale' in self.args_dict else None)

        x_datawidth = self.args[0].get_op_width()
        x_point = self.args[0].get_op_point()
        x_signed = self.args[0].get_signed()
        y_datawidth = self.args[1].get_op_width()
        y_point = self.args[1].get_op_point()
        y_signed = self.args[1].get_signed()

        if self.mul_dtype is not None:
            mul_width = self.mul_dtype.width
        else:
            mul_width = x_datawidth + y_datawidth

        if self.mul_dtype is not None:
            mul_point = self.mul_dtype.point
        else:
            mul_point = self.get_op_point()

        if self.mul_dtype is not None:
            mul_signed = self.mul_dtype.signed
        else:
            mul_signed = self.get_signed()

        if self.sum_dtype is not None:
            sum_width = self.sum_dtype.width
        else:
            sum_width = mul_width

        if self.sum_dtype is not None:
            sum_point = self.sum_dtype.point
        else:
            sum_point = mul_point

        if self.sum_dtype is not None:
            sum_signed = self.sum_dtype.signed
        else:
            sum_signed = mul_signed

        if arg_scale is not None:
            scale_width = arg_scale.get_op_width()
        else:
            scale_width = self.get_op_width()

        if arg_scale is not None:
            scale_point = arg_scale.get_op_point()
        else:
            scale_point = self.get_op_point()

        if arg_scale is not None:
            scale_signed = arg_scale.get_signed()
        else:
            scale_signed = self.get_signed()

        scl_width = sum_width + scale_width
        scl_point = max(sum_point, scale_point)
        scl_signed = sum_signed and scale_signed

        out_width = self.get_op_width()
        out_point = self.get_op_point()
        out_signed = self.get_signed()

        filter_num_col = self.filter_shape[-2]
        filter_num_row = self.filter_shape[-3]
        num_weights = filter_num_col * filter_num_row

        args = (x_datawidth, x_point, x_signed,
                y_datawidth, y_point, y_signed,
                mul_width, mul_point, mul_signed)

        if mul_point == 0:
            mulname = 'mul_rshift_round_madd'
        else:
            mulname = 'mul_rshift_round'

        substrms = [(mulname, args)] * (num_weights * self.par_ich *
                                        self.par_och * self.par_col * self.par_row)

        substrms.extend([('add_tree',
                          (sum_width, sum_point, sum_signed,
                           num_weights * self.par_ich))] *
                        self.par_och * self.par_col * self.par_row)
        substrms.extend([('acc_rshift_round_frac',
                          (sum_width, sum_point, sum_signed,
                           sum_width, sum_point, sum_signed))] *
                        self.par_och * self.par_col * self.par_row)
        substrms.extend([('mul_rshift_clip',
                          (sum_width, sum_point, sum_signed,
                           scale_width, scale_point, scale_signed,
                           scl_width, scl_point, scl_signed,
                           out_width, out_point, out_signed))] *
                        self.par_och * self.par_col * self.par_row)

        return substrms

    def get_stream_hash(self):
        base = bt._Operator.get_stream_hash(self)
        filter_num_col = self.filter_shape[-2]
        filter_num_row = self.filter_shape[-3]
        stride_col = self.strides[-2]  # width
        stride_row = self.strides[-3]  # height
        num_srcs = ((filter_num_col + stride_col * (self.par_col - 1)) *
                    (filter_num_row + stride_row * (self.par_row - 1)))
        num_weights = (filter_num_col * filter_num_row *
                       self.par_ich * self.par_och *
                       self.par_col * self.par_row)
        return (base, filter_num_col, filter_num_row,
                self.mul_dtype, self.sum_dtype,
                self.par_ich, self.par_och, self.par_col, self.par_row,
                num_srcs, num_weights)

    def get_stream_func(self):

        def func(strm):
            arg_input = self.args[0]
            arg_filter = self.args[1]
            arg_bias = (self.args[self.args_dict['bias']]
                        if 'bias' in self.args_dict else None)
            arg_scale = (self.args[self.args_dict['scale']]
                         if 'scale' in self.args_dict else None)
            arg_vshamt_mul = (self.args[self.args_dict['vshamt_mul']]
                              if 'vshamt_mul' in self.args_dict else None)
            arg_vshamt_sum = (self.args[self.args_dict['vshamt_sum']]
                              if 'vshamt_sum' in self.args_dict else None)
            arg_vshamt_out = (self.args[self.args_dict['vshamt_out']]
                              if 'vshamt_out' in self.args_dict else None)

            filter_num_col = self.filter_shape[-2]
            filter_num_row = self.filter_shape[-3]
            stride_col = self.strides[-2]  # width
            stride_row = self.strides[-3]  # height
            src_num_col = filter_num_col + stride_col * (self.par_col - 1)
            src_num_row = filter_num_row + stride_row * (self.par_row - 1)
            num_srcs = src_num_col * src_num_row
            num_weights = filter_num_col * filter_num_row

            # constant
            size = strm.constant(datawidth=self.stream_reduce_size.bit_length(),
                                 signed=False)
            col_select = strm.constant(datawidth=bt.log_width(src_num_col),
                                       signed=False)
            row_select = strm.constant(datawidth=bt.log_width(src_num_row),
                                       signed=False)
            mask = strm.constant(datawidth=num_srcs, signed=False)

            omit_mask = strm.constant(datawidth=self.par_ich, signed=False)
            omit_counter = strm.Counter(size=size)
            omits = [strm.Ands(b, omit_counter == size - 1)
                     for b in omit_mask]

            # bias
            datawidth = (arg_bias.get_op_width()
                         if arg_bias is not None else self.get_op_width())
            vec_datawidth = datawidth * self.par_och
            point = (arg_bias.get_op_point()
                     if arg_bias is not None else self.get_op_point())
            signed = (arg_bias.get_signed()
                      if arg_bias is not None else self.get_signed())
            dup_bias = strm.constant(datawidth=1, signed=False)
            vec_bias = strm.source(datawidth=vec_datawidth, signed=False)

            split_bias = strm.Split(vec_bias, datawidth, point, signed, reverse=True)
            bias_list = [strm.Mux(dup_bias, split_bias[0], value) for value in split_bias]

            # scale
            datawidth = (arg_scale.get_op_width()
                         if arg_scale is not None else self.get_op_width())
            vec_datawidth = datawidth * self.par_och
            point = (arg_scale.get_op_point()
                     if arg_scale is not None else self.get_op_point())
            signed = (arg_scale.get_signed()
                      if arg_scale is not None else self.get_signed())
            dup_scale = strm.constant(datawidth=1, signed=False)
            vec_scale = strm.source(datawidth=vec_datawidth, signed=False)

            split_scale = strm.Split(vec_scale, datawidth, point, signed, reverse=True)
            scale_list = [strm.Mux(dup_scale, split_scale[0], value) for value in split_scale]

            # vshamt_mul
            datawidth = (arg_vshamt_mul.get_op_width()
                         if arg_vshamt_mul is not None else self.get_op_width())
            vec_datawidth = datawidth * self.par_och
            point = 0
            signed = False
            dup_vshamt_mul = strm.constant(datawidth=1, signed=False)
            vec_vshamt_mul = strm.source(datawidth=vec_datawidth, signed=False)

            split_vshamt_mul = strm.Split(vec_vshamt_mul, datawidth, point, signed, reverse=True)
            vshamt_mul_list = [strm.Mux(dup_vshamt_mul, split_vshamt_mul[0], value)
                               for value in split_vshamt_mul]

            # vshamt_sum
            datawidth = (arg_vshamt_sum.get_op_width()
                         if arg_vshamt_sum is not None else self.get_op_width())
            vec_datawidth = datawidth * self.par_och
            point = 0
            signed = False
            dup_vshamt_sum = strm.constant(datawidth=1, signed=False)
            vec_vshamt_sum = strm.source(datawidth=vec_datawidth, signed=False)

            split_vshamt_sum = strm.Split(vec_vshamt_sum, datawidth, point, signed, reverse=True)
            vshamt_sum_list = [strm.Mux(dup_vshamt_sum, split_vshamt_sum[0], value)
                               for value in split_vshamt_sum]

            # vshamt_out
            datawidth = (arg_vshamt_out.get_op_width()
                         if arg_vshamt_out is not None else self.get_op_width())
            vec_datawidth = datawidth * self.par_och
            point = 0
            signed = False
            dup_vshamt_out = strm.constant(datawidth=1, signed=False)
            vec_vshamt_out = strm.source(datawidth=vec_datawidth, signed=False)

            split_vshamt_out = strm.Split(vec_vshamt_out, datawidth, point, signed, reverse=True)
            vshamt_out_list = [strm.Mux(dup_vshamt_out, split_vshamt_out[0], value)
                               for value in split_vshamt_out]

            # cshamt
            cshamt_mul = strm.constant(datawidth=self.cshamt_mul_value.bit_length(),
                                       signed=False)
            cshamt_sum = strm.constant(datawidth=self.cshamt_sum_value.bit_length(),
                                       signed=False)
            cshamt_out = strm.constant(datawidth=self.cshamt_out_value.bit_length(),
                                       signed=False)

            act_func_index_width = max(len(self.shared_attrs['act_func']).bit_length(), 1)
            act_func_index = strm.constant(datawidth=act_func_index_width, signed=False)

            # act
            act_rams = self.input_rams[:num_srcs]

            # vec_act
            datawidth = arg_input.get_op_width()
            vec_datawidth = arg_input.get_op_width() * self.par_ich
            point = arg_input.get_op_point()
            signed = arg_input.get_signed()

            vec_act_vars = []
            for act_ram in act_rams:
                vec_act_var = strm.source(datawidth=vec_datawidth, signed=False)
                vec_act_vars.append(vec_act_var)

            # vec_act MUX (col)
            vec_act_vars = line_to_2d(vec_act_vars, src_num_col)

            mux_vec_act_vars = []
            for act_line in vec_act_vars:
                for j in range(src_num_col):
                    sel = 0

                    for i in reversed(range(len(act_line))):
                        sel = strm.Mux(col_select == i,
                                       act_line[(i + j) % src_num_col], sel)

                    mux_vec_act_vars.append(sel)

            # vec_act MUX (row)
            vec_act_vars = transpose_2d(line_to_2d(mux_vec_act_vars, src_num_col))
            mux_vec_act_vars = []
            for act_line in vec_act_vars:
                for j in range(src_num_row):
                    sel = 0

                    for i in reversed(range(len(act_line))):
                        sel = strm.Mux(row_select == i,
                                       act_line[(i + j) % src_num_row], sel)

                    mux_vec_act_vars.append(sel)

            mux_vec_act_vars = transpose_2d(line_to_2d(mux_vec_act_vars, src_num_row))
            vec_act_vars = []
            for l in mux_vec_act_vars:
                vec_act_vars.extend(l)

            # vec_act -> act
            act_vars_list = []
            split_vec_act_vars = [strm.Split(vec_act_var,
                                             datawidth, point, signed, reverse=True)
                                  for vec_act_var in vec_act_vars]

            for i in range(self.par_ich):
                if self.par_ich == 1:
                    act_vars = [strm.ReinterpretCast(vec_act_var, datawidth, point, signed)
                                for vec_act_var in vec_act_vars]
                else:
                    omit = omits[i]
                    act_vars = [strm.Mux(omit, strm.Int(0), split_vec_act_var[i])
                                for split_vec_act_var in split_vec_act_vars]

                act_vars_list.append(act_vars)

            # filter
            filter_rams = self.input_rams[num_srcs:
                                          num_srcs + num_weights * self.par_och]

            # vec_filter
            datawidth = arg_filter.get_op_width()
            vec_datawidth = arg_filter.get_op_width() * self.par_ich
            point = arg_filter.get_op_point()
            signed = arg_filter.get_signed()

            vec_filter_vars_all = []
            for filter_ram in filter_rams:
                vec_filter_var = strm.source(datawidth=vec_datawidth, signed=False)
                vec_filter_vars_all.append(vec_filter_var)

            vec_filter_vars_och = [vec_filter_vars_all[i:i + num_weights]
                                   for i in range(0, len(vec_filter_vars_all), num_weights)]

            # vec_filter -> filter
            filter_vars_list_och = []
            for vec_filter_vars in vec_filter_vars_och:

                filter_vars_list = []
                split_vec_filter_vars = [strm.Split(vec_filter_var,
                                                    datawidth, point, signed, reverse=True)
                                         for vec_filter_var in vec_filter_vars]

                for i in range(self.par_ich):
                    if self.par_ich == 1:
                        filter_vars = [strm.ReinterpretCast(vec_filter_var,
                                                            datawidth, point, signed)
                                       for vec_filter_var in vec_filter_vars]
                    else:
                        omit = omits[i]
                        filter_vars = [strm.Mux(omit, strm.Int(0), split_vec_filter_var[i])
                                       for split_vec_filter_var in split_vec_filter_vars]

                    filter_vars_list.append(filter_vars)

                filter_vars_list_och.append(filter_vars_list)

            mask_2d = line_to_2d(mask, src_num_col)

            # pixel parallel
            for pos_row in range(self.par_row):
                for pos_col in range(self.par_col):

                    mask = []
                    for mask_row in mask_2d[pos_row * stride_row:
                                            pos_row * stride_row + filter_num_row]:
                        mask.extend(mask_row[pos_col * stride_col:
                                             pos_col * stride_col + filter_num_col])

                    out_vars = []
                    out_valids = []

                    # output channel parallel
                    for oc, (filter_vars_list, bias, scale,
                             vshamt_mul, vshamt_sum, vshamt_out) in enumerate(zip(
                            filter_vars_list_och, bias_list, scale_list,
                                 vshamt_mul_list, vshamt_sum_list, vshamt_out_list)):

                        mul_vars = []

                        # input channel parallel
                        for ic, (act_vars, filter_vars) in enumerate(zip(
                                act_vars_list, filter_vars_list)):

                            act_vars_2d = line_to_2d(act_vars, src_num_col)
                            used_act_vars = []
                            masked_used_act_vars = []
                            for act_row in act_vars_2d[pos_row * stride_row:
                                                       pos_row * stride_row + filter_num_row]:
                                used_act_vars.extend(act_row[pos_col * stride_col:
                                                             pos_col * stride_col + filter_num_col])

                            # mul
                            if len(used_act_vars) > len(mask):
                                raise ValueError('Not enough mask bits.')

                            for used_act_var, pmask in zip(used_act_vars, mask):
                                masked_var = strm.Mux(pmask, strm.Int(0), used_act_var)
                                masked_used_act_vars.append(masked_var)

                            for submul, act_var, filter_var in zip(
                                    self.substreams[
                                        num_weights *
                                        (pos_row * self.par_col * self.par_och * self.par_ich +
                                           pos_col * self.par_och * self.par_ich +
                                         oc * self.par_ich + ic):],
                                    masked_used_act_vars, filter_vars):

                                mul = strm.substream(submul)
                                mul.to_source('x', act_var)
                                mul.to_source('y', filter_var)
                                mul.to_source('rshift', vshamt_mul + cshamt_mul)
                                mul_var = mul.from_sink('z')
                                mul_vars.append(mul_var)

                        # add, scale, bias, vshamt_sum, vshamt_out
                        mul_vars_group = [mul_vars[i:i + num_weights]
                                          for i in range(0, len(mul_vars), num_weights)]
                        reshape_mul_vars = []
                        for vs in zip(*mul_vars_group):
                            for v in vs:
                                reshape_mul_vars.append(v)

                        addtree = strm.substream(self.substreams[
                            num_weights *
                            self.par_row * self.par_col * self.par_och * self.par_ich +
                            pos_row * self.par_col * self.par_och +
                            pos_col * self.par_och + oc])
                        for i, mul_var in enumerate(reshape_mul_vars):
                            addtree.to_source('var%d' % i, mul_var)

                        sum_var = addtree.from_sink('sum')

                        acc = strm.substream(self.substreams[
                            num_weights *
                            self.par_row * self.par_col * self.par_och * self.par_ich +
                            self.par_row * self.par_col * self.par_och +
                            pos_row * self.par_col * self.par_och +
                            pos_col * self.par_och + oc])
                        acc.to_source('x', sum_var)
                        acc.to_source('rshift', vshamt_sum + cshamt_sum)
                        acc.to_constant('size', size)
                        out_var = acc.from_sink('sum')
                        out_valid = acc.from_sink('valid')
                        out_valids.append(out_valid)

                        out_var += bias

                        mul = strm.substream(self.substreams[
                            num_weights *
                            self.par_row * self.par_col * self.par_och * self.par_ich +
                            self.par_row * self.par_col * self.par_och * 2 +
                            pos_row * self.par_col * self.par_och +
                            pos_col * self.par_och + oc])
                        mul.to_source('x', out_var)
                        mul.to_source('y', scale)
                        mul.to_source('rshift', vshamt_out + cshamt_out)

                        out_var = mul.from_sink('z')

                        act_func_vars = []
                        for act_func in self.shared_attrs['act_func']:
                            if act_func is not None:
                                act_func_vars.append(act_func.op(strm, out_var))
                            else:
                                act_func_vars.append(out_var)

                        if len(act_func_vars) == 1:
                            out_var = act_func_vars[0]
                        else:
                            for i, act_func_var in enumerate(act_func_vars):
                                out_var = strm.Mux(act_func_index == i, act_func_var, out_var)

                        width = self.get_op_width()
                        point = self.get_op_point()
                        signed = self.get_signed()

                        out_var = bt.out_rcast(strm, out_var, width, point, signed)
                        out_vars.append(out_var)

                    # out_vars -> vec_out_var
                    if self.par_och == 1:
                        vec_out_var = out_vars[0]
                    else:
                        vec_out_var = strm.Cat(*reversed(out_vars))

                    vec_out_valid = out_valids[0]
                    strm.sink(vec_out_var, when=vec_out_valid)

        return func

    def get_control_param_values(self):
        arg_input = self.args[0]
        arg_filter = self.args[1]
        arg_bias = (self.args[self.args_dict['bias']]
                    if 'bias' in self.args_dict else None)
        arg_scale = (self.args[self.args_dict['scale']]
                     if 'scale' in self.args_dict else None)
        arg_vshamt_mul = (self.args[self.args_dict['vshamt_mul']]
                          if 'vshamt_mul' in self.args_dict else None)
        arg_vshamt_sum = (self.args[self.args_dict['vshamt_sum']]
                          if 'vshamt_sum' in self.args_dict else None)
        arg_vshamt_out = (self.args[self.args_dict['vshamt_out']]
                          if 'vshamt_out' in self.args_dict else None)

        act = arg_input

        act_shape = self.input_shape
        act_num_ch = act_shape[-1]
        act_num_col = act_shape[-2]
        act_num_row = act_shape[-3]
        act_num_bat = act_shape[-4]

        filter = arg_filter
        filter_shape = self.filter_shape
        filter_num_col = filter_shape[-2]
        filter_num_row = filter_shape[-3]
        filter_num_och = filter_shape[-4]

        bias_scala = 1 if arg_bias is not None and arg_bias.shape[-1] == 1 else 0
        bias_num = (int(math.ceil(arg_bias.shape[-1] / self.par_och))
                    if arg_bias is not None else 0)
        scale_scala = 1 if arg_scale is not None and arg_scale.shape[-1] == 1 else 0
        scale_num = (int(math.ceil(arg_scale.shape[-1] / self.par_och))
                     if arg_scale is not None else 0)

        vshamt_mul_scala = 1 if arg_vshamt_mul is not None and arg_vshamt_mul.shape[-1] == 1 else 0
        vshamt_mul_num = (int(math.ceil(arg_vshamt_mul.shape[-1] / self.par_och))
                          if arg_vshamt_mul is not None else 0)
        vshamt_sum_scala = 1 if arg_vshamt_sum is not None and arg_vshamt_sum.shape[-1] == 1 else 0
        vshamt_sum_num = (int(math.ceil(arg_vshamt_sum.shape[-1] / self.par_och))
                          if arg_vshamt_sum is not None else 0)
        vshamt_out_scala = 1 if arg_vshamt_out is not None and arg_vshamt_out.shape[-1] == 1 else 0
        vshamt_out_num = (int(math.ceil(arg_vshamt_out.shape[-1] / self.par_och))
                          if arg_vshamt_out is not None else 0)

        cshamt_mul_value = 0 if self.cshamt_mul is None else self.cshamt_mul
        cshamt_sum_value = 0 if self.cshamt_sum is None else self.cshamt_sum
        cshamt_out_value = 0 if self.cshamt_out is None else self.cshamt_out

        act_func_index = self.shared_attrs['act_func'].index(self.act_func)

        # stride_ch = self.strides[-1]  # always 1
        stride_col = self.strides[-2]  # width
        stride_row = self.strides[-3]  # height
        stride_bat = self.strides[-4]  # always 1

        out_shape = self.orig_shape
        out_num_ch = out_shape[-1]
        out_num_col = out_shape[-2]
        out_num_row = out_shape[-3]
        out_num_bat = out_shape[-4]

        if isinstance(self.padding, str) and self.padding == 'SAME':
            # opposite order to pool
            pad_col, pad_col_right, pad_col_left = util.pad_size_split(
                act_num_col, filter_num_col, stride_col)
            pad_row, pad_row_bottom, pad_row_top = util.pad_size_split(
                act_num_row, filter_num_row, stride_row)
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

        # actual concur_och based on the RAM sizes
        src_num_col = filter_num_col + stride_col * (self.par_col - 1)
        src_num_row = filter_num_row + stride_row * (self.par_row - 1)
        num_srcs = src_num_col * src_num_row
        num_weights = filter_num_col * filter_num_row
        filter_rams = self.input_rams[num_srcs:
                                      num_srcs + num_weights * self.par_och]
        out_rams = self.output_rams
        out_ram_size = out_rams[0].length

        filter_ram_size = filter_rams[0].length
        flimit = (filter_ram_size //
                  int(math.ceil((act_num_ch / self.par_ich))) // 2 *
                  self.par_och)
        olimit = (out_ram_size // out_num_col // 2) * self.par_och
        concur_och = min(flimit, olimit)

        min_concur_och = self.get_min_concur_och()

        if self.concur_och is not None and concur_och > self.concur_och:
            concur_och = self.concur_och

        concur_och = max(int(math.floor(concur_och / min_concur_och)) * min_concur_och,
                         min_concur_och)

        # for __str__
        self.concur_och_value = concur_och

        # max counter values
        max_col_count = act_num_col + pad_col + 1 - filter_num_col - stride_col * self.par_col
        if max_col_count < 0:
            max_col_count = 0

        max_row_count = act_num_row + pad_row + 1 - filter_num_row - stride_row * self.par_row
        if max_row_count < 0:
            max_row_count = 0

        max_bat_count = act_num_bat - stride_bat
        if max_bat_count < 0:
            max_bat_count = 0

        max_och_count = (int(math.ceil(filter_num_och / self.par_och)) -
                         concur_och // self.par_och)
        if max_och_count < 0:
            max_och_count = 0

        och_count_step = concur_och // self.par_och

        dma_flag_conds = []
        for row_select in range(src_num_row):
            v = False
            for i in range(stride_row * self.par_row):
                v = v or (row_select == (i % src_num_row))

            dma_flag_conds.append(v)

        aligned_act_num_ch = bt.align_word(act_num_ch,
                                           act.get_word_alignment())

        act_step = bt.to_byte(aligned_act_num_ch * act.get_ram_width())

        act_offset_values = []
        for y in range(src_num_row):
            v = act_num_col * (y - pad_row_top) * act_step
            act_offset_values.append(v)

        act_row_step = act_step * act_num_col * stride_row * self.par_row
        act_bat_step = act_step * act_num_col * act_num_row

        act_read_size = (int(math.ceil(aligned_act_num_ch / self.par_ich)) *
                         act_num_col)
        act_read_block = int(math.ceil(aligned_act_num_ch / self.par_ich))
        act_read_step = (int(math.ceil(aligned_act_num_ch / self.par_ich)) *
                         int(math.ceil(act_num_col / src_num_col)))

        aligned_filter_num_ich = bt.align_word(act_num_ch,
                                               filter.get_word_alignment())

        filter_step = bt.to_byte(
            aligned_filter_num_ich * filter.get_ram_width())

        filter_base_step = (filter_step * filter_num_col * filter_num_row *
                            min(filter_num_och, concur_och))

        filter_read_size = (int(math.ceil(aligned_filter_num_ich / self.par_ich)) *
                            filter_num_col * filter_num_row *
                            min(filter_num_och, concur_och))
        filter_read_block = int(
            math.ceil(aligned_filter_num_ich / self.par_ich))
        filter_read_step = (int(math.ceil(aligned_filter_num_ich / self.par_ich)) *
                            min(int(math.ceil(filter_num_och / self.par_och)),
                                  concur_och // self.par_och))

        aligned_out_num_ch = bt.align_word(out_num_ch,
                                           self.get_word_alignment())

        out_step = bt.to_byte(aligned_out_num_ch * self.get_ram_width())

        out_offset_values = []
        for y in range(self.par_row):
            v = y * out_num_col * out_step
            out_offset_values.append(v)

        out_col_step = out_step
        out_row_step = out_step * out_num_col * self.par_row
        out_bat_step = out_step * out_num_col * out_num_row
        out_och_step = bt.to_byte(
            self.get_ram_width() * min(out_num_ch, concur_och))

        if (self.stationary == 'filter' and concur_och >= out_num_ch or
                self.stationary == 'input'):
            out_write_size = (int(math.ceil(aligned_out_num_ch / self.par_och)) *
                              out_num_col)
            out_write_size_res = out_write_size

            out_write_block = int(math.ceil(aligned_out_num_ch / self.par_och))

        else:
            out_write_size = int(math.ceil(min(aligned_out_num_ch, concur_och) /
                                           self.par_och))
            out_write_size_res = int(
                math.ceil(aligned_out_num_ch / self.par_och)) % out_write_size
            if out_write_size_res == 0:
                out_write_size_res = out_write_size

            out_write_block = 0

        keep_filter = concur_och >= out_num_ch

        act_ram_size = self.input_rams[0].length
        act_length = (act_read_step *
                      int(math.ceil(act_num_row / src_num_row)) *
                      act_num_bat)
        keep_input = (not self.disable_keep_input and
                      self.stationary == 'filter' and
                      act_ram_size >= act_length)

        # for __str__
        self.keep_filter_value = keep_filter
        self.keep_input_value = keep_input

        if self.stationary == 'filter':
            data_stationary = STATIONARY_FILETER
        else:
            data_stationary = STATIONARY_INPUT

        stream_num_ops = int(math.ceil(min(aligned_out_num_ch, concur_och) /
                                       self.par_och))
        stream_num_ops_res = (int(math.ceil(aligned_out_num_ch / self.par_och)) %
                              stream_num_ops)
        if stream_num_ops_res == 0:
            stream_num_ops_res = stream_num_ops

        stream_num_ops_par = stream_num_ops * self.par_col * self.par_row
        stream_num_ops_res_par = stream_num_ops_res * self.par_col * self.par_row

        stream_reduce_size = int(math.ceil(act_num_ch / self.par_ich))
        stream_aligned_reduce_size = int(
            math.ceil(aligned_filter_num_ich / self.par_ich))

        if act_num_ch % self.par_ich == 0:
            stream_omit_mask = 0
        else:
            stream_omit_mask = 0
            for i in range(self.par_ich):
                if self.par_ich - (act_num_ch % self.par_ich) >= (self.par_ich - i):
                    stream_omit_mask |= (0x1 << i)

        if pad_col_left == 0:
            col_select_initval = 0
        else:
            col_select_initval = (src_num_col - pad_col_left) % src_num_col

        stride_col_par_col = stride_col * self.par_col
        stride_row_par_row = stride_row * self.par_row

        stride_col_mod_filter_num = stride_col_par_col % src_num_col
        filter_num_col_minus_stride_col_mod = src_num_col - stride_col_mod_filter_num

        inc_act_laddr_conds = []
        for y in range(src_num_row):
            for x in range(src_num_col):
                for col_select in range(src_num_col):
                    v = False
                    for i in range(stride_col_mod_filter_num):
                        v = v or (col_select == ((x + src_num_col - i) %
                                                 src_num_col))

                    inc_act_laddr_conds.append(v)

        inc_act_laddr_small = (int(math.floor(stride_col_par_col / src_num_col)) *
                               int(math.ceil(aligned_act_num_ch / self.par_ich)))
        inc_act_laddr_large = (int(math.ceil(stride_col_par_col / src_num_col)) *
                               int(math.ceil(aligned_act_num_ch / self.par_ich)))
        inc_out_laddr_col = int(math.ceil(out_num_ch / self.par_och))

        stream_act_local_small_offset = (-1 * int(math.floor(pad_col_left / src_num_col)) *
                                         int(math.ceil(aligned_act_num_ch / self.par_ich)))
        stream_act_local_large_offset = (-1 * int(math.ceil(pad_col_left / src_num_col)) *
                                         int(math.ceil(aligned_act_num_ch / self.par_ich)))

        stream_act_local_small_flags = []
        stream_act_local_large_flags = []
        for x in range(src_num_col):
            s = (src_num_col - x) <= pad_col_left
            l = (src_num_col - x) <= (pad_col_left % src_num_col)
            stream_act_local_small_flags.append(s)
            stream_act_local_large_flags.append(s and l)

        if self.stationary == 'input':
            inc_sync_out = (int(math.ceil(out_num_col / self.par_col)) * self.par_col *
                            int(math.ceil(out_num_ch / concur_och)))
            inc_sync_out_res = 0
        elif self.stationary == 'filter' and concur_och >= out_num_ch:
            inc_sync_out = int(math.ceil(out_num_col / self.par_col)) * self.par_col
            inc_sync_out_res = 0
        else:
            inc_sync_out = 1
            inc_sync_out_res = (self.par_col - (out_num_col % self.par_col)) % self.par_col

        return OrderedDict([('act_num_col', act_num_col),
                            ('act_num_row', act_num_row),
                            ('filter_num_och', filter_num_och),
                            ('bias_scala', bias_scala),
                            ('bias_num', bias_num),
                            ('scale_scala', scale_scala),
                            ('scale_num', scale_num),
                            ('vshamt_mul_scala', vshamt_mul_scala),
                            ('vshamt_mul_num', vshamt_mul_num),
                            ('vshamt_sum_scala', vshamt_sum_scala),
                            ('vshamt_sum_num', vshamt_sum_num),
                            ('vshamt_out_scala', vshamt_out_scala),
                            ('vshamt_out_num', vshamt_out_num),
                            ('cshamt_mul_value', cshamt_mul_value),
                            ('cshamt_sum_value', cshamt_sum_value),
                            ('cshamt_out_value', cshamt_out_value),
                            ('act_func_index', act_func_index),
                            ('out_num_col', out_num_col),
                            ('out_num_row', out_num_row),
                            ('pad_col_left', pad_col_left),
                            ('pad_row_top', pad_row_top),
                            ('max_col_count', max_col_count),
                            ('max_row_count', max_row_count),
                            ('max_bat_count', max_bat_count),
                            ('max_och_count', max_och_count),
                            ('och_count_step', och_count_step),
                            ('dma_flag_conds', dma_flag_conds),
                            ('act_offset_values', act_offset_values),
                            ('act_row_step', act_row_step),
                            ('act_bat_step', act_bat_step),
                            ('act_read_size', act_read_size),
                            ('act_read_block', act_read_block),
                            ('act_read_step', act_read_step),
                            ('filter_base_step', filter_base_step),
                            ('filter_read_size', filter_read_size),
                            ('filter_read_block', filter_read_block),
                            ('filter_read_step', filter_read_step),
                            ('out_offset_values', out_offset_values),
                            ('out_col_step', out_col_step),
                            ('out_row_step', out_row_step),
                            ('out_bat_step', out_bat_step),
                            ('out_och_step', out_och_step),
                            ('out_write_size', out_write_size),
                            ('out_write_size_res', out_write_size_res),
                            ('out_write_block', out_write_block),
                            ('keep_filter', keep_filter),
                            ('keep_input', keep_input),
                            ('data_stationary', data_stationary),
                            ('stream_num_ops', stream_num_ops),
                            ('stream_num_ops_res', stream_num_ops_res),
                            ('stream_num_ops_par', stream_num_ops_par),
                            ('stream_num_ops_res_par', stream_num_ops_res_par),
                            ('stream_reduce_size', stream_reduce_size),
                            ('stream_aligned_reduce_size', stream_aligned_reduce_size),
                            ('stream_omit_mask', stream_omit_mask),
                            ('col_select_initval', col_select_initval),
                            ('stride_col_par_col', stride_col_par_col),
                            ('stride_row_par_row', stride_row_par_row),
                            ('stride_col_mod_filter_num', stride_col_mod_filter_num),
                            ('filter_num_col_minus_stride_col_mod',
                             filter_num_col_minus_stride_col_mod),
                            ('inc_act_laddr_conds', inc_act_laddr_conds),
                            ('inc_act_laddr_small', inc_act_laddr_small),
                            ('inc_act_laddr_large', inc_act_laddr_large),
                            ('inc_out_laddr_col', inc_out_laddr_col),
                            ('stream_act_local_small_offset', stream_act_local_small_offset),
                            ('stream_act_local_large_offset', stream_act_local_large_offset),
                            ('stream_act_local_small_flags', stream_act_local_small_flags),
                            ('stream_act_local_large_flags', stream_act_local_large_flags),
                            ('inc_sync_out', inc_sync_out),
                            ('inc_sync_out_res', inc_sync_out_res)])

    def control_sequence(self, fsm):
        arg_input = self.args[0]
        arg_filter = self.args[1]
        arg_bias = (self.args[self.args_dict['bias']]
                    if 'bias' in self.args_dict else None)
        arg_scale = (self.args[self.args_dict['scale']]
                     if 'scale' in self.args_dict else None)
        arg_vshamt_mul = (self.args[self.args_dict['vshamt_mul']]
                          if 'vshamt_mul' in self.args_dict else None)
        arg_vshamt_sum = (self.args[self.args_dict['vshamt_sum']]
                          if 'vshamt_sum' in self.args_dict else None)
        arg_vshamt_out = (self.args[self.args_dict['vshamt_out']]
                          if 'vshamt_out' in self.args_dict else None)

        filter_num_col = self.filter_shape[-2]
        filter_num_row = self.filter_shape[-3]
        stride_col = self.strides[-2]  # width
        stride_row = self.strides[-3]  # height

        src_num_col = filter_num_col + stride_col * (self.par_col - 1)
        src_num_row = filter_num_row + stride_row * (self.par_row - 1)

        num_srcs = src_num_col * src_num_row
        num_weights = filter_num_col * filter_num_row

        self.stride_bat = 1

        act_rams = self.input_rams[:num_srcs]
        filter_rams = self.input_rams[num_srcs:
                                      num_srcs + num_weights * self.par_och]
        out_rams = self.output_rams

        num_basic_args = 2
        bias_ram = (self.input_rams[len(act_rams) + len(filter_rams) +
                                    self.args_dict['bias'] - num_basic_args]
                    if 'bias' in self.args_dict else None)
        scale_ram = (self.input_rams[len(act_rams) + len(filter_rams) +
                                     self.args_dict['scale'] - num_basic_args]
                     if 'scale' in self.args_dict else None)
        vshamt_mul_ram = (self.input_rams[len(act_rams) + len(filter_rams) +
                                          self.args_dict['vshamt_mul'] - num_basic_args]
                          if 'vshamt_mul' in self.args_dict else None)
        vshamt_sum_ram = (self.input_rams[len(act_rams) + len(filter_rams) +
                                          self.args_dict['vshamt_sum'] - num_basic_args]
                          if 'vshamt_sum' in self.args_dict else None)
        vshamt_out_ram = (self.input_rams[len(act_rams) + len(filter_rams) +
                                          self.args_dict['vshamt_out'] - num_basic_args]
                          if 'vshamt_out' in self.args_dict else None)

        act_base_offset = self.m.Wire(self._name('act_base_offset'),
                                      self.maxi.addrwidth, signed=True)
        act_base_offset_row = self.m.Reg(self._name('act_base_offset_row'),
                                         self.maxi.addrwidth, initval=0, signed=True)
        act_base_offset_bat = self.m.Reg(self._name('act_base_offset_bat'),
                                         self.maxi.addrwidth, initval=0, signed=True)

        act_base_offset.assign(act_base_offset_row + act_base_offset_bat)

        filter_base_offset = self.m.Reg(self._name('filter_base_offset'),
                                        self.maxi.addrwidth, initval=0, signed=True)

        next_stream_num_ops = self.m.Reg(self._name('next_stream_num_ops'),
                                         self.maxi.addrwidth, initval=0)

        out_base_offset = self.m.Wire(self._name('out_base_offset'),
                                      self.maxi.addrwidth, signed=True)
        out_base_offset_val = self.m.Reg(self._name('out_base_offset_val'),
                                         self.maxi.addrwidth, initval=0, signed=True)
        out_base_offset_col = self.m.Reg(self._name('out_base_offset_col'),
                                         self.maxi.addrwidth, initval=0, signed=True)
        out_base_offset_row = self.m.Reg(self._name('out_base_offset_row'),
                                         self.maxi.addrwidth, initval=0, signed=True)
        out_base_offset_bat = self.m.Reg(self._name('out_base_offset_bat'),
                                         self.maxi.addrwidth, initval=0, signed=True)
        out_base_offset_och = self.m.Reg(self._name('out_base_offset_och'),
                                         self.maxi.addrwidth, initval=0, signed=True)

        out_base_offset.assign(out_base_offset_val + out_base_offset_col +
                               out_base_offset_row + out_base_offset_bat +
                               out_base_offset_och)

        dma_flags = [self.m.Reg(self._name('dma_flag_%d' % i), initval=0)
                     for i in range(src_num_row)]

        sync_comp_count = self.m.Reg(self._name('sync_comp_count'),
                                     self.maxi.addrwidth, initval=0)
        sync_out_count = self.m.Reg(self._name('sync_out_count'),
                                    self.maxi.addrwidth, initval=0)

        write_count = self.m.Reg(self._name('write_count'),
                                 self.maxi.addrwidth, initval=0)
        next_out_write_size = self.m.Reg(self._name('next_out_write_size'),
                                         self.maxi.addrwidth, initval=0)

        col_count = self.m.Reg(self._name('col_count'),
                               self.maxi.addrwidth, initval=0)
        row_count = self.m.Reg(self._name('row_count'),
                               self.maxi.addrwidth, initval=0)
        bat_count = self.m.Reg(self._name('bat_count'),
                               self.maxi.addrwidth, initval=0)
        och_count = self.m.Reg(self._name('och_count'),
                               self.maxi.addrwidth, initval=0)
        col_select = self.m.Reg(self._name('col_select'),
                                bt.log_width(src_num_col),
                                initval=0)
        row_select = self.m.Reg(self._name('row_select'),
                                bt.log_width(src_num_row),
                                initval=0)

        out_col_count = self.m.Reg(self._name('out_col_count'),
                                   self.maxi.addrwidth, initval=0)
        out_row_count = self.m.Reg(self._name('out_row_count'),
                                   self.maxi.addrwidth, initval=0)

        out_ram_select = self.m.Reg(self._name('out_ram_select'),
                                    self.maxi.addrwidth, initval=0)

        prev_col_count = self.m.Reg(self._name('prev_col_count'),
                                    self.maxi.addrwidth, initval=0)
        prev_row_count = self.m.Reg(self._name('prev_row_count'),
                                    self.maxi.addrwidth, initval=0)
        prev_bat_count = self.m.Reg(self._name('prev_bat_count'),
                                    self.maxi.addrwidth, initval=0)
        prev_och_count = self.m.Reg(self._name('prev_och_count'),
                                    self.maxi.addrwidth, initval=0)

        prev_row_select = self.m.Reg(self._name('prev_row_select'),
                                     bt.log_width(src_num_row),
                                     initval=0)

        stream_act_locals = [self.m.Reg(self._name('stream_act_local_%d' % i),
                                        self.maxi.addrwidth, initval=0)
                             for i in range(len(act_rams))]

        stream_out_local_val = self.m.Reg(self._name('stream_out_local_val'),
                                          self.maxi.addrwidth, initval=0)
        stream_out_local_col = self.m.Reg(self._name('stream_out_local_col'),
                                          self.maxi.addrwidth, initval=0)
        stream_out_local = self.m.Wire(self._name('stream_out_local'),
                                       self.maxi.addrwidth)
        stream_out_local.assign(stream_out_local_val + stream_out_local_col)

        # double buffer control
        act_page_comp_offsets = [self.m.Reg(self._name('act_page_comp_offset_%d' % i),
                                            self.maxi.addrwidth, initval=0)
                                 for i in range(src_num_row)]
        act_page_dma_offsets = [self.m.Reg(self._name('act_page_dma_offset_%d' % i),
                                           self.maxi.addrwidth, initval=0)
                                for i in range(src_num_row)]

        filter_page_comp_offset = self.m.Reg(self._name('filter_page_comp_offset'),
                                             self.maxi.addrwidth, initval=0)
        filter_page_dma_offset = self.m.Reg(self._name('filter_page_dma_offset'),
                                            self.maxi.addrwidth, initval=0)

        out_page = self.m.Reg(self._name('out_page'), initval=0)
        out_page_comp_offset = self.m.Reg(self._name('out_page_comp_offset'),
                                          self.maxi.addrwidth, initval=0)
        out_page_dma_offset = self.m.Reg(self._name('out_page_dma_offset'),
                                         self.maxi.addrwidth, initval=0)
        out_laddr_offset = self.m.Reg(self._name('out_laddr_offset'),
                                      self.maxi.addrwidth, initval=0)

        act_page_size = act_rams[0].length
        filter_page_size = filter_rams[0].length
        out_page_size = out_rams[0].length // 2

        skip_read_filter = self.m.Reg(
            self._name('skip_read_filter'), initval=0)
        skip_read_act = self.m.Reg(self._name('skip_read_act'), initval=0)
        skip_comp = self.m.Reg(self._name('skip_comp'), initval=0)
        skip_write_out = self.m.Reg(self._name('skip_write_out'), initval=1)

        # --------------------
        # initialization phase
        # --------------------
        # ReadFilter: offset
        fsm(
            filter_base_offset(0)
        )

        filter_offset = filter_base_offset

        # ReadFilter: double buffer control
        fsm(
            filter_page_comp_offset(0),
            filter_page_dma_offset(0)
        )

        # ReadAct: offset
        fsm(
            act_base_offset_row(0),
            act_base_offset_bat(0)
        )

        # ReadAct: DMA flag
        for y, dma_flag in enumerate(dma_flags):
            fsm(
                dma_flag(1)
            )

        # ReadAct: double buffer control
        fsm(
            [act_page_comp_offset(0)
             for act_page_comp_offset in act_page_comp_offsets],
            [act_page_dma_offset(0)
             for act_page_dma_offset in act_page_dma_offsets]
        )

        # WriteOutput: offset
        fsm(
            out_base_offset_val(0),
            out_base_offset_col(0),
            out_base_offset_row(0),
            out_base_offset_bat(0),
            out_base_offset_och(0)
        )

        # WriteOut: double buffer control
        fsm(
            out_page(0),
            out_page_comp_offset(0),
            out_page_dma_offset(0),
            out_laddr_offset(0)
        )

        fsm(
            sync_out_count(0),
            write_count(0)
        )

        fsm(
            next_out_write_size(vg.Mux(self.max_och_count == 0,
                                       self.out_write_size_res,
                                       self.out_write_size))
        )

        # counter
        fsm(
            row_count(0),
            bat_count(0),
            och_count(0),
            row_select(0),
            prev_row_count(0),
            prev_bat_count(0),
            prev_och_count(0),
            prev_row_select(0)
        )

        # out counter
        fsm(
            out_col_count(0),
            out_row_count(0),
            out_ram_select(0)
        )

        # double buffer control
        fsm(
            skip_read_filter(0),
            skip_read_act(0),
            skip_comp(0),
            skip_write_out(1)
        )

        # --------------------
        # ReadBias phase
        # --------------------
        if bias_ram is not None:
            bias_read_size = self.bias_num
            bias_laddr = 0
            bias_gaddr = self.arg_objaddrs[self.args_dict['bias']]

            bt.bus_lock(self.maxi, fsm)
            bt.dma_read(self.maxi, fsm, bias_ram, bias_laddr,
                        bias_gaddr, bias_read_size, port=1)
            bt.bus_unlock(self.maxi, fsm)

        # --------------------
        # ReadScale phase
        # --------------------
        if scale_ram is not None:
            scale_read_size = self.scale_num
            scale_laddr = 0
            scale_gaddr = self.arg_objaddrs[self.args_dict['scale']]

            bt.bus_lock(self.maxi, fsm)
            bt.dma_read(self.maxi, fsm, scale_ram, scale_laddr,
                        scale_gaddr, scale_read_size, port=1)
            bt.bus_unlock(self.maxi, fsm)

        # --------------------
        # ReadVshamt phase
        # --------------------
        if vshamt_mul_ram is not None:
            vshamt_mul_read_size = self.vshamt_mul_num
            vshamt_mul_laddr = 0
            vshamt_mul_gaddr = self.arg_objaddrs[self.args_dict['vshamt_mul']]

            bt.bus_lock(self.maxi, fsm)
            bt.dma_read(self.maxi, fsm, vshamt_mul_ram, vshamt_mul_laddr,
                        vshamt_mul_gaddr, vshamt_mul_read_size, port=1)
            bt.bus_unlock(self.maxi, fsm)

        if vshamt_sum_ram is not None:
            vshamt_sum_read_size = self.vshamt_sum_num
            vshamt_sum_laddr = 0
            vshamt_sum_gaddr = self.arg_objaddrs[self.args_dict['vshamt_sum']]

            bt.bus_lock(self.maxi, fsm)
            bt.dma_read(self.maxi, fsm, vshamt_sum_ram, vshamt_sum_laddr,
                        vshamt_sum_gaddr, vshamt_sum_read_size, port=1)
            bt.bus_unlock(self.maxi, fsm)

        if vshamt_out_ram is not None:
            vshamt_out_read_size = self.vshamt_out_num
            vshamt_out_laddr = 0
            vshamt_out_gaddr = self.arg_objaddrs[self.args_dict['vshamt_out']]

            bt.bus_lock(self.maxi, fsm)
            bt.dma_read(self.maxi, fsm, vshamt_out_ram, vshamt_out_laddr,
                        vshamt_out_gaddr, vshamt_out_read_size, port=1)
            bt.bus_unlock(self.maxi, fsm)

        state_init = fsm.current

        # state_read_filter
        fsm.If(self.data_stationary == STATIONARY_FILETER).goto_next()

        # --------------------
        # ReadFilter phase
        # --------------------
        state_read_filter = fsm.current

        filter_gaddr = self.arg_objaddrs[1] + filter_offset
        filter_laddr = filter_page_dma_offset

        bt.bus_lock(self.maxi, fsm)
        if len(filter_rams) == 1:
            bt.dma_read(self.maxi, fsm, filter_rams[0], filter_laddr,
                        filter_gaddr, self.filter_read_size, port=1)
        else:
            bt.dma_read_block(self.maxi, fsm, filter_rams, filter_laddr,
                              filter_gaddr, self.filter_read_size,
                              self.filter_read_block, port=1)
        bt.bus_unlock(self.maxi, fsm)

        fsm.goto_next()
        state_read_filter_end = fsm.current
        fsm.If(skip_read_filter).goto_from(
            state_read_filter, state_read_filter_end)
        # state_read_act
        fsm.If(self.data_stationary == STATIONARY_FILETER).goto_next()

        # --------------------
        # ReadAct phase
        # --------------------
        state_read_act = fsm.current

        act_offsets = []
        for v in self.act_offset_values:
            act_offset = act_base_offset + v
            act_offsets.append(act_offset)

        act_gaddrs = []
        for act_offset in act_offsets:
            act_gaddr = self.arg_objaddrs[0] + act_offset
            act_gaddrs.append(act_gaddr)

        act_rams_2d = line_to_2d(act_rams, src_num_col)
        mux_act_gaddr_values = mux_1d(act_gaddrs, row_select, src_num_row)
        mux_act_gaddrs = []
        for i, mux_act_gaddr_value in enumerate(mux_act_gaddr_values):
            mux_act_gaddr = self.m.Wire(self._name('mux_act_gaddr_%d' % i),
                                        self.maxi.addrwidth)
            mux_act_gaddr.assign(mux_act_gaddr_value)
            mux_act_gaddrs.append(mux_act_gaddr)

        dma_pad_masks = []
        for y in range(src_num_row):
            v = vg.Ors((row_count + y < self.pad_row_top),
                       (row_count + y >= self.act_num_row + self.pad_row_top))
            dma_pad_mask = self.m.Wire(self._name('dma_pad_mask_%d' % y))
            dma_pad_mask.assign(v)
            dma_pad_masks.append(dma_pad_mask)

        mux_dma_pad_mask_values = mux_1d(
            dma_pad_masks, row_select, src_num_row)
        mux_dma_pad_masks = []
        for i, mux_dma_pad_mask_value in enumerate(mux_dma_pad_mask_values):
            mux_dma_pad_mask = self.m.Wire(
                self._name('mux_dma_pad_mask_%d' % i))
            mux_dma_pad_mask.assign(mux_dma_pad_mask_value)
            mux_dma_pad_masks.append(mux_dma_pad_mask)

        # determined at the previous phase
        mux_dma_flag_values = mux_1d(
            dma_flags, prev_row_select, src_num_row)
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
        # state_read_act
        fsm.If(self.data_stationary == STATIONARY_FILETER).goto_next()

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
        stream_act_locals_2d = line_to_2d(stream_act_locals, src_num_col)
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
            stream_out_local_col(0)
        )
        comp_fsm.If(self.data_stationary == STATIONARY_INPUT,
                    och_count == 0)(
            stream_out_local_val(0)
        )

        # count and sel
        comp_fsm(
            col_count(0)
        )
        comp_fsm(
            col_select(self.col_select_initval)
        )

        filter_page_comp_offset_buf = self.m.Reg(self._name('filter_page_comp_offset_buf'),
                                                 self.maxi.addrwidth, initval=0)
        act_page_comp_offset_bufs = [self.m.Reg(self._name('act_page_comp_offset_buf_%d' % i),
                                                self.maxi.addrwidth, initval=0)
                                     for i in range(src_num_row)]
        out_page_comp_offset_buf = self.m.Reg(self._name('out_page_comp_offset_buf'),
                                              self.maxi.addrwidth, initval=0)
        row_count_buf = self.m.Reg(self._name('row_count_buf'),
                                   self.maxi.addrwidth, initval=0)
        row_select_buf = self.m.Reg(self._name('row_select_buf'),
                                    bt.log_width(src_num_row),
                                    initval=0)
        och_count_buf = self.m.Reg(self._name('och_count_buf'),
                                   self.maxi.addrwidth, initval=0)

        comp_fsm(
            filter_page_comp_offset_buf(filter_page_comp_offset),
            [act_page_comp_offset_buf(act_page_comp_offset)
             for act_page_comp_offset_buf, act_page_comp_offset in zip(
                act_page_comp_offset_bufs, act_page_comp_offsets)],
            out_page_comp_offset_buf(out_page_comp_offset),
            row_count_buf(row_count),
            row_select_buf(row_select),
            och_count_buf(och_count)
        )

        comp_fsm(
            next_stream_num_ops(vg.Mux(och_count >= self.max_och_count,
                                       self.stream_num_ops_res, self.stream_num_ops))
        )

        comp_fsm.goto_next()

        # repeat comp
        comp_state_rep = comp_fsm.current

        # pad_mask
        stream_pad_masks = []

        for y in range(src_num_row):
            for x in range(src_num_col):
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

        stream_pad_masks_reg = self.m.Reg(self._name('stream_pad_masks'),
                                          len(stream_pad_masks), initval=0)
        comp_fsm(
            stream_pad_masks_reg(vg.Cat(*reversed(stream_pad_masks)))
        )
        comp_fsm.goto_next()

        stream_col_select = col_select
        stream_row_select = row_select_buf
        stream_masks = stream_pad_masks_reg

        # set_constant
        name = list(self.stream.constants.keys())[0]
        self.stream.set_constant(comp_fsm, name, self.stream_reduce_size)
        comp_fsm.set_index(comp_fsm.current - 1)

        name = list(self.stream.constants.keys())[1]
        self.stream.set_constant(comp_fsm, name, stream_col_select)
        comp_fsm.set_index(comp_fsm.current - 1)

        name = list(self.stream.constants.keys())[2]
        self.stream.set_constant(comp_fsm, name, stream_row_select)
        comp_fsm.set_index(comp_fsm.current - 1)

        name = list(self.stream.constants.keys())[3]
        self.stream.set_constant(comp_fsm, name, stream_masks)
        comp_fsm.set_index(comp_fsm.current - 1)

        # set_constant (omit_mask)
        name = list(self.stream.constants.keys())[4]
        self.stream.set_constant(comp_fsm, name, self.stream_omit_mask)
        comp_fsm.set_index(comp_fsm.current - 1)

        # set_constant and set_source (bias)
        if bias_ram is not None:
            name = list(self.stream.constants.keys())[5]
            dup_bias = self.bias_scala
            self.stream.set_constant(comp_fsm, name, dup_bias)
            comp_fsm.set_index(comp_fsm.current - 1)

            name = list(self.stream.sources.keys())[0]
            local = vg.Mux(bias_read_size == 1, 0, och_count_buf)
            stride = vg.Mux(bias_read_size == 1, 0, 1)
            pat = ((self.stream_reduce_size, 0),
                   (next_stream_num_ops, stride))
            self.stream.set_source_pattern(comp_fsm, name, bias_ram,
                                           local, pat)
            comp_fsm.set_index(comp_fsm.current - 1)

        else:
            name = list(self.stream.constants.keys())[5]
            dup_bias = 1
            self.stream.set_constant(comp_fsm, name, dup_bias)
            comp_fsm.set_index(comp_fsm.current - 1)

            name = list(self.stream.sources.keys())[0]
            default_bias = 0
            self.stream.set_source_empty(comp_fsm, name, default_bias)
            comp_fsm.set_index(comp_fsm.current - 1)

        # set_constant and set_source (scale)
        if scale_ram is not None:
            name = list(self.stream.constants.keys())[6]
            dup_scale = self.scale_scala
            self.stream.set_constant(comp_fsm, name, dup_scale)
            comp_fsm.set_index(comp_fsm.current - 1)

            name = list(self.stream.sources.keys())[1]
            local = vg.Mux(scale_read_size == 1, 0, och_count_buf)
            stride = vg.Mux(scale_read_size == 1, 0, 1)
            pat = ((self.stream_reduce_size, 0),
                   (next_stream_num_ops, stride))
            self.stream.set_source_pattern(comp_fsm, name, scale_ram,
                                           local, pat)
            comp_fsm.set_index(comp_fsm.current - 1)

        else:
            name = list(self.stream.constants.keys())[6]
            dup_scale = 1
            self.stream.set_constant(comp_fsm, name, dup_scale)
            comp_fsm.set_index(comp_fsm.current - 1)

            name = list(self.stream.sources.keys())[1]
            default_scale = 1
            self.stream.set_source_empty(comp_fsm, name, default_scale)
            comp_fsm.set_index(comp_fsm.current - 1)

        # set_constant and set_source (vshamt_mul)
        if vshamt_mul_ram is not None:
            name = list(self.stream.constants.keys())[7]
            dup_vshamt_mul = self.vshamt_mul_scala
            self.stream.set_constant(comp_fsm, name, dup_vshamt_mul)
            comp_fsm.set_index(comp_fsm.current - 1)

            name = list(self.stream.sources.keys())[2]
            local = vg.Mux(vshamt_mul_read_size == 1, 0, och_count_buf)
            stride = vg.Mux(vshamt_mul_read_size == 1, 0, 1)
            pat = ((self.stream_reduce_size, 0),
                   (next_stream_num_ops, stride))
            self.stream.set_source_pattern(comp_fsm, name, vshamt_mul_ram,
                                           local, pat)
            comp_fsm.set_index(comp_fsm.current - 1)

        else:
            name = list(self.stream.constants.keys())[7]
            dup_vshamt_mul = 1
            self.stream.set_constant(comp_fsm, name, dup_vshamt_mul)
            comp_fsm.set_index(comp_fsm.current - 1)

            name = list(self.stream.sources.keys())[2]
            default_vshamt_mul = 0
            self.stream.set_source_empty(comp_fsm, name, default_vshamt_mul)
            comp_fsm.set_index(comp_fsm.current - 1)

        # set_constant and set_source (vshamt_sum)
        if vshamt_sum_ram is not None:
            name = list(self.stream.constants.keys())[8]
            dup_vshamt_sum = self.vshamt_sum_scala
            self.stream.set_constant(comp_fsm, name, dup_vshamt_sum)
            comp_fsm.set_index(comp_fsm.current - 1)

            name = list(self.stream.sources.keys())[3]
            local = vg.Mux(vshamt_sum_read_size == 1, 0, och_count_buf)
            stride = vg.Mux(vshamt_sum_read_size == 1, 0, 1)
            pat = ((self.stream_reduce_size, 0),
                   (next_stream_num_ops, stride))
            self.stream.set_source_pattern(comp_fsm, name, vshamt_sum_ram,
                                           local, pat)
            comp_fsm.set_index(comp_fsm.current - 1)

        else:
            name = list(self.stream.constants.keys())[8]
            dup_vshamt_sum = 1
            self.stream.set_constant(comp_fsm, name, dup_vshamt_sum)
            comp_fsm.set_index(comp_fsm.current - 1)

            name = list(self.stream.sources.keys())[3]
            default_vshamt_sum = 0
            self.stream.set_source_empty(comp_fsm, name, default_vshamt_sum)
            comp_fsm.set_index(comp_fsm.current - 1)

        # set_constant and set_source (vshamt_out)
        if vshamt_out_ram is not None:
            name = list(self.stream.constants.keys())[9]
            dup_vshamt_out = self.vshamt_out_scala
            self.stream.set_constant(comp_fsm, name, dup_vshamt_out)
            comp_fsm.set_index(comp_fsm.current - 1)

            name = list(self.stream.sources.keys())[4]
            local = vg.Mux(vshamt_out_read_size == 1, 0, och_count_buf)
            stride = vg.Mux(vshamt_out_read_size == 1, 0, 1)
            pat = ((self.stream_reduce_size, 0),
                   (next_stream_num_ops, stride))
            self.stream.set_source_pattern(comp_fsm, name, vshamt_out_ram,
                                           local, pat)
            comp_fsm.set_index(comp_fsm.current - 1)

        else:
            name = list(self.stream.constants.keys())[9]
            dup_vshamt_out = 1
            self.stream.set_constant(comp_fsm, name, dup_vshamt_out)
            comp_fsm.set_index(comp_fsm.current - 1)

            name = list(self.stream.sources.keys())[4]
            default_vshamt_out = 0
            self.stream.set_source_empty(comp_fsm, name, default_vshamt_out)
            comp_fsm.set_index(comp_fsm.current - 1)

        # set_constant (cshamt_mul)
        name = list(self.stream.constants.keys())[10]
        self.stream.set_constant(comp_fsm, name, self.cshamt_mul_value)
        comp_fsm.set_index(comp_fsm.current - 1)

        # set_constant (cshamt_sum)
        name = list(self.stream.constants.keys())[11]
        self.stream.set_constant(comp_fsm, name, self.cshamt_sum_value)
        comp_fsm.set_index(comp_fsm.current - 1)

        # set_constant (cshamt_out)
        name = list(self.stream.constants.keys())[12]
        self.stream.set_constant(comp_fsm, name, self.cshamt_out_value)
        comp_fsm.set_index(comp_fsm.current - 1)

        # set_constant (act_func_index)
        name = list(self.stream.constants.keys())[13]
        self.stream.set_constant(comp_fsm, name, self.act_func_index)
        comp_fsm.set_index(comp_fsm.current - 1)

        # set_source (act)
        act_page_comp_offset_bufs_dup = []
        for act_page_comp_offset_buf in act_page_comp_offset_bufs:
            act_page_comp_offset_bufs_dup.extend(
                [act_page_comp_offset_buf] * src_num_col)

        stream_act_names = list(self.stream.sources.keys())[5:5 + num_srcs]
        for name, ram, stream_act_local, act_page_comp_offset_buf in zip(
                stream_act_names, act_rams,
                stream_act_locals, act_page_comp_offset_bufs_dup):
            local = stream_act_local + act_page_comp_offset_buf
            pat = ((self.stream_reduce_size, 1),
                   (next_stream_num_ops, 0))
            self.stream.set_source_pattern(comp_fsm, name, ram,
                                           local, pat)
            comp_fsm.set_index(comp_fsm.current - 1)

        # set_source (filter)
        stream_filter_names = list(self.stream.sources.keys())[5 + num_srcs:]

        for name, ram in zip(stream_filter_names, filter_rams):
            local = filter_page_comp_offset_buf
            pat = ((self.stream_reduce_size, 1),
                   (next_stream_num_ops, self.stream_aligned_reduce_size))
            self.stream.set_source_pattern(comp_fsm, name, ram,
                                           local, pat)
            comp_fsm.set_index(comp_fsm.current - 1)

        # set_sink (out)
        stream_out_names = list(self.stream.sinks.keys())
        # remove valid outputs
        stream_out_var_names = stream_out_names[::2]

        for name, ram in zip(stream_out_var_names, out_rams):
            local = stream_out_local + out_page_comp_offset_buf
            self.stream.set_sink(comp_fsm, name, ram, local,
                                 next_stream_num_ops)
            comp_fsm.set_index(comp_fsm.current - 1)

        comp_fsm.goto_next()

        # stream barrier of previous run
        self.stream.source_join_and_run(comp_fsm)

        # stream_act_local
        stream_act_locals_2d = line_to_2d(stream_act_locals, src_num_col)

        i = 0
        for y, stream_act_locals_row in enumerate(stream_act_locals_2d):
            for x, stream_act_local in enumerate(stream_act_locals_row):
                patterns = []
                for col in range(src_num_col):
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
        # STATIONARY_FILETER
        comp_fsm.If(self.data_stationary == STATIONARY_FILETER)(
            stream_out_local_col.add(next_stream_num_ops)
        )
        comp_fsm.If(self.data_stationary == STATIONARY_FILETER,
                    col_count >= self.max_col_count)(
            stream_out_local_col(0)
        )

        # STATIONARY_INPUT
        comp_fsm.If(self.data_stationary == STATIONARY_INPUT)(
            stream_out_local_col.add(self.inc_out_laddr_col)
        )
        comp_fsm.If(self.data_stationary == STATIONARY_INPUT,
                    col_count >= self.max_col_count)(
            stream_out_local_val.add(next_stream_num_ops),
            stream_out_local_col(0)
        )

        # counter
        comp_fsm(
            col_count.add(self.stride_col_par_col)
        )
        comp_fsm.If(col_count >= self.max_col_count)(
            col_count(0)
        )

        comp_fsm(
            col_select.add(self.stride_col_mod_filter_num)
        )
        comp_fsm.If(col_select + self.stride_col_mod_filter_num >= src_num_col)(
            col_select.sub(self.filter_num_col_minus_stride_col_mod)
        )

        comp_fsm.If(col_count >= self.max_col_count)(
            col_select(self.col_select_initval),
        )

        # repeat
        comp_fsm.goto(comp_state_rep)
        comp_fsm.If(col_count >= self.max_col_count).goto_init()

        # sync with WriteOut control
        comp_fsm.seq.If(self.stream.end_flag)(
            sync_comp_count.add(self.par_col)
        )
        comp_fsm.seq.If(fsm.state == state_init)(
            sync_comp_count(0)
        )

        # --------------------
        # WriteOut phase
        # --------------------
        state_write_out = fsm.current

        out_offsets = []
        for v in self.out_offset_values:
            out_offset = out_base_offset + v
            out_offsets.append(out_offset)

        out_gaddrs = []
        for out_offset in out_offsets:
            out_gaddr = self.objaddr + out_offset
            out_gaddrs.append(out_gaddr)

        dma_out_masks = []
        for y in range(self.par_row):
            v = out_row_count + y >= self.out_num_row
            w = self.m.Wire(self._name('dma_out_mask_%d' % y))
            w.assign(v)
            dma_out_masks.append(w)

        out_laddr = out_laddr_offset + out_page_dma_offset

        out_rams_2d = line_to_2d(out_rams, self.par_col)

        bt.bus_lock(self.maxi, fsm)

        fsm.If(sync_comp_count >=
               sync_out_count + self.inc_sync_out).goto_next()

        for out_rams_row, out_gaddr, dma_out_mask in zip(
                out_rams_2d, out_gaddrs, dma_out_masks):

            if len(out_rams_row) == 1:
                b = fsm.current
                fsm.If(vg.Not(dma_out_mask)).goto_next()

                bt.dma_write(self.maxi, fsm, out_rams_row[0], out_laddr,
                             out_gaddr, next_out_write_size,
                             port=1, use_async=True)

                e = fsm.current

                fsm.If(dma_out_mask).goto_from(b, e)
                fsm.goto_next()

            else:
                ends = []

                state_mode_select = fsm.current

                # keep_filter or STATIONARY_INPUT
                b = fsm.current
                fsm.If(vg.Ors(
                    vg.Ands(self.data_stationary == STATIONARY_FILETER, self.keep_filter),
                    self.data_stationary == STATIONARY_INPUT)).goto_from(state_mode_select, b)

                fsm.If(vg.Not(dma_out_mask)).goto_next()

                bt.dma_write_block(self.maxi, fsm, out_rams_row, out_laddr,
                                   out_gaddr, next_out_write_size,
                                   self.out_write_block, port=1, use_async=True)

                e = fsm.current
                ends.append(e)

                fsm.If(dma_out_mask).goto_from(b, e)
                fsm.inc()

                # not keep_filter and not STATIONARY_INPUT
                state_ram_select = fsm.current
                fsm.If(vg.Not(vg.Ors(
                    vg.Ands(self.data_stationary == STATIONARY_FILETER, self.keep_filter),
                    self.data_stationary == STATIONARY_INPUT))).goto_from(state_mode_select,
                                                                          state_ram_select)
                fsm.inc()

                for sel, out_ram in enumerate(out_rams_row):

                    b = fsm.current
                    fsm.If(out_ram_select == sel).goto_from(state_ram_select, b)

                    fsm.If(vg.Not(dma_out_mask)).goto_next()

                    bt.dma_write(self.maxi, fsm, out_ram, out_laddr,
                                 out_gaddr, next_out_write_size,
                                 port=1, use_async=True)

                    e = fsm.current
                    ends.append(e)

                    fsm.If(dma_out_mask).goto_from(b, e)
                    fsm.inc()

                done = fsm.current

                for e in ends:
                    fsm.goto_from(e, done)

        bt.bus_unlock(self.maxi, fsm)

        # STATIONARY_FILTER
        fsm(
            write_count.inc()
        )

        fsm.If(out_ram_select == self.par_col - 1)(
            out_laddr_offset.add(next_out_write_size)
        )

        fsm.If(self.data_stationary == STATIONARY_FILETER,
               vg.Not(self.keep_filter))(
            out_base_offset_col.add(self.out_col_step),
            out_col_count.inc()
        )

        fsm(
            out_ram_select.inc()
        )

        fsm.If(out_ram_select == self.par_col - 1)(
            out_ram_select(0)
        )

        fsm(
            sync_out_count.add(self.inc_sync_out)
        )

        fsm.If(vg.Ors(vg.Ands(self.data_stationary == STATIONARY_FILETER,
                              vg.Not(self.keep_filter),
                              write_count >= self.out_num_col - 1),
                      vg.Ands(self.data_stationary == STATIONARY_FILETER,
                              self.keep_filter),
                      self.data_stationary == STATIONARY_INPUT))(
            sync_out_count.add(self.inc_sync_out + self.inc_sync_out_res)
        )

        fsm.If(self.data_stationary == STATIONARY_FILETER,
               vg.Not(self.keep_filter)).goto(state_write_out)

        # STATIONARY_FILTER and STATIONARY_INPUT
        fsm.If(vg.Ors(vg.Ands(self.data_stationary == STATIONARY_FILETER,
                              vg.Not(self.keep_filter),
                              write_count >= self.out_num_col - 1),
                      vg.Ands(self.data_stationary == STATIONARY_FILETER,
                              self.keep_filter),
                      self.data_stationary == STATIONARY_INPUT)).goto_next()

        state_write_out_end = fsm.current
        fsm.If(skip_write_out).goto_from(state_write_out, state_write_out_end)
        fsm.If(self.data_stationary == STATIONARY_INPUT,
               prev_och_count < self.max_och_count).goto_from(state_write_out, state_write_out_end)

        # --------------------
        # update for next iteration
        # --------------------
        # ReadFilter: offset
        update_filter = self.m.Wire(self._name('update_filter'))
        update_filter.assign(vg.Ors(
            vg.Ands(self.data_stationary == STATIONARY_FILETER,
                    row_count >= self.max_row_count,
                    bat_count >= self.max_bat_count),
            vg.Ands(self.data_stationary == STATIONARY_INPUT,
                    vg.Not(self.keep_filter))))

        fsm.If(update_filter)(
            filter_base_offset.add(self.filter_base_step),
        )

        fsm.If(self.data_stationary == STATIONARY_INPUT,
               och_count >= self.max_och_count)(
            filter_base_offset(0)
        )

        # ReadFilter: counter
        fsm.If(update_filter)(
            och_count.add(self.och_count_step)
        )

        fsm.If(self.data_stationary == STATIONARY_INPUT,
               och_count >= self.max_och_count)(
            och_count(0)
        )

        # ReadFilter: double buffer
        fsm.If(update_filter)(
            filter_page_comp_offset.add(self.filter_read_step),
            filter_page_dma_offset.add(self.filter_read_step)
        )
        fsm.If(update_filter,
               filter_page_comp_offset + self.filter_read_step +
               self.filter_read_step > filter_page_size)(
            filter_page_comp_offset(0),
            filter_page_dma_offset(0)
        )

        # ReadAct: offset
        update_act = self.m.Wire(self._name('update_act'))
        update_act.assign(vg.Ors(
            vg.Ands(self.data_stationary == STATIONARY_INPUT,
                    och_count >= self.max_och_count),
            self.data_stationary == STATIONARY_FILETER))

        fsm.If(update_act)(
            act_base_offset_row.add(self.act_row_step)
        )
        fsm.If(update_act,
               row_count >= self.max_row_count)(
            act_base_offset_row(0),
            act_base_offset_bat.add(self.act_bat_step)
        )
        fsm.If(update_act,
               row_count >= self.max_row_count,
               bat_count >= self.max_bat_count)(
            act_base_offset_bat(0)
        )

        # ReadAct: DMA flag
        next_dma_flags = []
        for dma_flag, dma_flag_cond in zip(dma_flags, self.dma_flag_conds):
            fsm.If(vg.Not(update_act))(
                dma_flag(0)
            )
            fsm.If(update_act)(
                dma_flag(dma_flag_cond)
            )
            fsm.If(update_act,
                   row_count >= self.max_row_count)(
                dma_flag(1)
            )

            next_dma_flags.append(
                vg.Mux(row_count >= self.max_row_count, 1, dma_flag_cond))

        # ReadAct: counter
        fsm.If(update_act)(
            row_count.add(self.stride_row_par_row)
        )
        fsm.If(update_act,
               row_count >= self.max_row_count)(
            row_count(0),
            bat_count.add(self.stride_bat)
        )
        fsm.If(update_act,
               row_count >= self.max_row_count,
               bat_count >= self.max_bat_count)(
            bat_count(0)
        )

        fsm.If(update_act,
               self.stride_row_par_row < src_num_row)(
            row_select.add(self.stride_row_par_row),
            prev_row_select(row_select)
        )
        fsm.If(update_act,
               self.stride_row_par_row < src_num_row,
               row_select + self.stride_row_par_row >= src_num_row)(
            row_select(row_select - (vg.Int(src_num_row) - self.stride_row_par_row)),
            prev_row_select(row_select)
        )
        fsm.If(update_act,
               vg.Not(self.stride_row_par_row < src_num_row))(
            row_select(0),
            prev_row_select(0)
        )

        fsm.If(update_act,
               row_count >= self.max_row_count)(
            row_select(0),
            prev_row_select(0)
        )

        # ReadAct and Comp: double buffer
        mux_next_dma_flag_values = mux_1d(
            next_dma_flags, row_select, src_num_row)
        mux_next_dma_flags = []
        for i, mux_next_dma_flag_value in enumerate(mux_next_dma_flag_values):
            mux_next_dma_flag = self.m.Wire(
                self._name('mux_next_dma_flag_%d' % i))
            mux_next_dma_flag.assign(mux_next_dma_flag_value)
            mux_next_dma_flags.append(mux_next_dma_flag)

        for (act_page_comp_offset, act_page_dma_offset, mux_next_dma_flag) in zip(
                act_page_comp_offsets, act_page_dma_offsets, mux_next_dma_flags):

            fsm.If(update_act, mux_next_dma_flag)(
                act_page_comp_offset.add(self.act_read_step),
                act_page_dma_offset.add(self.act_read_step)
            )
            fsm.If(update_act, mux_next_dma_flag,
                   act_page_comp_offset + self.act_read_step +
                   self.act_read_step > act_page_size)(
                act_page_comp_offset(0),
                act_page_dma_offset(0)
            )
            fsm.If(self.data_stationary == STATIONARY_FILETER,
                   row_count >= self.max_row_count,
                   bat_count >= self.max_bat_count,
                   self.keep_input)(
                act_page_comp_offset(0),
                act_page_dma_offset(0)
            )

        # WriteOut: write_size
        fsm(
            next_out_write_size(vg.Mux(och_count >= self.max_och_count,
                                       self.out_write_size_res,
                                       self.out_write_size))
        )

        # WriteOut: counter
        fsm.If(vg.Not(skip_write_out))(
            write_count(0),
            out_laddr_offset(0),
            out_ram_select(0)
        )

        fsm.If(self.data_stationary == STATIONARY_FILETER,
               vg.Not(skip_write_out))(
            out_base_offset_col(0),
            out_base_offset_row.add(self.out_row_step),
            out_col_count(0),
            out_row_count.add(self.par_row)
        )
        fsm.If(self.data_stationary == STATIONARY_FILETER,
               vg.Not(skip_write_out),
               prev_row_count >= self.max_row_count)(
            out_base_offset_row(0),
            out_base_offset_bat.add(self.out_bat_step),
            out_row_count(0)
        )
        fsm.If(self.data_stationary == STATIONARY_FILETER,
               vg.Not(skip_write_out),
               prev_row_count >= self.max_row_count,
               prev_bat_count >= self.max_bat_count)(
            out_base_offset_bat(0),
            out_base_offset_och.add(self.out_och_step)
        )

        fsm.If(self.data_stationary == STATIONARY_INPUT,
               prev_och_count >= self.max_och_count,
               vg.Not(skip_write_out))(
            out_base_offset_row.add(self.out_row_step)
        )

        # WriteOut and Comp: double buffer
        fsm.If(self.data_stationary == STATIONARY_FILETER,
               vg.Not(out_page))(
            out_page_comp_offset(out_page_size),
            out_page_dma_offset(0),
            out_page(1)
        )
        fsm.If(self.data_stationary == STATIONARY_FILETER,
               out_page)(
            out_page_comp_offset(0),
            out_page_dma_offset(out_page_size),
            out_page(0)
        )

        fsm.If(self.data_stationary == STATIONARY_INPUT,
               och_count >= self.max_och_count,
               vg.Not(out_page))(
            out_page_comp_offset(out_page_size),
            out_page_dma_offset(0),
            out_page(1)
        )
        fsm.If(self.data_stationary == STATIONARY_INPUT,
               och_count >= self.max_och_count,
               out_page)(
            out_page_comp_offset(0),
            out_page_dma_offset(out_page_size),
            out_page(0)
        )

        # ReadAct and WriteOut: prev
        fsm(
            prev_row_count(row_count),
            prev_bat_count(bat_count),
            prev_och_count(och_count)
        )

        # ReadFilter, ReadAct, Comp, WriteOut: skip
        fsm.If(row_count >= self.max_row_count,
               bat_count >= self.max_bat_count,
               och_count >= self.max_och_count)(
            skip_read_filter(1)
        )

        fsm.If(self.data_stationary == STATIONARY_INPUT,
               self.keep_filter)(
            skip_read_filter(1)
        )

        fsm.If(row_count >= self.max_row_count,
               bat_count >= self.max_bat_count,
               och_count >= self.max_och_count)(
            skip_read_act(1)
        )

        fsm.If(self.data_stationary == STATIONARY_FILETER,
               row_count >= self.max_row_count,
               bat_count >= self.max_bat_count,
               self.keep_input)(
            skip_read_act(1)
        )

        fsm.If(row_count >= self.max_row_count,
               bat_count >= self.max_bat_count,
               och_count >= self.max_och_count)(
            skip_comp(1)
        )

        fsm.If(skip_write_out,
               prev_row_count == 0,
               prev_bat_count == 0,
               prev_och_count == 0)(
            skip_write_out(0)
        )

        fsm.If(self.data_stationary == STATIONARY_FILETER).goto(state_read_act)
        fsm.If(self.data_stationary == STATIONARY_FILETER,
               row_count >= self.max_row_count,
               bat_count >= self.max_bat_count).goto(state_read_filter)

        fsm.If(self.data_stationary == STATIONARY_INPUT).goto(state_read_filter)
        fsm.If(self.data_stationary == STATIONARY_INPUT,
               och_count >= self.max_och_count).goto(state_read_act)

        fsm.If(vg.Ands(vg.Not(skip_write_out),
                       prev_och_count >= self.max_och_count,
                       prev_row_count >= self.max_row_count,
                       prev_bat_count >= self.max_bat_count)).goto_next()

        # wait for last DMA write
        bt.dma_wait_write(self.maxi, fsm)

        # --------------------
        # FSM controls for STATIONARY_INPUT (insert FSM transitions)
        # --------------------
        fsm.If(self.data_stationary == STATIONARY_INPUT).goto_from(
            state_init, state_read_act)
        fsm.If(self.data_stationary == STATIONARY_INPUT).goto_from(
            state_read_act_end, state_read_filter)
        fsm.If(self.data_stationary == STATIONARY_INPUT).goto_from(
            state_read_filter_end, state_comp)

    def eval(self, memo, input_dict, **kwargs):
        if id(self) in memo:
            return memo[id(self)]

        import nngen.verify as verify

        name = self.__class__.__name__
        method = getattr(verify, name, None)

        args = [arg.eval(memo, input_dict)
                for arg in self.args]

        input = args[0]
        filter = args[1]
        strides = self.strides

        bias = args[self.args_dict['bias']] if self.has_bias else None
        scale = args[self.args_dict['scale']] if self.has_scale else None

        rshift_mul = (args[self.args_dict['vshamt_mul']]
                      if self.has_vshamt_mul else self.cshamt_mul)
        rshift_sum = (args[self.args_dict['vshamt_sum']]
                      if self.has_vshamt_sum else self.cshamt_sum)
        rshift_out = (args[self.args_dict['vshamt_out']]
                      if self.has_vshamt_out else self.cshamt_out)

        kwargs['bias'] = bias
        kwargs['scale'] = scale
        kwargs['rshift_mul'] = rshift_mul
        kwargs['rshift_sum'] = rshift_sum
        kwargs['rshift_out'] = rshift_out
        kwargs['act_func'] = self.act_func
        kwargs['padding'] = self.padding
        kwargs['dtype'] = self.dtype
        kwargs['mul_dtype'] = self.mul_dtype
        kwargs['sum_dtype'] = self.sum_dtype
        kwargs['name'] = self.name
        kwargs['par_ich'] = self.par_ich
        kwargs['par_och'] = self.par_och
        kwargs['par_col'] = self.par_col
        kwargs['par_row'] = self.par_row
        kwargs['concur_och'] = self.concur_och
        kwargs['stationary'] = self.stationary
        kwargs['input_dtype'] = self.args[0].dtype
        kwargs['filter_dtype'] = self.args[1].dtype
        kwargs['bias_dtype'] = self.args[self.args_dict['bias']].dtype if self.has_bias else None
        kwargs['scale_dtype'] = self.args[self.args_dict['scale']].dtype if self.has_scale else None

        ret = method(input, filter, strides, **kwargs)
        memo[id(self)] = ret

        return ret


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
                             # line[(i + j) % col_size], ret)
                             line[(j + col_size - i) % col_size], ret)
            ret_list.append(ret)

    mat = transpose_2d(line_to_2d(ret_list, col_size))

    ret_list = []
    for line in mat:
        for j in range(row_size):
            ret = vg.Int(0, width=width)
            for i in reversed(range(len(line))):
                ret = vg.Mux(row_select == i,
                             # line[(i + j) % row_size], ret)
                             line[(j + row_size - i) % row_size], ret)
            ret_list.append(ret)

    return transpose_2d(line_to_2d(ret_list, row_size))


def mux_1d(line, select, size, width=1):
    ret_list = []
    for j in range(size):
        ret = vg.Int(0, width=width)
        for i in reversed(range(len(line))):
            ret = vg.Mux(select == i,
                         # line[(i + j) % size], ret)
                         line[(j + size - i) % size], ret)
        ret_list.append(ret)

    return ret_list


def to_aligned_shape(obj, shape):
    if obj.maxi is None:
        raise ValueError("maxi is required to determine alignment.")

    num_words = obj.get_word_alignment()

    aligned_shape = []
    for s in shape[:-1]:
        aligned_shape.append(s)

    res = num_words - shape[-1] % num_words
    if res == num_words:
        res = 0

    aligned_shape.append(shape[-1] + res)

    return aligned_shape
