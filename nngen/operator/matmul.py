from __future__ import absolute_import
from __future__ import print_function

import functools

import veriloggen as vg
from veriloggen.optimizer import try_optimize as optimize

import nngen.basic_types as bt
from . import basic
from . import conv2d


def to_shape_2d(shape):
    if bt.get_rank(shape) == 1:
        return tuple([1, shape[0]])

    return shape


def to_conv2d_act_shape(shape):
    bat = 1
    if bt.get_rank(shape) == 1:
        row = 1
    else:
        row = functools.reduce(lambda x, y: x * y, shape[:-1], 1)
    col = 1
    ch = shape[-1]

    return (bat, row, col, ch)


def to_conv2d_weight_shape(shape):
    # expect transposed data
    if bt.get_rank(shape) == 1:
        och = 1
    else:
        och = functools.reduce(lambda x, y: x * y, shape[:-1], 1)
    row = 1
    col = 1
    ich = shape[-1]

    return (och, row, col, ich)


class matmul(conv2d.conv2d):
    input_chainable = False
    output_chainable = False

    @property
    def par_left_col(self):
        return self.par_ich

    @par_left_col.setter
    def par_left_col(self, par_left_col):
        self.par_ich = par_left_col

    @property
    def par_left_row(self):
        return self.par_col

    @par_left_row.setter
    def par_left_row(self, par_left_row):
        self.par_col = par_left_row

    @property
    def par_out_col(self):
        return self.par_och

    @par_out_col.setter
    def par_out_col(self, par_out_col):
        self.par_och = par_out_col

    @property
    def concur_out_col(self):
        return self.concur_och

    @concur_out_col.setter
    def concur_out_col(self, concur_out_col):
        self.concur_och = concur_out_col

    @property
    def left_ram_size(self):
        return self.input_ram_size

    @left_ram_size.setter
    def left_ram_size(self, left_ram_size):
        self.input_ram_size = left_ram_size

    @property
    def right_ram_size(self):
        return self.filter_ram_size

    @right_ram_size.setter
    def right_ram_size(self, right_ram_size):
        self.filter_ram_size = right_ram_size

    def __sub_str__(self):
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

        par_left_col = ' par_left_col:%d' % self.par_left_col if self.par_left_col > 1 else ''
        par_left_row = ' par_left_row:%d' % self.par_left_row if self.par_left_row > 1 else ''
        par_out_col = ' par_out_col:%d' % self.par_out_col if self.par_out_col > 1 else ''
        if hasattr(self, 'concur_och_value'):
            concur_out_col = ' concur_out_col:%d' % self.concur_och_value
        else:
            concur_out_col = ' concur_out_col:%s' % str(self.concur_out_col)
        stationary = ' stationary:%s' % (
            'right' if self.stationary == 'filter' else 'left')

        left_ram_size = (' left_ram_size:%d' % self.input_ram_size
                         if self.left_ram_size is not None else '')
        right_ram_size = (' right_ram_size:%d' % self.filter_ram_size
                          if self.right_ram_size is not None else '')
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
            keep_left = ' keep_left' if self.keep_input_value else ''
        else:
            keep_left = ''
        if hasattr(self, 'keep_filter_value'):
            keep_right = ' keep_right' if self.keep_filter_value else ''
        else:
            keep_right = ''

        ret = [str(s)
               for s in (bias, scale,
                         vshamt_mul, vshamt_sum, vshamt_out,
                         cshamt_mul, cshamt_sum, cshamt_out,
                         act_func, mul_dtype, sum_dtype,
                         par_left_col, par_left_row, par_out_col,
                         concur_out_col, stationary,
                         left_ram_size, right_ram_size,
                         bias_ram_size, scale_ram_size,
                         vshamt_mul_ram_size, vshamt_sum_ram_size, vshamt_out_ram_size,
                         out_ram_size,
                         keep_left, keep_right)]
        return ''.join(ret)

    def __init__(self, a, b,
                 bias=None, scale=None,
                 transposed_a=False, transposed_b=False,
                 rshift_mul=None, rshift_sum=None, rshift_out=None,
                 act_func=None,
                 dtype=None, mul_dtype=None, sum_dtype=None,
                 name=None,
                 par_left_col=1, par_left_row=1, par_out_col=1,
                 concur_out_col=None, stationary='right',
                 left_ram_size=None, right_ram_size=None,
                 bias_ram_size=None, scale_ram_size=None,
                 vshamt_mul_ram_size=None, vshamt_sum_ram_size=None, vshamt_out_ram_size=None,
                 out_ram_size=None,
                 disable_keep_left=False):

        if transposed_a:
            perm = list(range(bt.get_rank(to_shape_2d(a.shape))))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            a = basic.transpose(a, perm=perm)

        if not transposed_b:
            # matrix B must be transposed for fast computation
            perm = list(range(bt.get_rank(to_shape_2d(b.shape))))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            b = basic.transpose(b, perm=perm)

        self.transposed_a = transposed_a
        self.transposed_b = transposed_b

        input_shape = to_conv2d_act_shape(a.shape)
        input = a

        filter_shape = to_conv2d_weight_shape(b.shape)
        filter = b

        strides = (1, 1, 1, 1)
        padding = 'SAME'
        out_shape = tuple(list(to_shape_2d(a.shape)[:-2]) +
                          [to_shape_2d(a.shape)[-2], to_shape_2d(b.shape)[-2]])

        if stationary == 'right':
            stationary = 'filter'
        elif stationary == 'left':
            stationary = 'input'
        else:
            raise ValueError("stationary must be 'left' or 'right'")

        conv2d.conv2d.__init__(self, input, filter, strides,
                               bias, scale,
                               rshift_mul, rshift_sum, rshift_out,
                               act_func, padding,
                               dtype, mul_dtype, sum_dtype,
                               name,
                               par_left_col, par_out_col, par_left_row, 1,
                               concur_out_col, stationary,
                               left_ram_size, right_ram_size,
                               bias_ram_size, scale_ram_size,
                               vshamt_mul_ram_size, vshamt_sum_ram_size, vshamt_out_ram_size,
                               out_ram_size,
                               disable_keep_left,
                               input_shape, filter_shape, out_shape)

    def attribute(self, cshamt_mul=None, cshamt_sum=None, cshamt_out=None,
                  par_left_col=None, par_out_col=None, par_left_row=None,
                  concur_out_col=None,
                  stationary=None,
                  left_ram_size=None, right_ram_size=None,
                  bias_ram_size=None, scale_ram_size=None,
                  vshamt_mul_ram_size=None, vshamt_sum_ram_size=None, vshamt_out_ram_size=None,
                  out_ram_size=None,
                  par_ich=None, par_och=None, par_col=None, par_row=None,
                  concur_och=None,
                  input_ram_size=None, filter_ram_size=None,
                  disable_keep_left=None):

        if stationary is None:
            pass
        elif stationary == 'filter' or stationary == 'input':
            pass
        elif stationary == 'right':
            stationary = 'filter'
        elif stationary == 'left':
            stationary = 'input'
        else:
            raise ValueError("stationary must be 'left' or 'right'")

        if par_left_col is None:
            par_left_col = par_ich

        if par_out_col is None:
            par_out_col = par_och

        if par_left_row is None:
            par_left_row = par_col

        conv2d.conv2d.attribute(self, cshamt_mul, cshamt_sum, cshamt_out,
                                par_left_col, par_out_col, par_left_row, None,
                                concur_out_col, stationary,
                                left_ram_size, right_ram_size,
                                bias_ram_size, scale_ram_size,
                                vshamt_mul_ram_size, vshamt_sum_ram_size, vshamt_out_ram_size,
                                out_ram_size,
                                disable_keep_left)

    def eval(self, memo, input_dict, **kwargs):
        if id(self) in memo:
            return memo[id(self)]

        import nngen.verify as verify

        name = self.__class__.__name__

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
        kwargs['transposed_a'] = self.transposed_a
        kwargs['transposed_b'] = self.transposed_b
        kwargs['rshift_mul'] = rshift_mul
        kwargs['rshift_sum'] = rshift_sum
        kwargs['rshift_out'] = rshift_out
        kwargs['act_func'] = self.act_func
        kwargs['dtype'] = self.dtype
        kwargs['mul_dtype'] = self.mul_dtype
        kwargs['sum_dtype'] = self.sum_dtype
        kwargs['name'] = self.name
        kwargs['par_left_col'] = self.par_left_col
        kwargs['par_left_row'] = self.par_left_row
        kwargs['par_out_col'] = self.par_out_col
        kwargs['concur_out_col'] = self.concur_out_col
        kwargs['stationary'] = self.stationary
        kwargs['a_dtype'] = self.args[0].dtype
        kwargs['b_dtype'] = self.args[1].dtype
        kwargs['bias_dtype'] = self.args[self.args_dict['bias']].dtype if self.has_bias else None
        kwargs['scale_dtype'] = self.args[self.args_dict['scale']].dtype if self.has_scale else None

        method = self.get_eval_method()
        ret = method(input, filter, **kwargs)
        memo[id(self)] = ret

        return ret
