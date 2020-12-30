from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
from collections import OrderedDict

import veriloggen as vg
from veriloggen.optimizer import try_optimize as optimize

import nngen.basic_types as bt
import nngen.util as util


class add(bt._ElementwiseOperator):
    """
    Returns :math:`x + y` element-wise.

    Args:
        x: A tensor.
        y: A tensor.
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """

    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.Add(*args)

    def __init__(self, x, y, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x, y,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        kwargs['y_dtype'] = self.args[1].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class sub(bt._ElementwiseOperator):
    """
    Returns :math:`x - y` element-wise.

    Args:
        x: A tensor.
        y: A tensor.
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """

    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.Sub(*args)

    def __init__(self, x, y, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x, y,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        kwargs['y_dtype'] = self.args[1].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class neg(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.Uminus(*args)

    def __init__(self, x, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class abs(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.Abs(*args)

    def __init__(self, x, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class zeros_imm(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    def op(self, strm, *args, **kwargs):
        return strm.ReinterpretCast(strm.Int(0),
                                    width=self.dtype.width,
                                    point=self.dtype.point,
                                    signed=self.dtype.signed)

    def __init__(self, shape, dtype=None, name=None, par=1):
        bt._ElementwiseOperator.__init__(self,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['shape'] = self.shape
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


def zeros_imm_like(x, dtype=None, name=None, par=1):
    shape = x.shape
    return zeros_imm(shape, dtype, name, par)


class ones_imm(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    def op(self, strm, *args, **kwargs):
        if self.dtype.point > 0:
            return strm.ReinterpretCast(strm.Int(1) << self.dtype.point,
                                        width=self.dtype.width,
                                        point=self.dtype.point,
                                        signed=self.dtype.signed)
        if self.dtype.point < 0:
            return strm.ReinterpretCast(strm.Int(1) >> self.dtype.point,
                                        width=self.dtype.width,
                                        point=self.dtype.point,
                                        signed=self.dtype.signed)

        return strm.ReinterpretCast(strm.Int(1),
                                    width=self.dtype.width,
                                    point=self.dtype.point,
                                    signed=self.dtype.signed)

    def __init__(self, shape, dtype=None, name=None, par=1):
        bt._ElementwiseOperator.__init__(self,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['shape'] = self.shape
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


def ones_imm_like(x, dtype=None, name=None, par=1):
    shape = x.shape
    return ones_imm(shape, dtype, name, par)


class full_imm(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    def __sub_str__(self):
        return ' fill_value:%d' % self.fill_value

    def get_local_control_param_values(self):
        return OrderedDict([('fill_value_cparam', self.fill_value)])

    def op(self, strm, *args, **kwargs):
        if self.dtype.point > 0:
            return strm.ReinterpretCast(self.fill_value_cparam << self.dtype.point,
                                        width=self.dtype.width,
                                        point=self.dtype.point,
                                        signed=self.dtype.signed)
        if self.dtype.point < 0:
            return strm.ReinterpretCast(self.fill_value_cparam >> self.dtype.point,
                                        width=self.dtype.width,
                                        point=self.dtype.point,
                                        signed=self.dtype.signed)

        return strm.ReinterpretCast(self.fill_value_cparam,
                                    width=self.dtype.width,
                                    point=self.dtype.point,
                                    signed=self.dtype.signed)

    def __init__(self, shape, fill_value, dtype=None, name=None, par=1):
        self.fill_value = fill_value
        bt._ElementwiseOperator.__init__(self,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['shape'] = self.shape
        kwargs['fill_value'] = self.fill_value
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


def full_imm_like(x, fill_value, dtype=None, name=None, par=1):
    shape = x.shape
    return full_imm(shape, fill_value, dtype, name, par)


class equal(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.Eq(*args)

    def __init__(self, x, y, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x, y,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        kwargs['y_dtype'] = self.args[1].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class not_equal(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.NotEq(*args)

    def __init__(self, x, y, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x, y,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        kwargs['y_dtype'] = self.args[1].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class less(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.LessThan(*args)

    def __init__(self, x, y, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x, y,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        kwargs['y_dtype'] = self.args[1].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class less_equal(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.LessEq(*args)

    def __init__(self, x, y, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x, y,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        kwargs['y_dtype'] = self.args[1].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class greater(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.GreaterThan(*args)

    def __init__(self, x, y, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x, y,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        kwargs['y_dtype'] = self.args[1].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class greater_equal(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.GreaterEq(*args)

    def __init__(self, x, y, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x, y,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        kwargs['y_dtype'] = self.args[1].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class sign_binary(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.Mux(args[0] > strm.Int(0), strm.Int(1), strm.Int(-1))

    def __init__(self, x, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class sign_ternary(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.Mux(args[0] > strm.Int(0), strm.Int(1),
                        strm.Mux(args[0] == strm.Int(0), strm.Int(0),
                                 strm.Int(-1)))

    def __init__(self, x, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class where(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.Mux(args[0], args[1], args[2])

    def __init__(self, condition, x, y, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, condition, x, y,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['condition_dtype'] = self.args[0].dtype
        kwargs['x_dtype'] = self.args[1].dtype
        kwargs['y_dtype'] = self.args[2].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class add_n(bt._ElementwiseOperator):
    """
    Adds all input tensors element-wise.

    Args:
        arg: A list of Tensor objects.
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """

    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.AddN(*args)

    def __init__(self, arg, dtype=None, name=None, par=1):
        if not isinstance(arg, (tuple, list)):
            raise TypeError('expected tuple or list')
        shape = None
        bt._ElementwiseOperator.__init__(self, *arg,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        if id(self) in memo:
            return memo[id(self)]

        import nngen.verify as verify

        name = self.__class__.__name__

        args = [arg.eval(memo, input_dict)
                for arg in self.args]

        arg_dtypes = [arg.dtype for arg in self.args]
        kwargs['arg_dtypes'] = arg_dtypes
        kwargs['dtype'] = self.dtype
        kwargs['name'] = self.name
        kwargs['par'] = self.par

        method = self.get_eval_method()
        ret = method(args, **kwargs)
        memo[id(self)] = ret

        return ret


class lshift(bt._ElementwiseOperator):
    """
    Elementwise computes the bitwise left-shift of x and y.

    Args:
        x: A tensor.
        y: A tensor.
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """

    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.Sll(*args)

    def __init__(self, x, y, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x, y,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        kwargs['y_dtype'] = self.args[1].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class rshift(bt._ElementwiseOperator):
    """
    Elementwise computes the bitwise right-shift of x and y.

    Args:
        x: A tensor.
        y: A tensor.
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """

    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.Sra(*args)

    def __init__(self, x, y, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x, y,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        kwargs['y_dtype'] = self.args[1].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class rshift_round(bt._ElementwiseOperator):
    """
    Elementwise computes the bitwise right-shift of x and y with rounding.

    Args:
        x: A tensor.
        y: A tensor.
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """

    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.SraRound(*args)

    def __init__(self, x, y, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x, y,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        kwargs['y_dtype'] = self.args[1].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class clip(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    def op(self, strm, *args, **kwargs):
        width = self.dtype.width

        if self.dtype.signed:
            p_th = (1 << (width - 1)) - 1
            if self.asymmetric_clip:
                n_th = -1 * p_th - 1
            else:
                n_th = -1 * p_th
        else:
            p_th = (1 << width) - 1
            n_th = 0

        p = strm.Mux(args[0] > p_th, p_th, args[0])
        n = strm.Mux(args[0] < n_th, n_th, args[0])

        return strm.Mux(args[0] >= 0, p, n)

    def __init__(self, x, asymmetric_clip=False,
                 dtype=None, name=None, par=1):

        shape = None
        bt._ElementwiseOperator.__init__(self, x,
                                         dtype=dtype, shape=shape, name=name, par=par)
        self.asymmetric_clip = asymmetric_clip

    def eval(self, memo, input_dict, **kwargs):
        kwargs['asymmetric_clip'] = self.asymmetric_clip
        kwargs['x_dtype'] = self.args[0].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class multiply(bt._ElementwiseOperator):
    """
    Returns :math:`x * y` element-wise.

    Args:
        x: A tensor.
        y: A tensor.
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """

    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.Times(*args)

    def __init__(self, x, y, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x, y,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        kwargs['y_dtype'] = self.args[1].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class multiply_shared(multiply):
    input_chainable = True
    output_chainable = True

    def op(self, strm, *args, **kwargs):
        index = kwargs['index']
        sub = strm.substream(self.substreams[index])
        sub.to_source('x', args[0])
        sub.to_source('y', args[1])
        return sub.from_sink('z')

    def get_required_substreams(self):
        substrms = []
        substrms.extend(bt._ElementwiseOperator.get_required_substreams(self))

        x_datawidth = self.args[0].get_op_width()
        x_point = self.args[0].get_op_point()
        x_signed = self.args[0].get_signed()
        y_datawidth = self.args[1].get_op_width()
        y_point = self.args[1].get_op_point()
        y_signed = self.args[1].get_signed()
        mul_width = self.get_op_width()
        mul_point = self.get_op_point()
        mul_signed = self.get_signed()
        substrms.extend([('mul',
                          (x_datawidth, x_point, x_signed,
                           y_datawidth, y_point, y_signed,
                           mul_width, mul_point, mul_signed))] * self.par)

        return substrms


class div(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    def op(self, strm, *args, **kwargs):
        index = kwargs['index']
        sub = strm.substream(self.substreams[index])
        sub.to_source('x', args[0])
        sub.to_source('y', args[1])
        return sub.from_sink('z')

    def get_required_substreams(self):
        substrms = []
        substrms.extend(bt._ElementwiseOperator.get_required_substreams(self))

        x_datawidth = self.args[0].get_op_width()
        x_point = self.args[0].get_op_point()
        x_signed = self.args[0].get_signed()
        y_datawidth = self.args[1].get_op_width()
        y_point = self.args[1].get_op_point()
        y_signed = self.args[1].get_signed()
        div_width = self.get_op_width()
        div_point = self.get_op_point()
        div_signed = self.get_signed()

        substrms.extend([('div',
                          (x_datawidth, x_point, x_signed,
                           y_datawidth, y_point, y_signed,
                           div_width, div_point, div_signed))] * self.par)

        return substrms

    def __init__(self, x, y, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, x, y,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        kwargs['y_dtype'] = self.args[1].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class multiply_add_rshift_clip(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True

    def op(self, strm, *args, **kwargs):
        x_datawidth = self.args[0].get_op_width()
        x_point = self.args[0].get_op_point()
        x_signed = self.args[0].get_signed()
        y_datawidth = self.args[1].get_op_width()
        y_point = self.args[1].get_op_point()
        y_signed = self.args[1].get_signed()
        z_datawidth = self.args[2].get_op_width()
        z_point = self.args[2].get_op_point()
        z_signed = self.args[2].get_signed()

        madd = strm.Madd(args[0], args[1], args[2])

        if self.sum_dtype is not None:
            madd.width = self.sum_dtype.width
            madd.signed = self.sum_dtype.signed
        else:
            madd.width = max(x_datawidth + y_datawidth, z_datawidth)
            madd.signed = self.dtype.signed

        if self.sum_dtype is not None and madd.point != self.sum_dtype.point:
            madd = strm.Cast(madd, point=self.sum_dtype.point)

        sra = strm.Sra(madd, args[3])

        width = self.dtype.width

        p_th = (1 << (width - 1)) - 1
        if self.asymmetric_clip:
            n_th = -1 * p_th - 1
        else:
            n_th = -1 * p_th

        p = strm.Mux(sra > p_th, p_th, sra)
        n = strm.Mux(sra < n_th, n_th, sra)

        return strm.Mux(sra >= 0, p, n)

    def __init__(self, x, y, z, shamt, asymmetric_clip=False,
                 dtype=None, sum_dtype=None, name=None, par=1):

        shape = None
        bt._ElementwiseOperator.__init__(self, x, y, z, shamt,
                                         dtype=dtype, shape=shape, name=name, par=par)
        self.asymmetric_clip = asymmetric_clip
        self.sum_dtype = sum_dtype

    def eval(self, memo, input_dict, **kwargs):
        kwargs['asymmetric_clip'] = self.asymmetric_clip
        kwargs['x_dtype'] = self.args[0].dtype
        kwargs['y_dtype'] = self.args[1].dtype
        kwargs['z_dtype'] = self.args[2].dtype
        kwargs['shamt_dtype'] = self.args[3].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class _reduce_op(bt._ReductionOperator):

    input_chainable = True
    output_chainable = False

    def __init__(self, input_tensor,
                 axis=None, keep_dims=False, dtype=None, name=None, par=1):

        rank = bt.get_rank(input_tensor.shape)
        axis = util.to_axis(axis, rank)
        shape = util.to_reduce_shape(input_tensor.shape, axis, keep_dims)

        bt._ReductionOperator.__init__(self, input_tensor,
                                       dtype=dtype, shape=shape, name=name,
                                       axis=axis, keep_dims=keep_dims, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['input_tensor_dtype'] = self.args[0].dtype
        return bt._ReductionOperator.eval(self, memo, input_dict, **kwargs)


class reduce_sum(_reduce_op):
    """
    Computes the sum of elements across dimensions of a tensor.

    Args:
        input_tensor: A tensor.
        axis: The dimensions to reduce (optional).
        keep_dims: If true, retains reduced dimensions with length 1 (optional).
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """

    @staticmethod
    def reduce_op(strm, value, size, par_offset, **kwargs):
        data, valid = strm.ReduceAddValid(value, size, **kwargs)
        return (data,), valid

    @staticmethod
    def carry_op(strm, *args, **kwargs):
        return strm.Add(*args)


class reduce_max(_reduce_op):
    """
    Returns the maximum value of elements across dimensions of a tensor.

    Args:
        input_tensor: A tensor.
        axis: The dimensions to reduce (optional).
        keep_dims: If true, retains reduced dimensions with length 1 (optional).
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """

    def reduce_op(self, strm, value, size, par_offset, **kwargs):
        kwargs['initval'] = self.carry_default_values[0]
        data, valid = strm.ReduceMaxValid(value, size, **kwargs)
        return (data,), valid

    @staticmethod
    def carry_op(strm, *args, **kwargs):
        return strm.Max(*args)

    def __init__(self, input_tensor,
                 dtype=None, shape=None, name=None,
                 axis=None, keep_dims=False, par=1):

        _reduce_op.__init__(self, input_tensor,
                            axis=axis, keep_dims=keep_dims, dtype=dtype, name=name, par=par)

        if self.get_signed():
            point = self.get_op_point()
            if point > 0:
                default_value = -1 * ((2 ** (self.dtype.width - 1)) >> point)
            else:
                default_value = -1 * ((2 ** (self.dtype.width - 1)) << -point)
        else:
            default_value = 0

        self.carry_default_values = [default_value]


class reduce_min(_reduce_op):
    """
    Returns the minimum value of elements across dimensions of a tensor.

    Args:
        input_tensor: A tensor.
        axis: The dimensions to reduce (optional).
        keep_dims: If true, retains reduced dimensions with length 1 (optional).
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """

    def reduce_op(self, strm, value, size, par_offset, **kwargs):
        kwargs['initval'] = self.carry_default_values[0]
        data, valid = strm.ReduceMinValid(value, size, **kwargs)
        return (data,), valid

    @staticmethod
    def carry_op(strm, *args, **kwargs):
        return strm.Min(*args)

    def __init__(self, input_tensor,
                 dtype=None, shape=None, name=None,
                 axis=None, keep_dims=False, par=1):

        _reduce_op.__init__(self, input_tensor,
                            axis=axis, keep_dims=keep_dims, dtype=dtype, name=name, par=par)

        if self.get_signed():
            point = self.get_op_point()
            if point > 0:
                default_value = (2 ** (self.dtype.width - 1) - 1) >> point
            else:
                default_value = (2 ** (self.dtype.width - 1) - 1) << -point
        else:
            if point > 0:
                default_value = (2 ** self.dtype.width - 1) >> point
            else:
                default_value = (2 ** self.dtype.width - 1) << -point

        self.carry_default_values = (default_value,)


class _arg_op(_reduce_op):

    carry_default_values = (0, 0)

    def __init__(self, input_tensor,
                 axis=None, keep_dims=False, dtype=None, name=None, par=1):

        if isinstance(axis, (tuple, list)) and len(axis) > 1:
            raise ValueError('size of axis must be 1.')

        # Because get_stream_reduce_output() cannot calculate
        # the flatten index value for a unflatten value,
        # input value must be flatten when axis is None.
        if axis is None and len(input_tensor.shape) > 1:
            input_tensor = reshape(input_tensor, [-1])

        _reduce_op.__init__(self, input_tensor,
                            axis=axis, keep_dims=keep_dims, dtype=dtype, par=par)


class argmax(_arg_op):
    """
    Returns the index of the maximum value of elements across dimensions of a tensor.

    Args:
        input_tensor: A tensor.
        axis: The dimensions to reduce (optional).
        keep_dims: If true, retains reduced dimensions with length 1 (optional).
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """

    def reduce_op(self, strm, value, size, par_offset, **kwargs):
        kwargs['initval'] = self.carry_default_values[1]
        index, data, valid = strm.ReduceArgMaxValid(value, size, **kwargs)
        shamt = math.ceil(math.log2(self.par))
        index = index << shamt
        index.latency = 0
        index = index + par_offset
        index.latency = 0
        return (index, data), valid

    def gather_op(self, strm, indexes, values, **kwargs):
        ret_index = indexes[0]
        ret_value = values[0]
        for index, value in zip(indexes[1:], values[1:]):
            ret_index = strm.Mux(ret_value < value, index, ret_index)
            ret_index.latency = 0
            ret_value = strm.Mux(ret_value < value, value, ret_value)
            ret_value.latency = 0
        return ret_index, ret_value

    def __init__(self, input_tensor,
                 axis=None, keep_dims=False, dtype=None, name=None, par=1):

        _arg_op.__init__(self, input_tensor,
                         axis=axis, keep_dims=keep_dims, dtype=dtype, name=name, par=par)

        if self.get_signed():
            point = self.get_op_point()
            if point > 0:
                default_value = -1 * ((2 ** (self.dtype.width - 1)) >> point)
            else:
                default_value = -1 * ((2 ** (self.dtype.width - 1)) << -point)
        else:
            default_value = 0

        self.carry_default_values = (0, default_value)


class argmin(_arg_op):
    """
    Returns the index of the minimum value of elements across dimensions of a tensor.

    Args:
        input_tensor: A tensor.
        axis: The dimensions to reduce (optional).
        keep_dims: If true, retains reduced dimensions with length 1 (optional).
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """

    def reduce_op(self, strm, value, size, par_offset, **kwargs):
        kwargs['initval'] = self.carry_default_values[1]
        index, data, valid = strm.ReduceArgMinValid(value, size, **kwargs)
        shamt = math.ceil(math.log2(self.par))
        index = index << shamt
        index.latency = 0
        index = index + par_offset
        index.latency = 0
        return (index, data), valid

    def gather_op(self, strm, indexes, values, **kwargs):
        ret_index = indexes[0]
        ret_value = values[0]
        for index, value in zip(indexes[1:], values[1:]):
            ret_index = strm.Mux(ret_value < value, ret_index, index)
            ret_index.latency = 0
            ret_value = strm.Mux(ret_value < value, ret_value, value)
            ret_value.latency = 0
        return ret_index, ret_value

    def __init__(self, input_tensor,
                 axis=None, keep_dims=False, dtype=None, name=None, par=1):

        _arg_op.__init__(self, input_tensor,
                         axis=axis, keep_dims=keep_dims, dtype=dtype, name=name, par=par)

        if self.get_signed():
            point = self.get_op_point()
            if point > 0:
                default_value = (2 ** (self.dtype.width - 1) - 1) >> point
            else:
                default_value = (2 ** (self.dtype.width - 1) - 1) << -point
        else:
            if point > 0:
                default_value = (2 ** self.dtype.width - 1) >> point
            else:
                default_value = (2 ** self.dtype.width - 1) << -point

        self.carry_default_values = (0, default_value)


class _reshape(bt._Reshape):
    pass


class _lazy_reshape(bt._LazyReshape):
    pass


def reshape(tensor, shape, dtype=None, name=None):
    """
    Reshapes a tensor.

    Given tensor, this operation returns a tensor \
    that has the same values as tensor with shape shape.

    Args:
        tensor: A tensor.
        shape: A tensor. Defines the shape of the output tensor.
        dtype: Output data type (optional).
        name: A name for the operation (optional).
    """
    return _lazy_reshape(tensor, shape, dtype, name)


def cast(x, dtype, name=None):
    if x.dtype.wordsize == dtype.wordsize:
        return bt._View(x, x.shape, dtype, name)

    return _reshape(x, x.shape, dtype, name)


def expand_dims(input, axis, name=None):
    """
    Inserts a dimension of 1 into a tensor's shape.

    Given a tensor input, this operation inserts a dimension \
    of 1 at the dimension index axis of input's shape. \
    The dimension index axis starts at zero; \
    if you specify a negative number for axis it is counted backward from the end.

    Args:
        input: A Tensor.
    """
    shape = input.shape
    rank = bt.get_rank(shape)
    axis = util.to_axis(axis, rank)[0]

    new_shape = []
    new_shape.extend(shape[:axis])
    new_shape.append(1)
    new_shape.extend(shape[axis:])

    if axis == rank:
        return _lazy_reshape(input, new_shape)

    return bt._View(input, shape=new_shape, name=name)


class transpose(bt._Operator):
    """
    Transposes a. Permutes the dimensions according to perm.

    Args:
        a: A Tensor.
        perm: A permutation of the dimensions of a.
        dtype: Output data type (optional).
        name: A name for the operation (optional).
    """

    input_chainable = False
    output_chainable = False
    thread_cachable = False

    def __sub_str__(self):
        perm = ' perm:%s' % str(tuple(self.transpose_perm))
        onnx_perm = (' onnx_perm:%s' % str(tuple(self.transpose_onnx_perm))
                     if self.transpose_onnx_perm is not None else '')
        return '%s%s' % (perm, onnx_perm)

    def __init__(self, a, perm=None, dtype=None, name=None):

        if bt.get_rank(a.shape) == 1:
            a_shape = tuple([1, a.shape[0]])
        else:
            a_shape = a.shape

        if perm is None:
            perm = tuple(reversed(range(bt.get_rank(a_shape))))

        self.transpose_perm = perm
        self.transpose_onnx_perm = None

        shape = []
        for p in perm:
            shape.append(a_shape[p])
        shape = tuple(shape)

        bt._Operator.__init__(self, a,
                              dtype=dtype, shape=shape, name=name)
        self.implicit = False

    def attribute(self):
        pass

    def get_required_rams(self):
        # burst read, scatter write
        min_size = self.args[0].shape[-1]
        input_width = self.args[0].get_ram_width()
        output_width = self.get_ram_width()

        inputs = [(input_width, min_size)]
        outputs = [(output_width, min_size)]
        temps = []
        return inputs, outputs, temps

    def get_stream_func(self):
        return None

    def get_control_param_values(self):
        return OrderedDict()

    def control_sequence(self, fsm):
        arg = self.args[0]
        ram = self.input_rams[0]

        shape = self.get_aligned_shape()
        arg_shape = arg.get_aligned_shape()

        # burst read, scatter write
        write_order = list(reversed([self.transpose_perm.index(i)
                                     for i in range(len(shape))]))
        write_pattern = bt.shape_to_pattern(shape, write_order)

        read_offset = self.m.TmpReg(self.maxi.addrwidth, initval=0)
        write_offsets = [self.m.TmpReg(self.maxi.addrwidth, initval=0)
                         for _ in write_pattern]
        write_all_offset = self.objaddr
        for write_offset in write_offsets:
            write_all_offset += write_offset

        read_counts = [self.m.TmpReg(self.maxi.addrwidth, initval=0)
                       for _ in write_pattern]

        # initialize
        fsm(
            read_offset(0),
            [write_offset(0) for write_offset in write_offsets],
            [read_count(0) for read_count in read_counts]
        )
        fsm.goto_next()

        # DMA read
        read_state = fsm.current

        laddr = 0
        gaddr = self.arg_objaddrs[0] + read_offset
        read_size = arg_shape[-1]

        bt.bus_lock(self.maxi, fsm)
        bt.dma_read(self.maxi, fsm, ram, laddr, gaddr, read_size)
        bt.bus_unlock(self.maxi, fsm)

        # read-modify-write
        modify_state = fsm.current

        laddr = read_counts[0]
        gaddr = write_all_offset

        bt.read_modify_write(self.m, fsm, self.maxi,
                             ram, self.output_rams[0],
                             laddr, gaddr)

        prev_done = 1
        for (read_count, maxval,
             write_offset,
             (out_size, out_stride)) in zip(read_counts, reversed(arg_shape),
                                            write_offsets,
                                            write_pattern):
            fsm.If(prev_done)(
                read_count.inc(),
                write_offset.add(
                    optimize(bt.to_byte(out_stride * self.get_ram_width())))
            )
            fsm.If(prev_done, read_count == maxval - 1)(
                read_count(0),
                write_offset(0)
            )
            prev_done = vg.Ands(prev_done, (read_count == maxval - 1))

        fsm.If(laddr == read_size - 1)(
            read_offset.add(
                optimize(bt.to_byte(read_size * arg.get_ram_width())))
        )
        fsm.If(laddr < read_size - 1).goto(modify_state)
        fsm.If(laddr == read_size - 1).goto(read_state)
        fsm.If(prev_done).goto_next()

    def get_layout(self):
        if self.layout is not None:
            return self.layout

        orig_layout = self.args[0].get_layout()
        if orig_layout is None:
            return None

        new_layout = [l for l in orig_layout]
        for i, p in enumerate(self.transpose_perm):
            new_layout[i] = orig_layout[p]
        return ''.join(new_layout)

    def get_onnx_layout(self):
        if self.onnx_layout is not None:
            return self.onnx_layout

        orig_onnx_layout = self.args[0].get_onnx_layout()
        if orig_onnx_layout is None:
            return None

        new_onnx_layout = [l for l in orig_onnx_layout]
        for i, p in enumerate(self.transpose_onnx_perm):
            new_onnx_layout[i] = orig_onnx_layout[p]
        return ''.join(new_onnx_layout)

    def eval(self, memo, input_dict, **kwargs):
        if id(self) in memo:
            return memo[id(self)]

        import nngen.verify as verify

        name = self.__class__.__name__

        args = [arg.eval(memo, input_dict)
                for arg in self.args]

        kwargs['perm'] = self.transpose_perm
        kwargs['dtype'] = self.dtype
        kwargs['name'] = self.name

        method = self.get_eval_method()
        ret = method(*args, **kwargs)
        memo[id(self)] = ret

        return ret
