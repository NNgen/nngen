from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math

import nngen.basic_types as bt


class leaky_relu_base(bt._ElementwiseOperator):
    input_chainable = True
    output_chainable = True
    slope = None
    rshift = None

    def __init__(self, features, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, features,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        if id(self) in memo:
            return memo[id(self)]

        import nngen.verify as verify

        name = 'leaky_relu'
        method = getattr(verify, name, None)

        args = [arg.eval(memo, input_dict)
                for arg in self.args]

        kwargs['slope'] = self.slope
        kwargs['rshift'] = self.rshift
        kwargs['dtype'] = self.dtype
        kwargs['name'] = self.name
        kwargs['par'] = self.par

        ret = method(*args, **kwargs)
        memo[id(self)] = ret

        return ret


leaky_relu_cache = {}


def get_leaky_relu_op(slope, rshift=None, dtype=None):

    if rshift is None:
        rshift = dtype.width - 1 if dtype is not None else 31

    if (slope, rshift) in leaky_relu_cache:
        return leaky_relu_cache[(slope, rshift)]

    @staticmethod
    def op(strm, *args, **kwargs):
        mul = strm.Mul(args[0], slope)
        slope_width = int(math.ceil(math.log(slope, 2)) + 1) + 1  # signed
        dtype_width = dtype.width if dtype is not None else 32
        mul.width = slope_width + dtype_width
        neg = strm.Sra(mul, rshift)
        neg.width = dtype_width
        return strm.Mux(args[0] > strm.Int(0), args[0], neg)

    cls = type(
        'leaky_relu_%d_%d' % (slope, rshift),
        (leaky_relu_base,),
        {'op': op,
         'slope': slope,
         'rshift': rshift}
    )

    leaky_relu_cache[(slope, rshift)] = cls

    return cls


def leaky_relu(features, slope, rshift=None, dtype=None, name=None, par=1):

    op = get_leaky_relu_op(slope, rshift, dtype)

    return op(features, dtype, name, par)
