from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import nngen.basic_types as bt


class relu(bt._ElementwiseOperator):
    """
    Applies the rectified linear unit function element-wise:

    .. math::
        {ReLU}(x) = \max(0, x)

    Args:
        features: A Tensor.
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """

    input_chainable = True
    output_chainable = True

    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.Mux(args[0] > strm.Int(0), args[0], strm.Int(0))

    def __init__(self, features, dtype=None, name=None, par=1):
        shape = None
        bt._ElementwiseOperator.__init__(self, features,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['features_dtype'] = self.args[0].dtype
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)


class relu6(relu):
    """
    Applies the element-wise function:

    .. math::
        {ReLU6}(x) = \min(\max(0, x), 6)

    Args:
        features: A Tensor.
        dtype: Output data type (optional).
        name: A name for the operation (optional).
        par: The number of parallel operations (optional).
    """
    
    @staticmethod
    def op(strm, *args, **kwargs):
        return strm.Mux(args[0] > strm.Int(0),
                        strm.Mux(args[0] > strm.Int(6), strm.Int(6), args[0]),
                        strm.Int(0))
