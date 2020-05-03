from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from nngen.operator.pool import _pool


class pad(_pool):

    def __sub_str__(self):
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

        return (' padding:%s%s%s%s' %
                (padding, par, value_ram_size, out_ram_size))

    def __init__(self, value, padding,
                 dtype=None, name=None, par=1,
                 value_ram_size=None, out_ram_size=None):

        ksize = (1, 1, 1, 1)
        strides = (1, 1, 1, 1)
        _pool.__init__(self, value, ksize, strides,
                       padding, dtype, name, par,
                       value_ram_size, out_ram_size)

    def get_pad_value(self, strm):
        return strm.Int(0)

    def pool_op(self, strm, index, *vars):
        return vars[0]

    def eval(self, memo, input_dict, **kwargs):
        if id(self) in memo:
            return memo[id(self)]

        import nngen.verify as verify

        name = self.__class__.__name__

        args = [arg.eval(memo, input_dict)
                for arg in self.args]

        value = args[0]

        kwargs['padding'] = self.padding
        kwargs['dtype'] = self.dtype
        kwargs['name'] = self.name
        kwargs['par'] = self.par

        method = self.get_eval_method()
        ret = method(value, **kwargs)
        memo[id(self)] = ret

        return ret
