from __future__ import absolute_import
from __future__ import print_function

from . import conv2d


class binary_weight_conv2d(conv2d):

    def get_required_substreams(self):
        if self.mul_dtype is not None:
            mul_width = self.mul_dtype.width
        else:
            mul_width = self.get_op_width()

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
            sum_width = self.get_op_width()

        if self.sum_dtype is not None:
            sum_point = self.sum_dtype.point
        else:
            sum_point = self.get_op_point()

        if self.sum_dtype is not None:
            sum_signed = self.sum_dtype.signed
        else:
            sum_signed = self.get_signed()

        filter_num_col = self.filter_shape[-2]
        filter_num_row = self.filter_shape[-3]
        num_macs = filter_num_col * filter_num_row

        x_datawidth = self.args[0].get_op_width()
        x_point = self.args[0].get_op_point()
        x_signed = self.args[0].get_signed()
        y_datawidth = self.args[1].get_op_width()
        y_point = self.args[1].get_op_point()
        y_signed = self.args[1].get_signed()

        if y_datawidth != 1:
            raise ValueError('binary weight must be 1-bit.')

        if y_point != 0:
            raise ValueError('binary weight must be int, not fixed.')

        if y_signed:
            raise ValueError('binary weight must not be signed.')

        args = (x_datawidth, x_point, x_signed,
                y_datawidth, y_point, y_signed,
                mul_width, mul_point, mul_signed)

        mulname = 'updown_rshift'
        substrms = [(mulname, args)] * (num_macs *
                                        self.par_ich * self.par_och)

        substrms.extend([('add_tree',
                          (sum_width, sum_point, sum_signed,
                           num_macs * self.par_ich))] * self.par_och)
        substrms.extend([('acc_rshift_round_frac',
                          (sum_width, sum_point, sum_signed,
                           sum_width, sum_point, sum_signed))] * self.par_och)
        substrms.extend([('mul_rshift',
                          (sum_width, sum_point, sum_signed,
                           sum_width, sum_point, sum_signed))] * self.par_och)

        return substrms
