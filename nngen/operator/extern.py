from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import OrderedDict

import nngen.basic_types as bt
import nngen.verilog as verilog


class extern(bt._Operator):
    input_chainable = False
    output_chainable = False
    parallel_scheduling_allowed = False

    def __sub_str__(self):
        opcode = ' opcode:%d' % self.opcode
        return opcode

    def __init__(self, values, opcode,
                 shape=None, dtype=None, name=None,
                 func=None):

        if not isinstance(values, (tuple, list)):
            values = [values]

        if not isinstance(opcode, int):
            raise TypeError("opcode must be int, not '%s'." %
                            str(type(opcode)))

        if opcode <= 0:
            raise ValueError('opcode must be greater than 0.')

        if shape is None:
            shape = values[0].shape

        if dtype is None:
            dtype = values[0].dtype

        bt._Operator.__init__(self, *values,
                              dtype=dtype, shape=shape, name=name)

        self.opcode = opcode

        # for verify
        self.func = func

    def attribute(self):
        pass

    def get_required_rams(self):
        """
        @return 3 tuples of (width, length)
        """

        inputs = ()
        outputs = ()
        temps = ()
        return inputs, outputs, temps

    def get_stream_func(self):
        return None

    def get_stream_hash(self):
        clsinfo = [type(self)]
        clsinfo = tuple(clsinfo)
        return (clsinfo,)

    def get_control_param_values(self):
        opcode_value = self.opcode

        return OrderedDict([('opcode_value', opcode_value)])

    def control_sequence(self, fsm):
        # send to SW
        self.saxi.write(fsm, verilog.control_reg_extern_send,
                        self.opcode_value)
        # receive from SW
        self.saxi.wait_flag(fsm, verilog.control_reg_extern_recv, 0,
                            resetvalue=0, polarity=False)
        # reset opcode
        self.saxi.write(fsm, verilog.control_reg_extern_send, 0)

    def eval(self, memo, input_dict, **kwargs):
        if id(self) in memo:
            return memo[id(self)]

        import nngen.verify as verify

        name = self.__class__.__name__
        method = getattr(verify, name, None)

        args = [arg.eval(memo, input_dict)
                for arg in self.args]

        values = args
        opcode = self.opcode
        kwargs['shape'] = self.shape
        kwargs['dtype'] = self.dtype
        kwargs['name'] = self.name
        kwargs['value_dtypes'] = [arg.dtype for arg in self.args]
        kwargs['func'] = self.func

        ret = method(values, opcode, **kwargs)
        memo[id(self)] = ret

        return ret
