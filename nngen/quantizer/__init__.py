from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import nngen.basic_types as bt

from . import conv2d
from . import matmul
from . import normalize


# describe custom quantize methods here
func_map = {
    'conv2d': conv2d.conv2d,
    'matmul': matmul.matmul,
    'normalize': normalize.normalize,
    'scaled_add': normalize.scaled_add,
    'scaled_concat': normalize.scaled_concat,
}


def _has_func(op_type):
    return op_type in func_map


def _get_func(op_type):
    if op_type not in func_map:
        return None

    return func_map[op_type]


class _QuantizeVisitor(object):

    def __init__(self, value_ranges, num_trials=5):
        self.value_ranges = value_ranges
        self.num_trials = num_trials

    def generic_visit(self, node):

        if not isinstance(node, bt._Operator):
            return

        for arg in node.args:
            self.visit(arg)

        if len(node.args) > 0:
            node.scale_factor = node.args[0].scale_factor

    def visit(self, node):

        if isinstance(node, (int, float, bool, str)):
            return

        if isinstance(node, bt._Storage):
            return

        if node.quantized:
            return

        op_type = node.__class__.__name__

        if not _has_func(op_type) and isinstance(node, bt._Operator):
            self.generic_visit(node)
            node.quantized = True
            return

        node_func = _get_func(op_type)

        if node_func is None:
            raise NotImplementedError()

        node_func(self, node)
        node.quantized = True


def quantize(outputs,
             value_ranges=None, num_trials=1):
    """
    Quantize pre-trained weights and determine right-shift amounts

    Parameters
    ----------

    outputs : list
        Output NNgen nodes

    value_ranges : dict
        numerical range dictionary of (min, max) tuples by name

    num_trials : int
        number of sampling trials to determine right-shift amounts
    """

    if isinstance(outputs, dict):
        outputs = outputs.values()

    if value_ranges is None:
        value_ranges = {}

    visitor = _QuantizeVisitor(value_ranges, num_trials=num_trials)

    for output in outputs:
        visitor.visit(output)
