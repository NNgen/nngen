from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

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

    def __init__(self, num_samples=1, value_ranges=None, input_generators=None, verbose=False):
        if value_ranges is None:
            value_ranges = {}

        if input_generators is None:
            input_generators = {}

        self.num_samples = num_samples
        self.value_ranges = value_ranges
        self.input_generators = input_generators
        self.verbose = verbose

        self.memo = {}
        self.input_dict = {}

    def generic_visit(self, node):

        if not isinstance(node, bt._Operator):
            return

        for arg in node.args:
            self.visit(arg)

        if len(node.args) > 0:
            node.scale_factor = node.args[0].scale_factor

    def visit(self, node):

        if isinstance(node, (int, float, bool, str)):
            self._verbose_node(node)
            return

        if isinstance(node, bt._Storage):
            if node.value is not None:
                self._verbose_node(node)
                return

            name = node.name

            if name in self.input_dict:
                self._verbose_node(node)
                return

            if name in self.input_generators:
                generator = self.input_generators[name]
                value = generator(node, self.num_samples)
                self.input_dict[name] = value
                self._verbose_node(node)
                return

            value = generate_uniform(node, self.num_samples, self.value_ranges)
            self.input_dict[name] = value
            self._verbose_node(node)
            return

        if node.quantized:
            self._verbose_node(node)
            return

        op_type = node.__class__.__name__

        if not _has_func(op_type) and isinstance(node, bt._Operator):
            self.generic_visit(node)
            node.quantized = True
            self._verbose_node(node)
            return

        node_func = _get_func(op_type)

        if node_func is None:
            raise NotImplementedError()

        node_func(self, node)
        node.quantized = True
        self._verbose_node(node)

    def _verbose_node(self, node):
        if self.verbose:
            print('[quantize] {}'.format(str(node)))


def generate_uniform(node, num_samples, value_ranges):
    """
    Generate a dummy ndarray by uniform random values to determine shift-amounts.

    Parameters
    ----------

    node : nngen.placeholder
        Target placeholder object to make a dummy ndarray.

    num_samples : int
        Number of sampling trials to determine the right-shift amount.
        This value is multiplied to the batch size of the placeholder.

    value_ranges : dict
        Numerical range dictionary of (min, max) tuples by name.
    """

    shape = list(node.shape)

    # enlarge the batch size by num_samples for shift-amount evaluation
    shape[0] *= num_samples
    shape = tuple(shape)

    length = np.multiply.reduce(shape)

    if node.name in value_ranges:
        min_val, max_val = value_ranges[node.name]
    elif node.dtype.signed:
        max_val = 2.0 ** (node.dtype.width - 1) - 1.0
        min_val = -1.0 * max_val
    else:
        max_val = 2.0 ** node.dtype.width - 1.0
        min_val = 0.0

    v = np.random.uniform(min_val, max_val, size=length).reshape(shape)
    v = np.round(v).astype(np.int64)

    return v


def quantize(outputs,
             num_samples=1, value_ranges=None, input_generators=None, verbose=False):
    """
    Quantize pre-trained weights and determine right-shift amounts

    Parameters
    ----------

    outputs : list
        Output NNgen nodes.

    num_samples : int
        Number of sampling trials to determine the right-shift amount.
        This value is multiplied to the batch size of the placeholder.

    value_ranges : dict
        Numerical range dictionary of (min, max) tuples by name.

    input_generators : dict
        Dictionary by name for random sample value generator methods of placeholders.
        The methods must have arguments for node and num_samples.

    verbose : bool
        If True, quantization steps are printed out.
    """

    if isinstance(outputs, dict):
        outputs = outputs.values()

    if input_generators is None:
        input_generators = {}

    visitor = _QuantizeVisitor(num_samples, value_ranges, input_generators, verbose)

    for output in outputs:
        visitor.visit(output)
