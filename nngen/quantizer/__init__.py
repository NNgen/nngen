from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import nngen.basic_types as bt

from . import conv2d
from . import matmul
from . import normalize
from . import sigmoid
from . import exp


# describe custom quantize methods here
func_map = {
    'conv2d': conv2d.conv2d,
    'matmul': matmul.matmul,
    'normalize': normalize.normalize,
    'scaled_add': normalize.scaled_add,
    'scaled_concat': normalize.scaled_concat,
    'sigmoid': sigmoid.sigmoid,
    'exp': exp.exp,
}


def _has_func(op_type):
    return op_type in func_map


def _get_func(op_type):
    if op_type not in func_map:
        return None

    return func_map[op_type]


class _QuantizeVisitor(object):

    def __init__(self, input_scale_factors,
                 input_means=None, input_stds=None, input_generators=None, num_samples=1,
                 verbose=False):

        if not isinstance(input_scale_factors, dict):
            raise TypeError('input_scale_factors must be dict.')

        if input_means is None:
            input_means = {}

        if input_stds is None:
            input_stds = {}

        if input_generators is None:
            input_generators = {}

        self.input_scale_factors = input_scale_factors
        self.input_means = input_means
        self.input_stds = input_stds
        self.input_generators = input_generators
        self.num_samples = num_samples

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
                # values are already assigned
                self._verbose_node(node)
                return

            if name in self.input_generators:
                generator = self.input_generators[name]
                value = generator(node, self.num_samples)
                self.input_dict[name] = value

                if name not in self.input_scale_factors:
                    raise ValueError("scale_factor of '%s' not found." % name)
                node.scale_factor = self.input_scale_factors[name]

                self._verbose_node(node)
                return

            value = generate_samples(node, self.input_scale_factors,
                                     self.input_means, self.input_stds, self.num_samples)
            self.input_dict[name] = value

            if name not in self.input_scale_factors:
                raise ValueError("scale_factor of '%s' not found." % name)
            node.scale_factor = self.input_scale_factors[name]

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


def generate_samples(node, input_scale_factors, input_means, input_stds, num_samples):
    """
    Generate a dummy ndarray of random values to determine shift-amounts.

    Parameters
    ----------

    node : nngen.placeholder
        Target placeholder object to make a dummy ndarray.

    input_scale_factors: dict
        Input scale factor dictionary by name.

    input_means : dict
        Input mean dictionary by name

    input_stds : dict
        Input standard deviation dictionary by name

    num_samples : int
        Number of sampling trials to determine the right-shift amount.
        This value is multiplied to the batch size of the placeholder.
    """

    shape = list(node.shape)

    # enlarge the batch size by num_samples for shift-amount evaluation
    shape[0] *= num_samples
    shape = tuple(shape)
    length = np.multiply.reduce(shape)

    if node.name not in input_means:
        mean = np.array([0.0] * shape[-1]).astype(np.float32)
    else:
        mean = np.array(input_means[node.name]).astype(np.float32)

    if node.name not in input_stds:
        if node.name not in input_scale_factors:
            raise ValueError("scale_factor of '%s' not found." % node.name)
        scale_factor = input_scale_factors[node.name] * 1.0
        std = np.array([1.0] * shape[-1]).astype(np.float32) * scale_factor / 3.0
    else:
        std = np.array(input_stds[node.name]).astype(np.float32)

    #p_shape = list(shape[:])
    #for i in range(len(p_shape)):
    #    if i != 0 and i != len(p_shape) - 1:
    #        p_shape[i] = 1
    #p_mean = mean * np.random.uniform(0.8, 1.2, np.multiply.reduce(p_shape)).reshape(p_shape)
    #p_std = std * np.random.uniform(0.8, 1.2, np.multiply.reduce(p_shape)).reshape(p_shape)

    #v = np.random.normal(size=length).reshape(shape) * std + mean

    width = np.sqrt(12.0) * std
    v = np.random.uniform(-0.5, 0.5, size=length).reshape(shape) * width + mean
    v = np.round(v).astype(np.int64)

    if node.dtype.signed:
        max_val = 2 ** (node.dtype.width - 1) - 1
        min_val = -1 * max_val
    else:
        max_val = 2 ** node.dtype.width - 1
        min_val = 0

    v = np.clip(v, min_val, max_val)

    return v


def quantize(outputs, input_scale_factors,
             input_means=None, input_stds=None, input_generators=None, num_samples=1,
             verbose=False):
    """
    Quantize pre-trained weights and determine right-shift amounts

    Parameters
    ----------

    outputs : list
        Output NNgen nodes.

    input_scale_factors: dict
        Input scale factor dictionary by name.

    input_means : dict
        Input mean dictionary by name

    input_stds : dict
        Input standard deviation dictionary by name

    input_generators : dict
        Dictionary by name for random sample value generator methods of placeholders.
        The methods must have arguments for node and num_samples.

    num_samples : int
        Number of sampling trials to determine the right-shift amount.
        This value is multiplied to the batch size of the placeholder.

    verbose : bool
        If True, quantization steps are printed out.
    """

    if isinstance(outputs, dict):
        outputs = outputs.values()

    if input_generators is None:
        input_generators = {}

    visitor = _QuantizeVisitor(input_scale_factors,
                               input_means, input_stds, input_generators, num_samples,
                               verbose)

    for output in outputs:
        visitor.visit(output)
