from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import collections

import nngen.storage as storage
import nngen.dtype_list as dtype_list

from . import util
from . import basic
from . import exp
from . import conv
from . import gemm
from . import pool
from . import pad
from . import act_func
from . import batchnormalization
from . import shape
from . import reshape
from . import flatten
from . import upsample
from . import transpose
from . import concat
from . import squeeze
from . import gather
from . import slice_
from . import cast
from . import ceil
from . import floor
from . import identity


# describe custom ONNX converting methods here
func_map = {
    'Add': basic.Add,
    'Sub': basic.Sub,
    'Mul': basic.Mul,
    'Div': basic.Div,
    'Exp': exp.Exp,
    'Conv': conv.Conv,
    'Gemm': gemm.Gemm,
    'AveragePool': pool.AveragePool,
    'GlobalAveragePool': pool.GlobalAveragePool,
    'MaxPool': pool.MaxPool,
    'Pad': pad.Pad,
    'Relu': act_func.Relu,
    'LeakyRelu': act_func.LeakyRelu,
    'Sigmoid': act_func.Sigmoid,
    'BatchNormalization': batchnormalization.BatchNormalization,
    'Shape': shape.Shape,
    'Reshape': reshape.Reshape,
    'Flatten': flatten.Flatten,
    'Upsample': upsample.Upsample,
    'Transpose': transpose.Transpose,
    'Concat': concat.Concat,
    'Squeeze': squeeze.Squeeze,
    'Unsqueeze': squeeze.Unsqueeze,
    'Gather': gather.Gather,
    'Slice': slice_.Slice,
    'Cast': cast.Cast,
    'Ceil': ceil.Ceil,
    'Floor': floor.Floor,
    'Identity': identity.Identity,
}


def _get_func(op_type):
    return func_map[op_type]


class _OperatorVisitor(object):

    def __init__(self, model,
                 placeholders, variables, constants, operators,
                 producers, consumers,
                 value_dtypes,
                 default_placeholder_dtype, default_variable_dtype,
                 default_constant_dtype, default_operator_dtype,
                 default_scale_dtype, default_bias_dtype,
                 onnx_input_layout=('N', 'C', 'H', 'W'),
                 onnx_filter_layout=('O', 'I', 'H', 'W'),
                 nngen_input_layout=('N', 'H', 'W', 'C'),
                 nngen_filter_layout=('O', 'H', 'W', 'I'),
                 disable_fusion=False, verbose=False):

        self.model = model

        self.placeholders = placeholders
        self.variables = variables
        self.constants = constants
        self.operators = operators

        self.producers = producers
        self.consumers = consumers

        self.value_dtypes = value_dtypes

        self.default_placeholder_dtype = default_placeholder_dtype
        self.default_variable_dtype = default_variable_dtype
        self.default_constant_dtype = default_constant_dtype
        self.default_operator_dtype = default_operator_dtype
        self.default_scale_dtype = default_scale_dtype
        self.default_bias_dtype = default_bias_dtype

        self.onnx_input_layout = onnx_input_layout
        self.onnx_filter_layout = onnx_filter_layout

        self.nngen_input_layout = nngen_input_layout
        self.nngen_filter_layout = nngen_filter_layout

        self.disable_fusion = disable_fusion

        self.verbose = verbose

    def _verbose_node(self, name, node_func, ret):
        if self.verbose:
            print('[onnx] {} {} -> {} {}'.format(name, node_func, type(ret), str(ret)))
        return ret

    def visit(self, name):
        if name in self.placeholders:
            # return self.placeholders[name]
            return self._verbose_node(name, None, self.placeholders[name])

        if name in self.variables:
            # return self.variables[name]
            return self._verbose_node(name, None, self.variables[name])

        if name in self.constants:
            # return self.constants[name]
            return self._verbose_node(name, None, self.constants[name])

        if name in self.operators:
            # return self.operators[name]
            return self._verbose_node(name, None, self.operators[name])

        node = util.search_node_from_model(self.model, name)

        node_func = _get_func(node.op_type)
        node_op = node_func(self, node)

        output_names = util.get_output_names(node)
        for output_name in output_names:
            self.operators[output_name] = node_op

        # return node_op
        return self._verbose_node(name, node_func, node_op)


def from_onnx(filename,
              value_dtypes=None,
              value_shapes=None,
              default_placeholder_dtype=dtype_list.int32,
              default_variable_dtype=dtype_list.int32,
              default_constant_dtype=dtype_list.int32,
              default_operator_dtype=dtype_list.int32,
              default_scale_dtype=dtype_list.int32,
              default_bias_dtype=dtype_list.int32,
              onnx_input_layout=('N', 'C', 'H', 'W'),
              onnx_filter_layout=('O', 'I', 'H', 'W'),
              disable_fusion=False, verbose=False):
    """
    Convert ONNX model to NNgen model

    Parameters
    ----------
    filename : str
        File name of ONNX model

    value_dtypes : dict
        dtype_info dictionary by name

    value_shapes : dict
        shape dictionary for undefined node shapes by name

    default_placeholder_dtype : nngen.dtype_info
        Default dtype for placeholder

    default_variable_dtype : nngen.dtype_info
        Default dtype for variable

    default_constant_dtype : nngen.dtype_info
        Default dtype for constant

    default_operator_dtype : nngen.dtype_info
        Default dtype for operator

    default_scale_dtype : nngen.dtype_info
        Default dtype for scale

    default_bias_dtype : nngen.dtype_info
        Default dtype for bias

    onnx_input_layout : str
        Layout of ONNX input values

    onnx_filter_layout : str
        Layout of ONNX filter (weight) values

    disable_fusion : bool
        Disable operator fusion

    Returns
    -------
    outputs : collections.OrderedDict
        Dict of output values

    placeholders : collections.OrderedDict
        Dictionary of placeholders

    variables : collections.OrderedDict
        Dictionary of variables

    constants : collections.OrderedDict
        Dictionary of constants

    operators : collections.OrderedDict
        Dictionary of operators
    """

    try:
        import onnx
        from onnx import numpy_helper
    except:
        raise ImportError('onnx is required.')

    if value_dtypes is None:
        value_dtypes = {}

    if value_shapes is None:
        value_shapes = {}

    # load model
    model = onnx.load(filename)

    # input/output node dict
    input_nodes = collections.OrderedDict()
    output_nodes = collections.OrderedDict()

    for input_var in model.graph.input:
        input_nodes[input_var.name] = input_var

    for output_var in model.graph.output:
        output_nodes[output_var.name] = output_var

    # variable ndarray dict
    variable_values = collections.OrderedDict()

    for weight in model.graph.initializer:
        name = weight.name
        np_weight = numpy_helper.to_array(weight)
        variable_values[name] = np_weight

    # constant ndarray dict
    constant_values = collections.OrderedDict()

    for node in model.graph.node:
        if node.op_type == 'Constant':
            value = numpy_helper.to_array(node.attribute[0].t)
            output_names = util.get_output_names(node)
            for output_name in output_names:
                constant_values[output_name] = value

    # placeholders
    placeholders = _to_placeholders(input_nodes, output_nodes,
                                    variable_values, constant_values,
                                    value_dtypes, value_shapes,
                                    default_placeholder_dtype,
                                    default_variable_dtype,
                                    default_constant_dtype,
                                    default_operator_dtype)

    # variables
    variables = _to_variables(input_nodes, output_nodes,
                              variable_values, constant_values,
                              value_dtypes, value_shapes,
                              default_placeholder_dtype,
                              default_variable_dtype,
                              default_constant_dtype,
                              default_operator_dtype)

    # constants
    constants = _to_constants(input_nodes, output_nodes,
                              variable_values, constant_values,
                              value_dtypes, value_shapes,
                              default_placeholder_dtype,
                              default_variable_dtype,
                              default_constant_dtype,
                              default_operator_dtype)

    # producer/consumer table
    producers = collections.defaultdict(list)
    consumers = collections.defaultdict(list)

    for node in model.graph.node:
        output_names = util.get_output_names(node)
        for output_name in output_names:
            for arg in node.input:
                if arg not in producers[output_name]:
                    producers[output_name].append(arg)
                if output_name not in consumers[arg]:
                    consumers[arg].append(output_name)

    # operators
    operators = collections.OrderedDict()
    visitor = _OperatorVisitor(model,
                               placeholders, variables, constants, operators,
                               producers, consumers,
                               value_dtypes,
                               default_placeholder_dtype, default_variable_dtype,
                               default_constant_dtype, default_operator_dtype,
                               default_scale_dtype, default_bias_dtype,
                               onnx_input_layout, onnx_filter_layout,
                               disable_fusion=disable_fusion, verbose=verbose)

    placeholders = visitor.placeholders
    variables = visitor.variables
    constants = visitor.constants
    operators = visitor.operators

    for name, output_node in output_nodes.items():
        visitor.visit(name)

    # outputs
    outputs = collections.OrderedDict()

    for name, node in output_nodes.items():
        if name in operators:
            outputs[name] = operators[name]
        elif name in placeholders:
            outputs[name] = placeholders[name]
        elif name in variables:
            outputs[name] = variables[name]
        elif name in constants:
            outputs[name] = constants[name]

    return outputs, placeholders, variables, constants, operators


def _to_placeholders(input_nodes, output_nodes, variable_values, constant_values,
                     value_dtypes, value_shapes,
                     default_placeholder_dtype, default_variable_dtype,
                     default_constant_dtype, default_operator_dtype):

    placeholders = collections.OrderedDict()

    for name, node in input_nodes.items():
        # exclude variables
        if name in variable_values:
            continue

        if name in value_dtypes:
            dtype = value_dtypes[name]
        else:
            dtype = default_placeholder_dtype

        shape = util.to_shape(node, value_shapes)
        p = storage.placeholder(dtype=dtype, shape=shape, name=name)
        placeholders[name] = p

    return placeholders


def _to_variables(input_nodes, output_nodes, variable_values, constant_values,
                  value_dtypes, value_shapes,
                  default_placeholder_dtype, default_variable_dtype,
                  default_constant_dtype, default_operator_dtype):

    variables = collections.OrderedDict()

    for name, node in variable_values.items():
        if name in value_dtypes:
            dtype = value_dtypes[name]
        else:
            dtype = default_variable_dtype

        shape = node.shape
        v = storage.variable(dtype=dtype, shape=shape, name=name)
        v.set_value(node)
        variables[name] = v

    return variables


def _to_constants(input_nodes, output_nodes, variable_values, constant_values,
                  value_dtypes, value_shapes,
                  default_placeholder_dtype, default_variable_dtype,
                  default_constant_dtype, default_operator_dtype):

    constants = collections.OrderedDict()

    for name, node in constant_values.items():
        if name in value_dtypes:
            dtype = value_dtypes[name]
        else:
            dtype = default_constant_dtype

        shape = node.shape
        c = storage.constant(value=node,
                             dtype=dtype, shape=shape, name=name)
        constants[name] = c

    return constants
