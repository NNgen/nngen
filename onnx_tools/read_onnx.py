from __future__ import absolute_import
from __future__ import print_function

# --------------------------------------
# from ONNX
# --------------------------------------

import sys
import collections

import onnx
from onnx import numpy_helper

#onnx_model = onnx.load("vgg11_cifar10.onnx")
filename = sys.argv[1]
onnx_model = onnx.load(filename)

# print(onnx.helper.printable_graph(onnx_model.graph))

# Node (Operator)
for i, node in enumerate(onnx_model.graph.node):
    print("[Node #{}]".format(i))
    print(node)

# Input
for i, input in enumerate(onnx_model.graph.input):
    print("[Input #{}]".format(i))
    print(input)

# Output
for i, output in enumerate(onnx_model.graph.output):
    print("[Output #{}]".format(i))
    print(output)


def get_name(obj):
    name = obj.name
    if name == '':  # if isinstance(obj, onnx.FOO.BAR):
        name = '_'.join([output for output in obj.output])
    return name


inputs = collections.OrderedDict()
outputs = collections.OrderedDict()

op_inputs = collections.defaultdict(list)
op_outputs = collections.OrderedDict()

input_values = collections.OrderedDict()

for input_var in onnx_model.graph.input:
    inputs[input_var.name] = input_var

for output_var in onnx_model.graph.output:
    outputs[output_var.name] = output_var

# act shape: (y, x, ch, bat)
# nngen act shape: (bat, y, x, ch)

# weight shape: (y, x, ich, och)
# nngen weight shape: (och, y, x, ich)

weights = onnx_model.graph.initializer

for weight in weights:
    name = weight.name
    inputs[name] = weight
    np_weight = numpy_helper.to_array(weight)
    input_values[name] = np_weight

weight_inputs = collections.OrderedDict([(name, node) for name, node in inputs.items()
                                         if name in input_values])
user_inputs = collections.OrderedDict([(name, node) for name, node in inputs.items()
                                       if name not in input_values])


for node in onnx_model.graph.node:
    if node.op_type == 'Constant':
        name = get_name(node)
        inputs[name] = node
        value = numpy_helper.to_array(node.attribute[0].t)
        input_values[name] = value
    else:
        for dst in list(node.output):
            op_outputs[dst] = node

for node in onnx_model.graph.node:
    name = get_name(node)
    for src in list(node.input):
        if src in inputs:
            op_inputs[name].append(inputs[src])
        else:
            op_inputs[name].append(op_outputs[src])

# for name, node in weight_inputs.items():
#    print(node.name)
#
#    if hasattr(node, 'type'):
#        print('# type: {}'.format(node.type.tensor_type.elem_type))
#        for i, d in enumerate(node.type.tensor_type.shape.dim):
#            print('# dim {}: {}'.format(i, d.dim_value))

breakpoint()

print('# end')
