from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import functools
import numpy as np

import veriloggen.types.axi as axi

from . import basic_types as bt
from . import storage as st


def to_axis(axis, rank):
    if axis is None:
        return None

    if not isinstance(axis, (tuple, list)):
        axis = (axis,)

    new_axis = []
    for i in axis:
        if i < 0:
            i += rank
        new_axis.append(i)

    return tuple(new_axis)


def to_reduce_shape(input_shape, axis=None, keep_dims=False):
    if not keep_dims:
        if axis is None:
            shape = tuple([1])

        else:
            shape = []
            for i, s in enumerate(input_shape):
                if i not in axis:
                    shape.append(s)

            if len(shape) == 0:
                shape.append(1)

            shape = tuple(shape)

    else:
        if axis is None:
            shape = []
            for s in input_shape:
                shape.append(1)

            shape = tuple(shape)

        else:
            shape = []
            for i, s in enumerate(input_shape):
                if i not in axis:
                    shape.append(s)
                else:
                    shape.append(1)

            shape = tuple(shape)

    return shape


def pix_size(size, ksize, stride, padding="SAME"):
    if padding == "SAME":
        return int(math.ceil(size / stride))
    elif padding == "VALID":
        return int(math.ceil((size - ksize + 1) / stride))
    raise ValueError("not supported padding type: '%s'" % padding)


def pad_size(size, ksize, stride):
    if size % stride == 0:
        return max(ksize - stride, 0)
    return max(ksize - (size % stride), 0)


def pad_size_split(size, ksize, stride):
    pad = pad_size(size, ksize, stride)
    a = pad // 2
    b = pad - a
    return pad, a, b


def to_storage_dict(*args, **kwargs):
    d = {}

    for key, arg in kwargs.items():
        if not bt.is_storage(arg) and not arg.is_output:
            raise ValueError("'%s' is not input/output storage." % str(arg))

        while bt.is_view(arg) or bt.is_removable_reshape(arg):
            arg = arg.args[0]

        d[key] = arg

    for arg in args:
        if not bt.is_storage(arg) and not arg.is_output:
            raise ValueError("'%s' is not input/output storage." % str(arg))

        while bt.is_view(arg) or bt.is_removable_reshape(arg):
            arg = arg.args[0]

        d[arg.name] = arg

    return d


def export_ndarray(objs, chunk_size=64):

    if not isinstance(objs, (list, tuple)):
        objs = [objs]

    objs = _collect_numerics(objs)

    variables = []
    for obj in objs:
        if isinstance(obj, st.variable):
            variables.append(obj)

    constants = []
    for obj in objs:
        if isinstance(obj, st.constant):
            constants.append(obj)

    return make_ndarray(variables, constants, chunk_size)


def _collect_numerics(objs):
    new_objs = []
    for obj in objs:
        new_objs.extend(obj.collect_numerics())

    ret = sorted(set(new_objs), key=new_objs.index)
    ret.sort(key=lambda x: x.object_id)
    return ret


def make_ndarray(variables, constants, chunk_size=64):
    min_addr = 2 ** 64
    max_addr = 0

    if isinstance(variables, dict):
        variables = list(variables.values())

    if isinstance(constants, dict):
        constants = list(constants.values())

    for variable in variables:
        if not isinstance(variable, st.variable):
            raise TypeError("'%s' is not variable.'" % str(variable))

        if variable.maxi is None:
            continue

        min_addr = min(min_addr, variable.addr)
        size = aligned_size(variable.memory_size, chunk_size)
        max_addr = max(max_addr, variable.addr + size)

    for constant in constants:
        if not isinstance(constant, st.constant):
            raise TypeError("'%s' is not _Constant.'" % str(constant))

        if constant.maxi is None:
            continue

        min_addr = min(min_addr, constant.addr)
        size = aligned_size(constant.memory_size, chunk_size)
        max_addr = max(max_addr, constant.addr + size)

    # return empty
    if max_addr < min_addr:
        param = np.zeros([0], dtype=np.uint8)
        return param

    param = np.zeros([max_addr - min_addr], dtype=np.uint8)
    dst_width = 8

    for variable in variables:
        if variable.maxi is None:
            continue
        if variable.value is None:
            continue

        src_width = variable.dtype.width
        dst_offset = variable.addr - min_addr
        alignment = variable.get_word_alignment()
        axi.set_memory(param, variable.value, dst_width, src_width, dst_offset, alignment)

    for constant in constants:
        if constant.maxi is None:
            continue

        src_width = constant.dtype.width
        dst_offset = constant.addr - min_addr
        alignment = constant.get_word_alignment()
        axi.set_memory(param, constant.value, dst_width, src_width, dst_offset, alignment)

    return param


def aligned_size(size, chunk_size):
    return int(math.ceil(size / chunk_size)) * chunk_size


def is_elementwise_operator(obj):
    return isinstance(obj, bt._ElementwiseOperator)


def clip_threshold(width, signed, asymmetric_clip=False):
    if signed:
        p_th = (1 << (width - 1)) - 1
        if asymmetric_clip:
            n_th = -1 * p_th - 1
        else:
            n_th = -1 * p_th
    else:
        p_th = (1 << width) - 1
        n_th = 0

    return p_th, n_th
