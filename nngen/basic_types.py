from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import inspect
import math
import functools
import numpy as np
from collections import OrderedDict, defaultdict

import veriloggen as vg
import veriloggen.thread as vthread
from veriloggen.optimizer import try_optimize as optimize

from . import dtype_list


# Object ID counter for object sorting key
_object_counter = 0


class _Node(object):

    def __init__(self):
        global _object_counter
        self.object_id = _object_counter
        _object_counter += 1


class _Numeric(_Node):
    __intrinsics__ = ('control_sequence',)

    parallel_scheduling_allowed = True

    # for ONNX
    layout = None
    onnx_layout = None

    def __init__(self, dtype=None, shape=None, name=None):
        _Node.__init__(self)

        if shape is None:
            pass
        elif isinstance(shape, tuple):
            pass
        elif isinstance(shape, list):
            shape = tuple(shape)
        elif isinstance(shape, int):
            shape = tuple([shape])
        else:
            raise ValueError("illegal shape type: '%s'" % str(type(shape)))

        # zero-sized shape check
        if shape is not None:
            for s in shape:
                if s == 0:
                    raise ValueError("shape contains '0': %s" % str(shape))

        self.name = name

        self.shape = shape
        self.dtype = dtype

        self.consumers = []
        self.is_output = False
        self.stage = None

        # data alignment
        self.word_alignment = 0

        # hardware signals
        self.m = None
        self.clk = None
        self.rst = None
        self.maxi = None
        self.saxi = None

        # memory access information
        self.global_index = None
        self.local_index = None
        self.default_global_addr = None
        self.default_local_addr = None

        self.objaddr = None  # Reg

        # numpy value for variable/constant
        self.value = None

        # for quantization
        self.quantized = False
        # scale factor for quantization
        self.scale_factor = 1.0
        # remember applied permutation pattern for ONNX
        self.perm = None

    def __str__(self):
        clsname = self.__class__.__name__
        name = self.name if self.name is not None else 'None'
        dtype = self.dtype.to_str() if self.dtype is not None else 'None'
        shape = (str(self.shape)
                 if isinstance(self.shape, (tuple, list)) else '()')
        sub_str = self.__sub_str__()
        default_addr = (self.default_global_addr
                        if self.default_global_addr is not None else 0)
        default_addr += (self.default_local_addr
                         if self.default_local_addr is not None and
                         self.local_index != 0 else 0)
        default_addr = (' default_addr:%d' % default_addr)
        global_index = (' g_index:%d' % self.global_index
                        if self.global_index is not None else '')
        local_index = (' l_index:%d' % self.local_index
                       if self.local_index is not None and
                       self.local_index != 0 else '')
        alignment = (' word_alignment:%d' % self.get_word_alignment()
                     if self.maxi is not None else '')
        aligned_shape = (' aligned_shape:%s' % str(tuple(self.get_aligned_shape()))
                         if isinstance(self.shape, (tuple, list)) and
                         self.maxi is not None else '()')
        layout = (" layout:'%s'" % self.get_layout()
                  if self.get_layout() is not None else '')
        onnx_layout = (" onnx_layout:'%s'" % self.get_onnx_layout()
                       if self.get_onnx_layout() is not None else '')
        scale_factor = ' scale_factor:%f' % self.scale_factor
        return '<%s %s dtype:%s shape:%s%s%s%s%s%s%s%s%s%s>' % (
            clsname, name, dtype, shape, sub_str,
            default_addr, global_index, local_index, alignment, aligned_shape,
            layout, onnx_layout, scale_factor)

    def __sub_str__(self):
        return ''

    def _name(self, postfix):
        name = '_'.join(
            [self.__class__.__name__, str(self.object_id), postfix])
        return name

    @property
    def aligned_shape(self):
        return self.get_aligned_shape()

    @property
    def length(self):
        return self.get_length()

    @property
    def size(self):
        return self.get_length()

    @property
    def aligned_length(self):
        return self.get_aligned_length()

    @property
    def aligned_size(self):
        return self.get_aligned_length()

    @property
    def memory_size(self):
        return int(math.ceil(self.aligned_length * self.dtype.width / 8))

    @property
    def addr(self):
        if self.default_global_addr is None:
            return -1

        if self.default_local_addr is None:
            return -1

        return self.default_global_addr + self.default_local_addr

    def attribute(self):
        raise NotImplementedError()

    def add_consumer(self, obj):
        self.consumers.append(obj)

    def set_output(self):
        self.is_output = True

    def set_stage(self, stage):
        self.stage = stage

    def add_alignment_request(self, num_words):
        if num_words is None:
            return

        if 2 ** int(math.ceil(math.log(num_words, 2))) != num_words:
            raise ValueError('alignment must be power of 2.')

        self.word_alignment = max(self.word_alignment, num_words)

    def set_module_info(self, m, clk, rst, maxi, saxi):
        self.m = m
        self.clk = clk
        self.rst = rst
        self.maxi = maxi
        self.saxi = saxi

    def collect_numerics(self):
        return [self]

    def collect_sources(self):
        return ()

    def is_scheduled(self):
        return self.stage is not None

    def is_schedulable(self, stage):
        return True

    def get_signed(self, default_signed=True):
        if self.dtype is not None:
            return self.dtype.signed
        return default_signed

    def get_length(self):
        return shape_to_length(self.shape)

    def get_default_alignment(self):
        if self.maxi is None:
            raise ValueError("maxi is required to determine alignment.")

        return int(math.ceil(self.maxi.datawidth / self.get_ram_width()))

    def get_word_alignment(self):
        return max(self.get_default_alignment(),
                   self.word_alignment)

    def get_aligned_length(self):
        shape = self.get_aligned_shape()
        return shape_to_length(shape)

    def get_aligned_shape(self):
        if self.maxi is None:
            raise ValueError("maxi is required to determine alignment.")

        num_words = self.get_word_alignment()

        aligned_shape = []
        for s in self.shape[:-1]:
            aligned_shape.append(s)

        res = num_words - self.shape[-1] % num_words

        if res == num_words:
            res = 0

        aligned_shape.append(self.shape[-1] + res)

        return aligned_shape

    def get_op_width(self, default_width=32):
        if self.dtype is None:
            return default_width
        width = self.dtype.width
        return width

    def get_op_point(self, default_point=0):
        if self.dtype is None:
            return default_point
        point = self.dtype.point
        return point

    def get_ram_width(self, default_width=32):
        if self.dtype is None:
            width = default_width
        else:
            width = self.dtype.width

        return 2 ** int(math.ceil(math.log(width, 2)))

    def set_global_index(self, global_index):
        self.global_index = global_index

    def set_local_index(self, local_index):
        self.local_index = local_index

    def set_default_global_addr(self, default_global_addr):
        self.default_global_addr = default_global_addr

    def set_default_local_addr(self, default_local_addr):
        self.default_local_addr = default_local_addr

    def make_objaddr(self):
        self.objaddr = self.m.Reg(self._name('objaddr'),
                                  self.maxi.addrwidth, initval=0)

    def set_value(self, value):
        if isinstance(value, (list, tuple)):
            value = np.array(value)

        if not isinstance(value, np.ndarray):
            raise TypeError('value must be np.ndarray, list, or tuple.')

        value = value.reshape(self.shape)

        self.value = value

    def get_original_shape(self):
        return self.shape

    def get_layout(self):
        return self.layout

    def get_original_layout(self):
        return self.get_layout()

    def get_onnx_layout(self):
        return self.onnx_layout

    def get_original_onnx_layout(self):
        return self.get_onnx_layout()

    @property
    def reversed_perm(self):
        if self.perm is None:
            return None

        rev = []
        for i, _ in enumerate(self.perm):
            rev.append(self.perm.index(i))

        return rev

    def eval(self, memo, input_dict, **kwargs):
        if self.value is None:
            raise ValueError('no value is assigned.')

        return self.value


class _Storage(_Numeric):

    def __init__(self, dtype=None, shape=None, name=None, is_input=False):
        _Numeric.__init__(self, dtype=dtype, shape=shape, name=name)
        self.is_input = is_input

    def eval(self, memo, input_dict, **kwargs):
        if self.name is not None and self.name in input_dict:
            return input_dict[self.name]

        return _Numeric.eval(self, memo, input_dict, **kwargs)


class _Constant(_Storage):

    def __init__(self, value, dtype=None, shape=None, name=None):
        if isinstance(value, (list, tuple)):
            value = np.array(value)

        if not isinstance(value, np.ndarray):
            raise TypeError('value must be np.ndarray, list, or tuple.')

        if shape is None:
            shape = value.shape
        else:
            value = value.reshape(shape)

        _Storage.__init__(self, dtype=dtype, shape=shape, name=name, is_input=False)
        self.value = value


class _Operator(_Numeric):
    input_chainable = False
    output_chainable = False
    chain_head = False
    stream_cachable = True
    control_cachable = True

    control_param_custom_width = {}
    control_param_custom_signed = {}

    shared_attr_names = ()

    def __sub_str__(self):
        par = ' par:%d' % self.par if self.par > 1 else ''
        return par

    def __init__(self, *args, **opts):
        dtype = opts['dtype'] if 'dtype' in opts else None
        shape = opts['shape'] if 'shape' in opts else None
        name = opts['name'] if 'name' in opts else None
        par = opts['par'] if 'par' in opts else 1

        if dtype is None:
            dtype = dtype_list.get_max_dtype(*args)

        if shape is None:
            shape = same_shape(*args)

        if par is not None and 2 ** int(math.ceil(math.log(par, 2))) != par:
            raise ValueError('par must be power of 2.')

        _Numeric.__init__(self, dtype=dtype, shape=shape, name=name)
        self.args = args

        # attribute
        self.par = par
        _Operator.attribute(self, par)

        for arg in self.args:
            arg.add_consumer(self)

        if is_input_chainable_operator(self):
            self.chain_head = True

        for arg in self.args:
            if are_chainable_operators(self, arg):
                arg.chain_head = False
            else:
                arg.output_chainable = False
                arg.chain_head = False

        self.shared_attrs = defaultdict(OrderedDict)

        self.input_rams = None
        self.output_rams = None
        self.temp_rams = None

        self.substreams = ()
        self.stream = None
        self.control = None

        self.arg_objaddrs = None  # list of Reg

        self.control_param_names = ()  # list of str
        self.control_param_index = None  # int
        self.control_param_index_reg = None  # Reg
        self.control_param_ram = None

        self.cached_ram_set = False
        self.cached_stream = False
        self.cached_control = False

    def attribute(self, par=None):
        if par is not None:
            if (par - 1) & par != 0:
                raise ValueError('par must be power of 2.')

            self.par = par

            for arg in self.args:
                arg.add_alignment_request(self.par)

            self.add_alignment_request(self.par)

    def merge_shared_attrs(self, obj):
        """ merge obj's and self's shared_attrs. """

        for attr in self.shared_attr_names:
            v = getattr(obj, attr)

            if v is None:
                self.shared_attrs[attr][None] = None
                continue

            key = v.get_stream_hash()
            if key not in self.shared_attrs[attr]:
                self.shared_attrs[attr][key] = v

        obj.shared_attrs = self.shared_attrs

    def get_shared_attr_index(self, name, obj):
        for index, key in enumerate(self.shared_attrs[name].keys()):
            if key is None and obj is None:
                return index
            if key is None:
                continue
            if obj is None:
                continue
            if key == obj.get_stream_hash():
                return index

        raise ValueError('no such object %s' % str(obj))

    def collect_numerics(self):
        ret = []
        ret.append(self)

        for arg in self.args:
            ret.extend(arg.collect_numerics())

        ret = sorted(set(ret), key=ret.index)
        return ret

    def collect_arg_numerics(self):
        ret = []

        for arg in self.args:
            if are_chainable_operators(self, arg):
                ret.append(arg)
                ret.extend(arg.collect_arg_numerics())

        ret = sorted(set(ret), key=ret.index)
        return ret

    def collect_sources(self):
        ret = []
        for arg in self.args:
            if are_chainable_operators(self, arg):
                ret.extend(arg.collect_sources())
            else:
                ret.append(arg)

        return ret

    def is_schedulable(self, stage):
        for arg in self.args:
            if not arg.is_scheduled():
                return False

            if arg.stage > stage:
                return False

            if (arg.stage == stage and
                    not are_chainable_operators(self, arg)):
                return False

        return True

    def get_required_rams(self):
        """ 
        @return 3 tuples of (width, length)
        """

        shape = self.get_aligned_shape()
        min_size = int(math.ceil(shape[-1] / self.par)) * 2
        input_widths = [arg.get_ram_width() * self.par for arg in self.args]
        output_width = self.get_ram_width() * self.par

        inputs = []
        temps = []

        for arg, width in zip(self.args, input_widths):
            if are_chainable_operators(self, arg):
                arg_input_rams, arg_output_rams, arg_temp_rams = arg.get_required_rams()
                inputs.extend(arg_input_rams)
                temps.extend(arg_temp_rams)
            else:
                arg_ram = [(width, min_size)]
                inputs.extend(arg_ram)

        outputs = [(output_width, min_size)]

        return inputs, outputs, temps

    def set_rams(self, input_rams, output_rams, temp_rams):
        input_index = 0
        temp_index = 0

        for arg in self.args:
            if are_chainable_operators(self, arg):
                arg_input_rams, arg_output_rams, arg_temp_rams = arg.get_required_rams()
                si = input_index
                ei = input_index + len(arg_input_rams)
                st = temp_index
                et = temp_index + len(arg_temp_rams)
                input_set = input_rams[si:ei] if len(input_rams) > 0 else ()
                temp_set = temp_rams[st:et] if len(temp_rams) > 0 else ()
                arg.set_rams(input_set, (), temp_set)
                input_index += len(arg_input_rams)
                temp_index += len(arg_temp_rams)

        self.input_rams = input_rams
        self.output_rams = output_rams
        self.temp_rams = temp_rams

    def check_ram_requirements(self, input_rams, output_rams, temp_rams):
        my_inputs, my_outputs, my_temps = self.get_required_rams()

        if len(my_inputs) > len(input_rams):
            return False

        for (my_input_width, my_input_size), input_ram in zip(my_inputs, input_rams):
            if input_ram.length < my_input_size:
                return False
            if input_ram.datawidth < my_input_width:
                return False

        if len(my_outputs) > len(output_rams):
            return False

        for (my_output_width, my_output_size), output_ram in zip(my_outputs, output_rams):
            if output_ram.length < my_output_size:
                return False
            if output_ram.datawidth < my_output_width:
                return False

        if len(my_temps) > len(temp_rams):
            return False

        for (my_temp_width, my_temp_size), temp_ram in zip(my_temps, temp_rams):
            if temp_ram.length < my_temp_size:
                return False
            if temp_ram.datawidth < my_temp_width:
                return False

        return True

    def get_ram_set_obj_hash(self):
        input_ids = tuple([id(input_ram) for input_ram in self.input_rams])
        output_ids = tuple([id(output_ram) for output_ram in self.output_rams])
        temp_ids = tuple([id(temp_ram) for temp_ram in self.temp_rams])
        return (input_ids, output_ids, temp_ids)

    def get_required_substreams(self):
        """
        @return a tuple of (method_name, args)
        """
        substreams = ()
        return substreams

    def set_substreams(self, substreams):
        self.substreams = substreams

    def make_stream(self, datawidth=32, fsm_as_module=False,
                    dump=False, dump_base=10):

        stream_func = self.get_stream_func()

        if stream_func is None:
            return None

        name = '_'.join(
            ['stream', self.__class__.__name__, str(self.object_id)])
        stream = vthread.Stream(self.m, name, self.clk, self.rst,
                                datawidth=datawidth,
                                fsm_as_module=fsm_as_module,
                                dump=dump, dump_base=dump_base)

        # synthesize stream definition
        stream_func(stream)
        self.set_stream(stream)

        return stream

    def get_stream_func(self):
        """ must be implemented in each _Operator class """
        # def func(strm):
        #    pass
        # return func
        raise NotImplementedError()

    def set_stream(self, stream):
        self.stream = stream

    def get_stream_hash(self):
        clsinfo = [type(self)]

        for arg in self.args:
            if are_chainable_operators(self, arg):
                clsinfo.append(arg.get_stream_hash())
            else:
                clsinfo.append(arg.dtype)

        clsinfo = tuple(clsinfo)
        dtype = self.dtype
        par = self.par
        return (clsinfo, dtype, par)

    def get_stream_obj_hash(self):
        return id(self.stream)

    def get_arg_global_indexes(self):
        sources = self.collect_sources()
        return [source.global_index for source in sources]

    def get_arg_local_indexes(self):
        sources = self.collect_sources()
        return [source.local_index for source in sources]

    def get_arg_default_global_addrs(self):
        sources = self.collect_sources()
        return [source.default_global_addr for source in sources]

    def get_arg_default_local_addrs(self):
        sources = self.collect_sources()
        return [source.default_local_addr for source in sources]

    def set_control_param_index(self, control_param_index):
        self.control_param_index = control_param_index

    def get_control_param_values(self):
        """
        This method must be implemented in each _Operator class.
        For chained _StreamingOperator, this method of the head operator only is called.
        If control params of intermediate operators in the chain are required,
        use get_local_control_param_values()
        """
        # return OrderedDict([('param_name', param_value), ... ])
        raise NotImplementedError()

    def get_local_control_param_values(self):
        # return OrderedDict([('param_name', param_value), ... ])
        return OrderedDict()

    def to_local_control_param_name(self, index, name):
        return '_'.join(['local', str(index), name])

    def collect_local_control_param_values(self, index_offset=0):
        values = OrderedDict()

        for name, lparam in self.get_local_control_param_values().items():
            signame = self.to_local_control_param_name(index_offset, name)
            values[signame] = lparam

        numerics = self.collect_arg_numerics()

        for i, arg in enumerate(numerics):
            if not isinstance(arg, _Operator):
                continue

            for name, lparam in arg.get_local_control_param_values().items():
                signame = self.to_local_control_param_name(index_offset + i + 1, name)
                values[signame] = lparam

        return values

    def collect_all_control_param_values(self):
        ret = OrderedDict()
        ret.update(self.get_control_param_values())
        ret.update(self.collect_local_control_param_values())
        return ret

    def collect_local_control_param_names(self):
        return self.collect_local_control_param_values().keys()

    def collect_all_control_param_names(self):
        ret = []
        ret.extend(self.control_param_names)
        ret.extend(self.collect_local_control_param_names())
        ret = sorted(set(ret), key=ret.index)
        return ret

    def make_arg_objaddrs(self):
        sources = self.collect_sources()
        self.arg_objaddrs = [self.m.Reg(self._name('arg_objaddr_%d' % i),
                                        self.maxi.addrwidth, initval=0)
                             for i, source in enumerate(sources)]

    def make_control_params(self, control_param_len, width_dict, signed_dict,
                            use_param_ram=False, min_param_ram_len=0):

        if control_param_len <= 1:
            self.make_control_params_wire(width_dict, signed_dict)

        elif not use_param_ram or control_param_len < min_param_ram_len:
            self.make_control_params_wire(width_dict, signed_dict)

        else:
            self.make_control_params_reg(width_dict, signed_dict)

        self.copy_local_control_params(self)

    def make_control_params_wire(self, width_dict, signed_dict):
        for name, width in width_dict.items():
            signed = signed_dict[name]

            if isinstance(width, (tuple, list)):
                wires = []
                for i, (w, s) in enumerate(zip(width, signed)):
                    if name in self.control_param_custom_width:
                        if callable(self.control_param_custom_width[name]):
                            w = self.control_param_custom_width[name](self)
                        else:
                            w = self.control_param_custom_width[name]

                    if name in self.control_param_custom_signed:
                        if callable(self.control_param_custom_signed[name]):
                            s = self.control_param_custom_signed[name](self)
                        else:
                            s = self.control_param_custom_signed[name]

                    wire = self.CparamWire(name + ('_%d' % i), w, signed=s)
                    wires.append(wire)

                setattr(self, name, wires)

            else:
                if name in self.control_param_custom_width:
                    if callable(self.control_param_custom_width[name]):
                        width = self.control_param_custom_width[name](self)
                    else:
                        width = self.control_param_custom_width[name]

                if name in self.control_param_custom_signed:
                    if callable(self.control_param_custom_signed[name]):
                        signed = self.control_param_custom_signed[name](self)
                    else:
                        signed = self.control_param_custom_signed[name]

                wire = self.CparamWire(name, width, signed=signed)
                setattr(self, name, wire)

        self.control_param_names = tuple(width_dict.keys())

    def make_control_params_reg(self, width_dict, signed_dict):
        for name, width in width_dict.items():
            signed = signed_dict[name]

            if isinstance(width, (tuple, list)):
                regs = []
                for i, (w, s) in enumerate(zip(width, signed)):
                    if name in self.control_param_custom_width:
                        if callable(self.control_param_custom_width[name]):
                            w = self.control_param_custom_width[name](self)
                        else:
                            w = self.control_param_custom_width[name]

                    if name in self.control_param_custom_signed:
                        if callable(self.control_param_custom_signed[name]):
                            s = self.control_param_custom_signed[name](self)
                        else:
                            s = self.control_param_custom_signed[name]

                    reg = self.Cparam(name + ('_%d' % i), w, initval=0, signed=s)
                    regs.append(reg)

                setattr(self, name, regs)

            else:
                if name in self.control_param_custom_width:
                    if callable(self.control_param_custom_width[name]):
                        width = self.control_param_custom_width[name](self)
                    else:
                        width = self.control_param_custom_width[name]

                if name in self.control_param_custom_signed:
                    if callable(self.control_param_custom_signed[name]):
                        signed = self.control_param_custom_signed[name](self)
                    else:
                        signed = self.control_param_custom_signed[name]

                reg = self.Cparam(name, width, initval=0, signed=signed)
                setattr(self, name, reg)

        self.control_param_names = tuple(width_dict.keys())

    def make_control_param_buf(self, control_param_list,
                               use_param_ram=False, min_param_ram_len=0, ram_style=None):
        if len(control_param_list) <= 1:
            return self.make_control_param_single(control_param_list)

        if not use_param_ram or len(control_param_list) < min_param_ram_len:
            return self.make_control_param_mux(control_param_list)

        return self.make_control_param_ram(control_param_list, ram_style)

    def make_control_param_single(self, control_param_list):
        for name in self.collect_all_control_param_names():
            dst = getattr(self, name)
            value = control_param_list[0][name]
            if isinstance(value, (tuple, list)):
                for d, v in zip(dst, value):
                    d.assign(v)

            else:
                dst.assign(value)

    def make_control_param_mux(self, control_param_list):
        length = len(control_param_list)
        addrwidth = int(math.ceil(math.log(length, 2)))
        self.control_param_index_reg = self.m.Reg(self._name('control_param_index'),
                                                  addrwidth, initval=0)

        pattern_dict = defaultdict(list)
        for i, values in enumerate(control_param_list):
            for name, value in values.items():
                if isinstance(value, (tuple, list)):
                    lst = [(self.control_param_index_reg == i, v) for v in value]
                    pattern_dict[name].append(lst)
                else:
                    pattern_dict[name].append((self.control_param_index_reg == i, value))

        for name in self.collect_all_control_param_names():
            dst = getattr(self, name)
            if isinstance(dst, (tuple, list)):
                for i, d in enumerate(dst):
                    vlist = [lst[i] for lst in pattern_dict[name]]
                    _, val = vlist[-1]
                    vlist[-1] = (None, val)
                    v = vg.PatternMux(*vlist)
                    d.assign(v)

            else:
                _, val = pattern_dict[name][-1]
                pattern_dict[name][-1] = (None, val)
                value = vg.PatternMux(*pattern_dict[name])
                dst.assign(value)

    def make_control_param_ram(self, control_param_list, ram_style=None):

        datawidth = self.get_total_control_param_width()

        initvals = []
        for values in control_param_list:

            res = set(values.keys()) - set(self.collect_all_control_param_names())
            if res:
                raise ValueError(
                    "too much control_param values: %s" % str(res))

            cat_params = []
            for name in self.collect_all_control_param_names():
                value = values[name]
                reg = getattr(self, name)

                if isinstance(reg, (tuple, list)):
                    group_params = []

                    if len(reg) != len(value):
                        raise ValueError(
                            "control_param length mismatch: %d (reg) != %d (value)" %
                            (len(reg), len(value)))

                    for r, v in zip(reg, value):
                        width = (r.width if r.width is not None else 1)
                        if v.bit_length() > width:
                            raise ValueError(
                                'control_param_value is too wide.')

                        group_params.append(vg.Int(int(v), width, base=16))

                    cat_params.append(vg.Cat(*reversed(group_params)))

                else:
                    width = (reg.width if reg.width is not None else 1)
                    if value.bit_length() > width:
                        raise ValueError('control_param_value is too large.')

                    cat_params.append(vg.Int(int(value), width, base=16))

            if not cat_params:
                continue

            ram_value = vg.Cat(*reversed(cat_params))
            initvals.append(ram_value)

        if not initvals:
            self.control_param_ram = None
            return

        ram_name = self._name('control_param_ram')
        length = len(control_param_list)
        if length > 1:
            addrwidth = int(math.ceil(math.log(length, 2)))
        else:
            addrwidth = 1
        numports = 1

        self.control_param_ram = vthread.RAM(self.m, ram_name, self.clk, self.rst,
                                             datawidth, addrwidth,
                                             numports, initvals, nocheck_initvals=True,
                                             ram_style=ram_style)
        for i in range(numports):
            self.control_param_ram.disable_write(i)

    def get_total_control_param_width(self):
        width = 0
        for name in self.collect_all_control_param_names():
            reg = getattr(self, name)
            if isinstance(reg, (tuple, list)):
                for r in reg:
                    width += (r.width if r.width is not None else 1)
            else:
                width += (reg.width if reg.width is not None else 1)
        return width

    def make_control(self, datawidth=None, fsm_as_module=False):

        if datawidth is None:
            datawidth = self.dtype.width

        control_func = self.get_control_func()

        if control_func is None:
            return None

        name = '_'.join(
            ['control', self.__class__.__name__, str(self.object_id)])

        control = vthread.Thread(self.m, name, self.clk, self.rst, control_func,
                                 fsm_as_module=fsm_as_module)

        frame = inspect.currentframe()
        control.start_frame = frame

        self.control = control

        return control

    def get_control_func(self):
        def func():
            self.control_sequence()

        return func

    def control_sequence(self, fsm):
        """ must be implemented in each _Operator class """
        raise NotImplementedError()

    def Cparam(self, name, *args, **kwargs):
        return self.m.Reg('cparam_%s' % self._name(name), *args, **kwargs)

    def CparamWire(self, name, *args, **kwargs):
        return self.m.Wire('cparam_%s' % self._name(name), *args, **kwargs)

    def set_control_params(self, fsm, control_param_len,
                           use_param_ram=False, min_param_ram_len=0):
        if control_param_len <= 1:
            return

        if not use_param_ram or control_param_len < min_param_ram_len:
            return self.set_control_params_mux(fsm)

        return self.set_control_params_ram(fsm)

    def set_control_params_ram(self, fsm):
        dst_regs = []
        for name in self.collect_all_control_param_names():
            v = getattr(self, name)
            if isinstance(v, (tuple, list)):
                dst_regs.extend(v)
            else:
                dst_regs.append(v)

        if not dst_regs:
            return

        dst_value_vec = self.control_param_ram.read(fsm, self.control_param_index)

        lsb = 0
        for dst_reg in dst_regs:
            width = dst_reg.width
            value = dst_value_vec[lsb:lsb + width]
            fsm(
                dst_reg(value)
            )
            lsb += width

        fsm.goto_next()

    def set_control_params_mux(self, fsm):
        fsm(
            self.control_param_index_reg(self.control_param_index)
        )

        fsm.goto_next()

    def run_control(self, fsm):
        if self.control is None:
            return

        self.control.run(fsm)

    def join_control(self, fsm):
        if self.control is None:
            return

        self.control.join(fsm)

    def reset_control(self, fsm):
        if self.control is None:
            return

        self.control.reset(fsm)

    def get_control_hash(self):
        return (self.get_stream_obj_hash(), self.get_ram_set_obj_hash())

    def copy_control_params(self, obj):
        for name in obj.collect_all_control_param_names():
            setattr(self, name, getattr(obj, name))

        self.copy_local_control_params(obj)

        self.control_param_names = obj.control_param_names

        self.control_param_index_reg = obj.control_param_index_reg
        self.control_param_ram = obj.control_param_ram

    def copy_local_control_params(self, obj, index_offset=0):
        for name, lparam in self.get_local_control_param_values().items():
            signame = self.to_local_control_param_name(index_offset, name)
            if hasattr(obj, signame):
                setattr(self, name, getattr(obj, signame))

        numerics = self.collect_arg_numerics()

        for i, arg in enumerate(numerics):
            if not isinstance(arg, _Operator):
                continue
            for name, lparam in arg.get_local_control_param_values().items():
                signame = self.to_local_control_param_name(index_offset + i + 1, name)
                if hasattr(obj, signame):
                    setattr(arg, name, getattr(obj, signame))

    def copy_control(self, obj):
        self.objaddr = obj.objaddr
        self.arg_objaddrs = obj.arg_objaddrs
        self.control = obj.control

    def get_layout(self):
        if self.layout is not None:
            return self.layout

        for arg in self.args:
            layout = arg.get_layout()
            if layout is not None:
                return layout

        return None

    def get_onnx_layout(self):
        if self.onnx_layout is not None:
            return self.onnx_layout

        for arg in self.args:
            onnx_layout = arg.get_onnx_layout()
            if onnx_layout is not None:
                return onnx_layout

        return None

    def get_eval_method(self):
        import nngen.verify as verify

        name = self.__class__.__name__
        method = getattr(verify, name, None)
        return method

    def eval(self, memo, input_dict, **kwargs):
        if id(self) in memo:
            return memo[id(self)]

        import nngen.verify as verify

        name = self.__class__.__name__

        args = [arg.eval(memo, input_dict)
                for arg in self.args]

        kwargs['dtype'] = self.dtype
        kwargs['name'] = self.name
        kwargs['par'] = self.par

        method = self.get_eval_method()
        ret = method(*args, **kwargs)
        memo[id(self)] = ret

        return ret


class _StreamingOperator(_Operator):
    input_chainable = True
    output_chainable = True
    chain_head = True

    @staticmethod
    def op(strm, *args, **kwargs):
        # return strm.Plus(*args)
        raise NotImplementedError()

    def __sub_str__(self):
        ret = []
        ret.append(_Operator.__sub_str__(self))
        chained = (' chained' if is_output_chainable_operator(self) and
                   not self.chain_head else '')
        ret.append(chained)
        return ''.join(ret)

    def __init__(self, *args, **opts):
        dtype = opts['dtype'] if 'dtype' in opts else None
        shape = opts['shape'] if 'shape' in opts else None
        name = opts['name'] if 'name' in opts else None
        par = opts['par'] if 'par' in opts else 1

        if shape is None:
            shape = max_shape(*args)

        _Operator.__init__(self, *args, dtype=dtype,
                           shape=shape, name=name, par=par)

    def get_required_substreams(self):
        substreams = []
        for arg in self.args:
            if are_chainable_operators(self, arg):
                substreams.extend(arg.get_required_substreams())
        return substreams

    def set_substreams(self, substreams):
        index = 0
        for arg in self.args:
            if are_chainable_operators(self, arg):
                num = len(arg.get_required_substreams())
                if num > 0:
                    arg.set_substreams(substreams[index: index + num])
                    index += num
        self.substreams = substreams[index:]

    def get_stream_func(self):
        def func(strm):
            values = self.get_stream_values(strm)
            if self.par == 1:
                data = values[0]
            else:
                data = strm.Cat(*reversed(values))
            strm.sink(data)

        return func

    def get_stream_values(self, strm):
        vars = []

        for arg in self.args:
            if are_chainable_operators(self, arg):
                values = arg.get_stream_values(strm)
                vars.append(values)

            else:
                datawidth = arg.get_op_width()
                vec_datawidth = datawidth * self.par
                point = arg.get_op_point()
                signed = arg.get_signed()
                dup = strm.constant(datawidth=1, signed=False)
                vec_var = strm.source(datawidth=vec_datawidth, signed=False)

                if self.par == 1:
                    values = [strm.ReinterpretCast(vec_var, datawidth, point, signed)]
                else:
                    split_values = strm.Split(vec_var, datawidth, point, signed, reverse=True)
                    values = [strm.Mux(dup, split_values[0], value) for value in split_values]

                vars.append(values)

        if len(vars) == 1:
            values = vars[0]
            width = self.get_op_width()
            point = self.get_op_point()
            signed = self.get_signed()
            rslts = [out_rcast(strm, self.op(strm, value, index=i),
                               width, point, signed)
                     for i, value in enumerate(values)]

        else:
            width = self.get_op_width()
            point = self.get_op_point()
            signed = self.get_signed()
            rslts = []
            for i in range(self.par):
                args = [values[i] for values in vars]
                rslts.append(out_rcast(strm, self.op(strm, *args, index=i),
                                       width, point, signed))

        return rslts

    def get_control_param_values(self):
        aligned_shape = self.get_aligned_shape()
        aligned_length = self.get_aligned_length()
        total_size = int(math.ceil(aligned_length / self.par))
        dma_size = int(math.ceil(aligned_shape[-1] / self.par))
        num_comp = int(math.ceil(total_size / dma_size))

        addr_inc = to_byte(align_word(self.shape[-1], self.get_word_alignment()) *
                           self.get_ram_width())

        sources = self.collect_sources()

        arg_addr_incs = []
        wrap_modes = []
        wrap_sizes = []
        for arg in sources:
            arg_addr_inc = to_byte(align_word(arg.shape[-1], arg.get_word_alignment()) *
                                   arg.get_ram_width())
            if tuple(arg.shape) == tuple(self.shape):
                wrap_mode = 0
                wrap_size = 0
            elif len(arg.shape) == 1 and arg.shape[-1] == 1:
                # stride-0
                wrap_mode = 2
                wrap_size = get_wrap_size(self.shape, arg.shape)
            else:
                # repeat
                wrap_mode = 1
                wrap_size = get_wrap_size(self.shape, arg.shape)
            arg_addr_incs.append(arg_addr_inc)
            wrap_modes.append(wrap_mode)
            wrap_sizes.append(wrap_size)

        return OrderedDict([('dma_size', dma_size),
                            ('num_comp', num_comp),
                            ('addr_inc', addr_inc),
                            ('arg_addr_incs', arg_addr_incs),
                            ('wrap_modes', wrap_modes),
                            ('wrap_sizes', wrap_sizes)])

    def control_sequence(self, fsm):
        sources = self.collect_sources()

        arg_gaddrs = [self.m.Reg(self._name('arg_gaddr_%d' % i),
                                 self.maxi.addrwidth, initval=0)
                      for i, _ in enumerate(self.arg_objaddrs)]
        out_gaddr = self.m.Reg(self._name('out_gaddr'),
                               self.maxi.addrwidth, initval=0)
        comp_count = self.m.Reg(self._name('comp_count'),
                                self.maxi.addrwidth + 1, initval=0)
        wrap_counts = [self.m.Reg(self._name('wrap_count_%d' % i),
                                  self.maxi.addrwidth + 1, initval=0)
                       for i, arg in enumerate(sources)]

        arg_pages = [self.m.Reg(self._name('arg_page_%d' % i), initval=0)
                     for i, _ in enumerate(self.arg_objaddrs)]
        arg_page_comp_offsets = [self.m.Reg(self._name('arg_page_comp_offset_%d' % i),
                                            self.maxi.addrwidth, initval=0)
                                 for i, _ in enumerate(self.arg_objaddrs)]
        arg_page_dma_offsets = [self.m.Reg(self._name('arg_page_dma_offset_%d' % i),
                                           self.maxi.addrwidth, initval=0)
                                for i, _ in enumerate(self.arg_objaddrs)]

        out_page = self.m.Reg(self._name('out_page'), initval=0)
        out_page_comp_offset = self.m.Reg(self._name('out_page_comp_offset'),
                                          self.maxi.addrwidth, initval=0)
        out_page_dma_offset = self.m.Reg(self._name('out_page_dma_offset'),
                                         self.maxi.addrwidth, initval=0)

        arg_page_size = min(ram.length for ram in self.input_rams) // 2
        out_page_size = self.output_rams[0].length // 2

        skip_read = self.m.Reg(self._name('skip_read'), initval=0)
        skip_comp = self.m.Reg(self._name('skip_comp'), initval=0)
        skip_write = self.m.Reg(self._name('skip_write'), initval=0)

        # --------------------
        # initialization phase
        # --------------------
        fsm(
            [arg_gaddr(0) for arg_gaddr in arg_gaddrs],
            out_gaddr(0),
            comp_count(0),
            [wrap_count(0) for wrap_count in wrap_counts]
        )

        fsm(
            [arg_page(0) for arg_page in arg_pages],
            [arg_page_comp_offset(0)
             for arg_page_comp_offset in arg_page_comp_offsets],
            [arg_page_dma_offset(0)
             for arg_page_dma_offset in arg_page_dma_offsets]
        )

        fsm(
            out_page(0),
            out_page_comp_offset(0),
            out_page_dma_offset(out_page_size)
        )

        fsm(
            skip_read(0),
            skip_comp(0),
            skip_write(1)
        )

        fsm.goto_next()

        # --------------------
        # Read phase
        # --------------------
        state_read = fsm.current

        # DMA read -> Stream run -> Stream wait -> DMA write
        for (ram, arg_objaddr,
             arg_gaddr, arg_page_dma_offset,
             wrap_mode, wrap_count, arg) in zip(self.input_rams, self.arg_objaddrs,
                                                arg_gaddrs, arg_page_dma_offsets,
                                                self.wrap_modes, wrap_counts, sources):

            b = fsm.current
            fsm.goto_next()

            # normal
            laddr = arg_page_dma_offset
            gaddr = arg_objaddr + arg_gaddr
            bus_lock(self.maxi, fsm)
            dma_read(self.maxi, fsm, ram, laddr, gaddr, self.dma_size)
            bus_unlock(self.maxi, fsm)
            fsm.goto_next()

            b_stride0 = fsm.current
            fsm.goto_next()

            # stride-0
            bus_lock(self.maxi, fsm)
            dma_read(self.maxi, fsm, ram, laddr, gaddr, 1)
            bus_unlock(self.maxi, fsm)
            fsm.goto_next()

            # for reuse
            e = fsm.current
            fsm.If(wrap_mode == 2, wrap_count > 0).goto_from(b, e)
            fsm.If(wrap_mode == 2, wrap_count == 0).goto_from(b, b_stride0)
            fsm.If(wrap_mode != 2).goto_from(b_stride0, e)

        state_read_end = fsm.current
        fsm.If(skip_read).goto_from(state_read, state_read_end)

        # --------------------
        # Comp phase
        # --------------------
        state_comp = fsm.current

        self.stream.source_join(fsm)

        # set_source, set_constant (dup)
        for (source_name, dup_name,
             arg_page_comp_offset,
             ram, wrap_mode) in zip(self.stream.sources.keys(),
                                    self.stream.constants.keys(),
                                    arg_page_comp_offsets,
                                    self.input_rams, self.wrap_modes):
            read_laddr = arg_page_comp_offset
            read_size = self.dma_size
            stride = vg.Mux(wrap_mode == 2, 0, 1)
            dup = vg.Mux(wrap_mode == 2, 1, 0)
            self.stream.set_constant(fsm, dup_name, dup)
            fsm.set_index(fsm.current - 1)
            self.stream.set_source(fsm, source_name, ram,
                                   read_laddr, read_size, stride)
            fsm.set_index(fsm.current - 1)

        # set_sink
        write_laddr = out_page_comp_offset
        write_size = self.dma_size

        for name, ram in zip(self.stream.sinks.keys(), self.output_rams):
            self.stream.set_sink(fsm, name, ram, write_laddr, write_size)
            fsm.set_index(fsm.current - 1)

        fsm.goto_next()

        self.stream.run(fsm)

        state_comp_end = fsm.current

        self.stream.join(fsm)

        state_comp_end_join = fsm.current

        fsm.If(skip_comp).goto_from(state_comp, state_comp_end)
        fsm.If(vg.Not(skip_comp)).goto_from(
            state_comp_end, state_comp_end_join)

        # --------------------
        # Write phase
        # --------------------
        state_write = fsm.current

        laddr = out_page_dma_offset
        gaddr = self.objaddr + out_gaddr
        bus_lock(self.maxi, fsm)
        dma_write(self.maxi, fsm,
                  self.output_rams[0], laddr, gaddr, self.dma_size, use_async=True)
        bus_unlock(self.maxi, fsm)

        fsm.goto_next()

        state_write_end = fsm.current
        fsm.If(skip_write).goto_from(state_write, state_write_end)

        # --------------------
        # update for next iteration
        # --------------------
        fsm(
            comp_count.inc()
        )

        fsm.If(vg.Not(skip_write))(
            out_gaddr.add(self.addr_inc)
        )

        for (arg_gaddr, arg_addr_inc,
             arg_page, arg_page_comp_offset, arg_page_dma_offset,
             wrap_mode, wrap_size,
             wrap_count, arg) in zip(arg_gaddrs, self.arg_addr_incs,
                                     arg_pages, arg_page_comp_offsets,
                                     arg_page_dma_offsets,
                                     self.wrap_modes, self.wrap_sizes,
                                     wrap_counts, sources):

            fsm.If(wrap_mode == 2)(
                wrap_count(1)
            )

            fsm.If(wrap_mode == 1)(
                arg_gaddr.add(arg_addr_inc),
                wrap_count.inc()
            )
            fsm.If(wrap_mode == 1, wrap_count == wrap_size - 1)(
                arg_gaddr(0),
                wrap_count(0)
            )

            fsm.If(wrap_mode == 0)(
                arg_gaddr.add(arg_addr_inc)
            )

            fsm.If(vg.Not(arg_page), wrap_mode != 2)(
                arg_page_comp_offset(arg_page_size),
                arg_page_dma_offset(arg_page_size),
                arg_page(1)
            )
            fsm.If(arg_page, wrap_mode != 2)(
                arg_page_comp_offset(0),
                arg_page_dma_offset(0),
                arg_page(0)
            )

        fsm.If(vg.Not(out_page))(
            out_page_comp_offset(out_page_size),
            out_page_dma_offset(0),
            out_page(1)
        )
        fsm.If(out_page)(
            out_page_comp_offset(0),
            out_page_dma_offset(out_page_size),
            out_page(0)
        )

        fsm(
            skip_write(0)
        )
        fsm.If(comp_count == self.num_comp - 1)(
            skip_read(1),
            skip_comp(1)
        )

        fsm.If(comp_count < self.num_comp).goto(state_read)
        fsm.If(comp_count == self.num_comp).goto_next()

        # wait for last DMA write
        dma_wait_write(self.maxi, fsm)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['par'] = self.par
        return _Operator.eval(self, memo, input_dict, **kwargs)


class _ElementwiseOperator(_StreamingOperator):
    pass


class _ReductionOperator(_StreamingOperator):
    input_chainable = True
    output_chainable = False

    default_value = 0

    @staticmethod
    def reduce_op(strm, *args, **kwargs):
        # return strm.ReduceAddValid(*args, **kwargs)
        raise NotImplementedError()

    @staticmethod
    def carry_op(strm, *args, **kwargs):
        # return strm.Plus(*args)
        raise NotImplementedError()

    def __init__(self, arg,
                 dtype=None, shape=None, name=None,
                 axis=None, keep_dims=False, par=1):

        _StreamingOperator.__init__(self, arg,
                                    dtype=dtype, shape=shape, name=name, par=par)

        if axis is not None:
            rank = get_rank(self.args[0].shape)
            prev = rank
            for i in reversed(sorted(axis)):
                if i < 0:
                    i += rank
                if i != prev - 1:
                    raise ValueError(
                        "not supported axis pattern: '%s'" % str(axis))
                prev = i

        self.axis = axis
        self.keep_dims = keep_dims

    def get_required_rams(self):
        inputs, _, temps = _Operator.get_required_rams(self)

        outputs = []

        shape = self.get_aligned_shape()
        min_size = shape[-1] * 2
        output_width = self.get_ram_width()

        outputs = [(output_width, min_size)]

        return inputs, outputs, temps

    def get_stream_func(self):
        def func(strm):
            data, valid = self.get_stream_reduce_value(strm)
            strm.sink(data, when=valid)

        return func

    def get_stream_reduce_value(self, strm):
        arg = self.args[0]

        if are_chainable_operators(self, arg):
            values = arg.get_stream_values(strm)
            vars.append(values)

        else:
            datawidth = arg.get_op_width()
            vec_datawidth = datawidth * self.par
            point = arg.get_op_point()
            signed = arg.get_signed()
            dup = strm.constant(datawidth=1, signed=False)
            vec_var = strm.source(datawidth=vec_datawidth, signed=False)

            if self.par == 1:
                values = [strm.ReinterpretCast(vec_var, datawidth, point, signed)]
            else:
                split_values = strm.Split(vec_var, datawidth, point, signed, reverse=True)
                values = [strm.Mux(dup, split_values[0], value) for value in split_values]

        width = self.get_op_width()
        vec_width = width * self.par
        point = self.get_op_point()
        signed = self.get_signed()

        size = strm.constant(datawidth=self.read_dma_size.bit_length(),
                             signed=False)

        omit_mask = strm.constant(datawidth=self.par, signed=False)
        omit_counter = strm.Counter(size=size)
        omits = [strm.Ands(b, omit_counter == size - 1)
                 for b in omit_mask]

        carry = strm.constant(datawidth=width, point=point, signed=signed)

        data_list = []
        for value, omit in zip(values, omits):
            if self.par > 1:
                value = strm.Mux(omit, strm.Int(self.default_value), value)

            data, valid = self.reduce_op(strm, value, size,
                                         width=width, signed=signed)
            data = out_rcast(strm, data, width, point, signed)
            data_list.append(data)

        data_list.append(carry)

        carry_func = functools.partial(self.carry_op, strm)
        data = strm.op_tree(carry_func,
                            strm.Int(self.default_value), None, *data_list)
        data = out_rcast(strm, data, width, point, signed)

        return data, valid

    def get_control_param_values(self):
        aligned_shape = self.args[0].get_aligned_shape()
        aligned_length = self.args[0].get_aligned_length()
        total_size = int(math.ceil(aligned_length / self.par))
        read_dma_size = int(math.ceil(aligned_shape[-1] / self.par))
        write_dma_size = self.shape[-1]
        num_comp = int(math.ceil(total_size / read_dma_size))

        addr_inc = to_byte(align_word(self.shape[-1], self.get_word_alignment()) *
                           self.get_ram_width())

        sources = self.collect_sources()

        arg_addr_incs = []
        wrap_modes = []
        wrap_sizes = []
        for arg in sources:
            arg_addr_inc = to_byte(align_word(arg.shape[-1], arg.get_word_alignment()) *
                                   arg.get_ram_width())
            if tuple(arg.shape) == tuple(self.args[0].shape):
                wrap_mode = 0
                wrap_size = 0
            elif len(arg.shape) == 1 and arg.shape[-1] == 1:
                # stride-0
                wrap_mode = 2
                wrap_size = get_wrap_size(self.args[0].shape, arg.shape)
            else:
                # repeat
                wrap_mode = 1
                wrap_size = get_wrap_size(self.args[0].shape, arg.shape)
            arg_addr_incs.append(arg_addr_inc)
            wrap_modes.append(wrap_mode)
            wrap_sizes.append(wrap_size)

        if self.args[0].shape[-1] % self.par == 0:
            stream_omit_mask = 0
        else:
            stream_omit_mask = 0
            for i in range(self.par):
                if self.par - (self.args[0].shape[-1] % self.par) >= (self.par - i):
                    stream_omit_mask |= (0x1 << i)

        if self.axis is None:
            max_reduce_op_count = num_comp - 1
        else:
            arg_aligned_shape = self.args[0].get_aligned_shape()
            max_reduce_op_count = 1
            for i in self.axis:
                max_reduce_op_count *= arg_aligned_shape[i]
            max_reduce_op_count = max_reduce_op_count // arg_aligned_shape[-1] - 1

        max_out_op_count = self.shape[-1] - 1

        return OrderedDict([('read_dma_size', read_dma_size),
                            ('write_dma_size', write_dma_size),
                            ('num_comp', num_comp),
                            ('addr_inc', addr_inc),
                            ('arg_addr_incs', arg_addr_incs),
                            ('wrap_modes', wrap_modes),
                            ('wrap_sizes', wrap_sizes),
                            ('stream_omit_mask', stream_omit_mask),
                            ('max_reduce_op_count', max_reduce_op_count),
                            ('max_out_op_count', max_out_op_count)])

    def control_sequence(self, fsm):
        sources = self.collect_sources()

        arg_gaddrs = [self.m.Reg(self._name('arg_gaddr_%d' % i),
                                 self.maxi.addrwidth, initval=0)
                      for i, _ in enumerate(self.arg_objaddrs)]
        out_gaddr = self.m.Reg(self._name('out_gaddr'),
                               self.maxi.addrwidth, initval=0)
        comp_count = self.m.Reg(self._name('comp_count'),
                                self.maxi.addrwidth + 1, initval=0)
        wrap_counts = [self.m.Reg(self._name('wrap_count_%d' % i),
                                  self.maxi.addrwidth + 1, initval=0)
                       for i, arg in enumerate(sources)]

        arg_pages = [self.m.Reg(self._name('arg_page_%d' % i), initval=0)
                     for i, _ in enumerate(self.arg_objaddrs)]
        arg_page_comp_offsets = [self.m.Reg(self._name('arg_page_comp_offset_%d' % i),
                                            self.maxi.addrwidth, initval=0)
                                 for i, _ in enumerate(self.arg_objaddrs)]
        arg_page_dma_offsets = [self.m.Reg(self._name('arg_page_dma_offset_%d' % i),
                                           self.maxi.addrwidth, initval=0)
                                for i, _ in enumerate(self.arg_objaddrs)]

        out_page = self.m.Reg(self._name('out_page'), initval=0)
        out_page_comp_offset = self.m.Reg(self._name('out_page_comp_offset'),
                                          self.maxi.addrwidth, initval=0)
        out_page_dma_offset = self.m.Reg(self._name('out_page_dma_offset'),
                                         self.maxi.addrwidth, initval=0)

        arg_page_size = min(ram.length for ram in self.input_rams) // 2
        out_page_size = self.output_rams[0].length // 2

        skip_read = self.m.Reg(self._name('skip_read'), initval=0)
        skip_comp = self.m.Reg(self._name('skip_comp'), initval=0)
        skip_write = self.m.Reg(self._name('skip_write'), initval=0)

        skip_carry_read = self.m.Reg(self._name('skip_carry_read'), initval=0)
        carry = self.m.Reg(self._name('carry'),
                           self.get_op_width(), initval=self.default_value,
                           signed=self.get_signed())

        reduce_op_count = self.m.Reg(self._name('reduce_op_count'),
                                     self.maxi.addrwidth, initval=0)
        out_op_count = self.m.Reg(self._name('out_op_count'),
                                  self.maxi.addrwidth, initval=0)

        # --------------------
        # initialization phase
        # --------------------
        fsm(
            [arg_gaddr(0) for arg_gaddr in arg_gaddrs],
            out_gaddr(0),
            comp_count(0),
            [wrap_count(0) for wrap_count in wrap_counts]
        )

        fsm(
            [arg_page(0) for arg_page in arg_pages],
            [arg_page_comp_offset(0)
             for arg_page_comp_offset in arg_page_comp_offsets],
            [arg_page_dma_offset(0)
             for arg_page_dma_offset in arg_page_dma_offsets]
        )

        fsm(
            out_page(0),
            out_page_comp_offset(0),
            out_page_dma_offset(out_page_size)
        )

        fsm(
            skip_read(0),
            skip_comp(0),
            skip_write(1)
        )

        fsm(
            skip_carry_read(1),
            carry(self.default_value)
        )

        fsm(
            reduce_op_count(0),
            out_op_count(0)
        )

        fsm.goto_next()

        # --------------------
        # Read phase
        # --------------------
        state_read = fsm.current

        # DMA read -> Stream run -> Stream wait -> DMA write
        for (ram, arg_objaddr,
             arg_gaddr, arg_page_dma_offset,
             wrap_mode, wrap_count, arg) in zip(self.input_rams, self.arg_objaddrs,
                                                arg_gaddrs, arg_page_dma_offsets,
                                                self.wrap_modes, wrap_counts, sources):

            b = fsm.current
            fsm.goto_next()

            # normal
            laddr = arg_page_dma_offset
            gaddr = arg_objaddr + arg_gaddr
            bus_lock(self.maxi, fsm)
            dma_read(self.maxi, fsm, ram, laddr, gaddr, self.read_dma_size)
            bus_unlock(self.maxi, fsm)
            fsm.goto_next()

            b_stride0 = fsm.current
            fsm.goto_next()

            # stride-0
            bus_lock(self.maxi, fsm)
            dma_read(self.maxi, fsm, ram, laddr, gaddr, 1)
            bus_unlock(self.maxi, fsm)
            fsm.goto_next()

            # for reuse
            e = fsm.current
            fsm.If(wrap_mode == 2, wrap_count > 0).goto_from(b, e)
            fsm.If(wrap_mode == 2, wrap_count == 0).goto_from(b, b_stride0)
            fsm.If(wrap_mode != 2).goto_from(b_stride0, e)

        state_read_end = fsm.current
        fsm.If(skip_read).goto_from(state_read, state_read_end)

        # --------------------
        # Comp phase
        # --------------------
        state_comp = fsm.current

        self.stream.source_join(fsm)

        # update carry
        laddr = out_page_comp_offset + out_op_count
        ram_value = self.output_rams[0].read(fsm, laddr)

        fsm.If(vg.Not(skip_carry_read))(
            carry(ram_value)
        )

        # set_constant (carry)
        name = list(self.stream.constants.keys())[len(self.stream.sources) + 2]
        self.stream.set_constant(fsm, name, carry)

        # set_source, set_constant (dup)
        for (source_name, dup_name,
             arg_page_comp_offset,
             ram, wrap_mode) in zip(self.stream.sources.keys(),
                                    self.stream.constants.keys(),
                                    arg_page_comp_offsets,
                                    self.input_rams, self.wrap_modes):
            read_laddr = arg_page_comp_offset
            read_size = self.read_dma_size
            stride = vg.Mux(wrap_mode == 2, 0, 1)
            dup = vg.Mux(wrap_mode == 2, 1, 0)
            self.stream.set_constant(fsm, dup_name, dup)
            fsm.set_index(fsm.current - 1)
            self.stream.set_source(fsm, source_name, ram,
                                   read_laddr, read_size, stride)
            fsm.set_index(fsm.current - 1)

        # set_sink
        write_laddr = out_page_comp_offset + out_op_count
        write_size = 1

        for name, ram in zip(self.stream.sinks.keys(), self.output_rams):
            self.stream.set_sink(fsm, name, ram, write_laddr, write_size)
            fsm.set_index(fsm.current - 1)

        # set_constant (size)
        name = list(self.stream.constants.keys())[len(self.stream.sources)]
        self.stream.set_constant(fsm, name, self.read_dma_size)

        # set_constant (omit_mask)
        name = list(self.stream.constants.keys())[len(self.stream.sources) + 1]
        self.stream.set_constant(fsm, name, self.stream_omit_mask)

        # set_constant (carry)
        name = list(self.stream.constants.keys())[len(self.stream.sources) + 2]
        self.stream.set_constant(fsm, name, carry)

        fsm.goto_next()

        self.stream.run(fsm)

        state_comp_end = fsm.current

        self.stream.join(fsm)

        state_comp_end_join = fsm.current

        fsm.If(skip_comp).goto_from(state_comp, state_comp_end)
        fsm.If(vg.Not(skip_comp)).goto_from(
            state_comp_end, state_comp_end_join)

        # --------------------
        # Write phase
        # --------------------
        state_write = fsm.current

        laddr = out_page_dma_offset
        gaddr = self.objaddr + out_gaddr
        out_size = self.write_dma_size

        bus_lock(self.maxi, fsm)
        dma_write(self.maxi, fsm,
                  self.output_rams[0], laddr, gaddr, out_size, use_async=True)
        bus_unlock(self.maxi, fsm)

        state_write_end = fsm.current
        fsm.If(skip_write).goto_from(state_write, state_write_end)

        # --------------------
        # update for next iteration
        # --------------------
        fsm(
            comp_count.inc()
        )

        fsm.If(vg.Not(skip_write))(
            out_gaddr.add(self.addr_inc)
        )

        for (arg_gaddr, arg_addr_inc,
             arg_page, arg_page_comp_offset, arg_page_dma_offset,
             wrap_mode, wrap_size,
             wrap_count, arg) in zip(arg_gaddrs, self.arg_addr_incs,
                                     arg_pages, arg_page_comp_offsets,
                                     arg_page_dma_offsets,
                                     self.wrap_modes, self.wrap_sizes,
                                     wrap_counts, sources):

            fsm.If(wrap_mode == 2)(
                wrap_count(1)
            )

            fsm.If(wrap_mode == 1)(
                arg_gaddr.add(arg_addr_inc),
                wrap_count.inc()
            )
            fsm.If(wrap_mode == 1, wrap_count == wrap_size - 1)(
                arg_gaddr(0),
                wrap_count(0)
            )

            fsm.If(wrap_mode == 0)(
                arg_gaddr.add(arg_addr_inc)
            )

            fsm.If(vg.Not(arg_page), wrap_mode != 2)(
                arg_page_comp_offset(arg_page_size),
                arg_page_dma_offset(arg_page_size),
                arg_page(1)
            )
            fsm.If(arg_page, wrap_mode != 2)(
                arg_page_comp_offset(0),
                arg_page_dma_offset(0),
                arg_page(0)
            )

        fsm.If(reduce_op_count == self.max_reduce_op_count,
               out_op_count == self.max_out_op_count,
               vg.Not(out_page))(
            out_page_comp_offset(out_page_size),
            out_page_dma_offset(0),
            out_page(1)
        )
        fsm.If(reduce_op_count == self.max_reduce_op_count,
               out_op_count == self.max_out_op_count,
               out_page)(
            out_page_comp_offset(0),
            out_page_dma_offset(out_page_size),
            out_page(0)
        )

        fsm.If(comp_count == self.num_comp - 1)(
            skip_read(1),
            skip_comp(1)
        )

        fsm(
            reduce_op_count.inc()
        )
        fsm.If(reduce_op_count == self.max_reduce_op_count)(
            reduce_op_count(0)
        )

        fsm(
            skip_write(1),
            skip_carry_read(0)
        )

        fsm.If(reduce_op_count == self.max_reduce_op_count)(
            carry(self.default_value),
            skip_carry_read(1),
            out_op_count.inc()
        )
        fsm.If(reduce_op_count == self.max_reduce_op_count,
               out_op_count == self.max_out_op_count)(
            out_op_count(0),
            skip_write(0)
        )

        fsm.If(comp_count < self.num_comp).goto(state_read)
        fsm.If(comp_count == self.num_comp).goto_next()

        # wait for last DMA write
        dma_wait_write(self.maxi, fsm)

    def eval(self, memo, input_dict, **kwargs):
        kwargs['axis'] = self.axis
        kwargs['keep_dims'] = self.keep_dims
        return _StreamingOperator.eval(self, memo, input_dict, **kwargs)


class _ActFuncOperator(_ElementwiseOperator):

    def get_act_func(self):
        method = self.get_eval_method()
        method = functools.partial(method, dtype=self.dtype)
        return method


class _View(_Operator):
    __redirects__ = ('word_alignment',
                     'global_index', 'local_index',
                     'default_global_addr', 'default_local_addr',
                     'objaddr',
                     'add_consumer', 'set_output', 'add_alignment_request')

    def __init__(self, value, shape=None, dtype=None, name=None):
        if shape is None:
            shape = value.shape
        if dtype is None:
            dtype = value.dtype
        _Operator.__init__(self, value, dtype=dtype, shape=shape, name=name)

    def set_global_index(self, global_index):
        raise ValueError('_View object does not accept global_index.')

    def set_local_index(self, local_index):
        raise ValueError('_View object does not accept local_index.')

    def set_default_global_addr(self, default_global_addr):
        raise ValueError('_View object does not accept default_global_addr.')

    def set_default_local_addr(self, default_local_addr):
        raise ValueError('_View object does not accept default_local_addr.')

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

    def get_control_param_values(self):
        return OrderedDict()

    def get_control_func(self):
        return None

    def __getattribute__(self, attr):
        if attr in _View.__redirects__:
            return getattr(self.args[0], attr)

        return _Operator.__getattribute__(self, attr)

    def eval(self, memo, input_dict, **kwargs):
        return self.args[0].eval(memo, input_dict)


class _Reshape(_Operator):
    input_chainable = False
    output_chainable = False

    def __init__(self, tensor, shape, dtype=None, name=None):

        length = tensor.get_length()

        use_minus_one = False
        minus_one_index = 0
        all_mul = 1

        for i, s in enumerate(shape):
            if s is None or s == -1:
                use_minus_one = True
                minus_one_index = i
            else:
                all_mul *= s

        if use_minus_one:
            shape[minus_one_index] = length // all_mul

        shape = tuple(shape)

        reshaped_length = functools.reduce(lambda x, y: x * y, shape, 1)
        if length != reshaped_length:
            raise ValueError('size mismatch: %d != %d' % (length, reshaped_length))

        _Operator.__init__(self, tensor,
                           dtype=dtype, shape=shape, name=name)

    def attribute(self):
        pass

    def get_required_rams(self):
        input_width = self.args[0].get_ram_width()
        input_size = max(self.args[0].shape[-1], self.shape[-1])
        inputs = [(input_width, input_size)]
        output_width = self.get_ram_width()
        output_size = input_size
        outputs = [(output_width, output_size)]
        temps = []
        return inputs, outputs, temps

    def get_stream_func(self):
        def func(strm):
            datawidth = self.args[0].get_op_width()
            src = strm.source(datawidth=datawidth)
            strm.sink(src)

        return func

    def get_control_param_values(self):
        ram = self.input_rams[0]

        total_size = self.get_length()
        read_size = self.args[0].shape[-1]
        in_offset_inc = to_byte(align_word(read_size, self.args[0].get_word_alignment()) *
                                self.args[0].get_ram_width())
        write_size = self.shape[-1]
        out_offset_inc = to_byte(align_word(write_size, self.get_word_alignment()) *
                                 self.get_ram_width())

        return OrderedDict([('total_size', total_size),
                            ('read_size', read_size),
                            ('in_offset_inc', in_offset_inc),
                            ('write_size', write_size),
                            ('out_offset_inc', out_offset_inc)])

    def control_sequence(self, fsm):
        in_ram = self.input_rams[0]
        out_ram = self.output_rams[0]

        total_count = self.m.Reg(self._name('total_count'),
                                 self.maxi.addrwidth, initval=0)

        in_offset = self.m.Reg(self._name('in_offset'),
                               self.maxi.addrwidth, initval=0)
        out_offset = self.m.Reg(self._name('out_offset'),
                                self.maxi.addrwidth, initval=0)

        count_read = self.m.Reg(self._name('count_read'),
                                self.maxi.addrwidth, initval=0)
        count_copy = self.m.Reg(self._name('count_copy'),
                                self.maxi.addrwidth, initval=0)
        count_write = self.m.Reg(self._name('count_write'),
                                 self.maxi.addrwidth, initval=0)

        copy_src = self.m.Reg(self._name('copy_src'),
                              self.maxi.addrwidth, initval=0)
        copy_dst = self.m.Reg(self._name('copy_dst'),
                              self.maxi.addrwidth, initval=0)
        copy_size = self.m.Reg(self._name('copy_size'),
                               self.maxi.addrwidth, initval=0)

        # initialize
        fsm(
            total_count(0),
            in_offset(0),
            out_offset(0),
            count_read(0),
            count_copy(0),
            count_write(0),
            copy_src(0),
            copy_dst(0),
            copy_size(0)
        )
        fsm.goto_next()

        # read
        state_read = fsm.current

        gaddr = self.arg_objaddrs[0] + in_offset

        bus_lock(self.maxi, fsm)
        dma_read(self.maxi, fsm, in_ram, 0, gaddr, self.read_size)
        bus_unlock(self.maxi, fsm)

        fsm(
            in_offset.add(self.in_offset_inc),
            count_read.add(self.read_size),
            copy_src(0)
        )

        fsm.goto_next()

        # copy
        state_copy = fsm.current

        fsm.If(self.read_size <= self.write_size)(
            copy_size(self.read_size)
        )
        fsm.If(self.read_size <= self.write_size,
               count_read - count_copy < self.read_size)(
            copy_size(count_read - count_copy)
        )

        fsm.If(self.write_size < self.read_size)(
            copy_size(self.write_size)
        )
        fsm.If(self.write_size < self.read_size,
               count_read - count_copy < self.write_size)(
            copy_size(count_read - count_copy)
        )

        fsm.If(count_read == count_copy).goto(state_read)
        fsm.If(count_read > count_copy).goto_next()

        name = list(self.stream.sources.keys())[0]
        self.stream.set_source(fsm, name, in_ram, copy_src, copy_size)
        fsm.set_index(fsm.current - 1)
        name = list(self.stream.sinks.keys())[0]
        self.stream.set_sink(fsm, name, out_ram, copy_dst, copy_size)

        self.stream.run(fsm)
        self.stream.join(fsm)

        fsm(
            count_copy.add(copy_size)
        )

        fsm.If(count_write + self.write_size >
               count_copy + copy_size)(
            copy_src(0),
            copy_dst.add(copy_size)
        )
        fsm.If(count_write + self.write_size <=
               count_copy + copy_size)(
            copy_src.add(copy_size),
            copy_dst(0)
        )
        fsm.If(count_write + self.write_size >
               count_copy + copy_size).goto(state_read)
        fsm.If(count_write + self.write_size <=
               count_copy + copy_size).goto_next()

        # write
        state_write = fsm.current

        gaddr = self.objaddr + out_offset

        bus_lock(self.maxi, fsm)
        dma_write(self.maxi, fsm, out_ram, 0, gaddr, self.write_size)
        bus_unlock(self.maxi, fsm)

        fsm(
            total_count.add(self.write_size),
            out_offset.add(self.out_offset_inc),
            count_write.add(self.write_size),
            copy_dst(0)
        )

        fsm.goto(state_copy)
        fsm.If(total_count + self.write_size >= self.total_size).goto_next()

    def get_original_shape(self):
        return self.args[0].get_original_shape()

    def get_layout(self):
        return self.layout

    def get_onnx_layout(self):
        return self.onnx_layout

    def get_original_layout(self):
        return self.args[0].get_original_layout()

    def get_original_onnx_layout(self):
        return self.args[0].get_original_onnx_layout()

    def eval(self, memo, input_dict, **kwargs):
        if id(self) in memo:
            return memo[id(self)]

        arg = self.args[0].eval(memo, input_dict)
        return np.reshape(arg, [-1] + list(self.shape[1:]))


class _LazyReshape(_Reshape):

    def __sub_str__(self):
        alias = (' alias_of:%s' % self._get_alias_name()
                 if self.maxi is not None and self._condition() else '')
        return alias

    def _condition(self):
        return (self.args[0].get_aligned_length() == self.get_aligned_length() and
                self.args[0].dtype.width == self.dtype.width)

    def _get_alias_name(self):
        if self._condition():
            if isinstance(self.args[0], _LazyReshape):
                name = self.args[0]._get_alias_name()
            else:
                name = self.args[0].name

            if name is None:
                name = '<%s>' % self.args[0].__class__.__name__

        else:
            name = self.name

            if name is None:
                name = '<%s>' % self.__class__.__name__

        return name

    def __init__(self, tensor, shape, dtype=None, name=None):
        _Reshape.__init__(self, tensor, shape, dtype=dtype, name=name)
        self._global_index = None
        self._local_index = None
        self._default_global_addr = None
        self._default_local_addr = None

    def add_alignment_request(self, num_words):
        _Reshape.add_alignment_request(self, num_words)
        self.args[0].add_alignment_request(num_words)

    def get_required_rams(self):
        if self._condition():
            inputs = ()
            outputs = ()
            temps = ()
            return inputs, outputs, temps

        return _Reshape.get_required_rams(self)

    def get_stream_hash(self):
        h = _Reshape.get_stream_hash(self)
        return (h, self._condition())

    def get_stream_func(self):
        if self._condition():
            return None

        return _Reshape.get_stream_func(self)

    def get_control_param_values(self):
        if self._condition():
            return OrderedDict()

        return _Reshape.get_control_param_values(self)

    def make_control_params(self, control_param_len, width_dict, signed_dict,
                            use_param_ram=False, min_param_ram_len=0):
        if self._condition():
            self.control_param_names = ()
            return

        return _Reshape.make_control_params(self, control_param_len, width_dict, signed_dict,
                                            use_param_ram, min_param_ram_len)

    def get_control_func(self):
        if self._condition():
            return None

        return _Reshape.get_control_func(self)

    @property
    def global_index(self):
        if self.maxi is not None and self._condition():
            return self.args[0].global_index

        return self._global_index

    @global_index.setter
    def global_index(self, global_index):
        self._global_index = global_index

    @property
    def local_index(self):
        if self.maxi is not None and self._condition():
            return self.args[0].local_index

        return self._local_index

    @local_index.setter
    def local_index(self, local_index):
        self._local_index = local_index

    @property
    def default_global_addr(self):
        if self.maxi is not None and self._condition():
            return self.args[0].default_global_addr

        return self._default_global_addr

    @default_global_addr.setter
    def default_global_addr(self, default_global_addr):
        self._default_global_addr = default_global_addr

    @property
    def default_local_addr(self):
        if self.maxi is not None and self._condition():
            return self.args[0].default_local_addr

        return self._default_local_addr

    @default_local_addr.setter
    def default_local_addr(self, default_local_addr):
        self._default_local_addr = default_local_addr

    @property
    def objaddr(self):
        if self._condition():
            return self.args[0].objaddr

        return self._objaddr

    @objaddr.setter
    def objaddr(self, objaddr):
        self._objaddr = objaddr


def is_storage(obj):
    return isinstance(obj, _Storage)


def is_input_storage(obj):
    if not is_storage(obj):
        return False
    return obj.is_input


def is_constant(obj):
    return isinstance(obj, _Constant)


def is_operator(obj):
    return isinstance(obj, _Operator)


def is_input_chainable_operator(obj):
    return (isinstance(obj, _Operator) and
            obj.input_chainable)


def is_output_chainable_operator(obj):
    return (isinstance(obj, _Operator) and
            obj.output_chainable)


def are_chainable_operators(consumer, producer):
    return (is_input_chainable_operator(consumer) and
            is_output_chainable_operator(producer) and
            hasattr(consumer, 'par') and
            hasattr(producer, 'par') and
            consumer.par == producer.par)


def is_elementwise_operator(obj):
    return isinstance(obj, _ElementwiseOperator)


def is_reduction_operator(obj):
    return isinstance(obj, _ReductionOperator)


def is_view(obj):
    return isinstance(obj, _View)


def is_removable_reshape(obj):
    return isinstance(obj, _LazyReshape) and obj._condition()


def same_dtype(*args):
    ret = None
    for arg in args:
        if arg.dtype is None:
            raise ValueError('not supported')

        if ret is None:
            ret = arg.dtype
            continue

        if ret != arg.dtype:
            raise ValueError('dtype mismatch: %s != %s' %
                             (str(ret), str(arg.dtype)))
        ret = arg.dtype

    return ret


def same_shape(*args):
    ret = None
    for arg in args:
        if arg.shape is None:
            raise ValueError('not supported')

        if ret is None:
            ret = tuple(arg.shape)
            continue

        if ret != tuple(arg.shape):
            raise ValueError('shape mismatch: %s != %s' %
                             (str(ret), str(tuple(arg.shape))))
        ret = tuple(arg.shape)

    return ret


def max_shape(*args):
    ret = None
    for arg in args:
        if arg.shape is None:
            raise ValueError('not supported')

        if ret is None:
            ret = tuple(arg.shape)
            continue

        if ret != tuple(arg.shape):
            ret_rank = get_rank(ret)
            arg_rank = get_rank(tuple(arg.shape))

            if ret_rank == arg_rank:
                if ret_rank == 1 and ret[0] == 1:
                    ret = tuple(arg.shape)
                elif arg_rank == 1 and arg.shape[0] == 1:
                    pass
                else:
                    raise ValueError('shape mismatch: %s != %s' %
                                     (str(ret), str(tuple(arg.shape))))

            if ret_rank > arg_rank:
                if arg_rank == 1 and arg.shape[0] == 1:
                    continue

                for r, a in zip(ret[-arg_rank:], arg.shape):
                    if r != a:
                        raise ValueError('shape mismatch: %s != %s' %
                                         (str(ret), str(tuple(arg.shape))))

                shape = []
                for r in ret[:-arg_rank]:
                    shape.append(r)
                shape.extend(arg.shape)
                ret = tuple(shape)
                continue

            if arg_rank > ret_rank:
                for a, r in zip(arg[-ret_rank:], ret.shape):
                    if a != r:
                        raise ValueError('shape mismatch: %s != %s' %
                                         (str(arg), str(tuple(ret.shape))))

                shape = []
                for r in arg[:-ret_rank]:
                    shape.append(r)
                shape.extend(ret.shape)
                ret = tuple(shape)
                continue

    return ret


def to_byte(width, ceil=True):
    v = width // 8
    if ceil and v < 1:
        return 1
    return v


def shape_to_length(shape):
    return functools.reduce(lambda x, y: x * y, shape, 1)


def shape_to_pattern(shape, order):
    pattern = []

    for p in order:
        size = shape[p]
        stride = shape_to_length(shape[p + 1:])
        pattern.append((size, stride))

    return pattern


def get_wrap_size(base_shape, shape):
    if len(shape) == 1 and shape[-1] == 1:
        return 0

    ptr = -1
    for base, orig in zip(reversed(base_shape[:-1]), reversed(shape[:-1])):
        if base != orig:
            raise ValueError('shape mismatch: %s != %s' %
                             (str(base_shape), str(tuple(shape))))
        ptr -= 1

    return shape_to_length(shape[ptr:-1])


def get_value_shape(value):
    if isinstance(value, (tuple, list)):
        if isinstance(value[0], (tuple, list)):
            return [len(value)] + get_value_shape(value[0])
        else:
            return [len(value)]
    return [1]


def get_rank(shape):
    return len(shape)


def get_min_size(*rams):
    size = 0

    for ram in rams:
        l = ram.length
        if not isinstance(l, int):
            raise TypeError('illegal length format')
        size = min(size, l) if size > 0 else l

    return size


def align_word(size, word_alignment):
    res = word_alignment - (size % word_alignment)
    if res == word_alignment:
        res = 0
    return size + res


def log_width(v):
    lv = int(math.ceil(math.log(v, 2)))
    if lv == 0:
        return 1
    return lv


def flatten_list(*values):
    ret = []
    for value in values:
        if isinstance(value, (tuple, list)):
            ret.extend(flatten_list(*value))
        else:
            ret.append(value)
    return tuple(ret)


def bus_lock(maxi, fsm):
    """ do nothing """
    pass


def bus_unlock(maxi, fsm):
    """ do nothing """
    pass


def dma_len(ram, words):
    if not isinstance(ram, vthread.MultibankRAM):
        return words
    shift = int(math.ceil(math.log(ram.numbanks, 2)))
    if shift == 0:
        return words
    mask = vg.Repeat(vg.Int(1, 1), shift)
    rest = vg.Mux(vg.And(words, mask) > 0, 1, 0)
    return vg.Srl(words, shift) + rest


def dma_laddr(ram, laddr):
    if not isinstance(ram, vthread.MultibankRAM):
        return laddr
    shift = int(math.ceil(math.log(ram.numbanks, 2)))
    return laddr >> shift


def dma_read(maxi, fsm, ram, laddr, gaddr, size, port=1, use_async=False):
    size = dma_len(ram, size)
    laddr = dma_laddr(ram, laddr)
    if use_async:
        return maxi.dma_read_async(fsm, ram, laddr, gaddr, size, port=1)
    return maxi.dma_read(fsm, ram, laddr, gaddr, size, port=1)


def dma_write(maxi, fsm, ram, laddr, gaddr, size, port=1, use_async=False):
    size = dma_len(ram, size)
    laddr = dma_laddr(ram, laddr)
    if use_async:
        return maxi.dma_write_async(fsm, ram, laddr, gaddr, size, port=1)
    return maxi.dma_write(fsm, ram, laddr, gaddr, size, port=1)


def dma_read_block(maxi, fsm, rams, laddr, gaddr, size, block_size, port=1, use_async=False):
    if not isinstance(rams, (tuple, list)):
        raise TypeError('rams must be tuple or list.')

    if len(rams) > 1:
        ram = vthread.to_multibank_ram(rams, keep_hierarchy=True)
    else:
        ram = rams[0]

    if len(rams) == 1 and not isinstance(ram, vthread.MultibankRAM):
        return dma_read(maxi, fsm, ram, laddr, gaddr, size, port=port, use_async=use_async)

    size = dma_len(rams[0], size)
    laddr = dma_laddr(rams[0], laddr)

    if isinstance(rams[0], vthread.MultibankRAM):
        pack_size = rams[0].numbanks
        block_size = block_size >> int(math.ceil(math.log(pack_size, 2)))

    if use_async:
        return ram.dma_read_block_async(fsm, maxi, laddr, gaddr, size, block_size, port=1)
    return ram.dma_read_block(fsm, maxi, laddr, gaddr, size, block_size, port=1)


def dma_write_block(maxi, fsm, rams, laddr, gaddr, size, block_size, port=1, use_async=False):
    if not isinstance(rams, (tuple, list)):
        raise TypeError('rams must be tuple or list.')

    if len(rams) > 1:
        ram = vthread.to_multibank_ram(rams, keep_hierarchy=True)
    else:
        ram = rams[0]

    if len(rams) == 1 and not isinstance(ram, vthwrite.MultibankRAM):
        return dma_write(maxi, fsm, ram, laddr, gaddr, size, port=port, use_async=use_async)

    size = dma_len(rams[0], size)
    laddr = dma_laddr(rams[0], laddr)

    if isinstance(rams[0], vthread.MultibankRAM):
        pack_size = rams[0].numbanks
        block_size = block_size >> int(math.ceil(math.log(pack_size, 2)))

    if use_async:
        return ram.dma_write_block_async(fsm, maxi, laddr, gaddr, size, block_size, port=1)
    return ram.dma_write_block(fsm, maxi, laddr, gaddr, size, block_size, port=1)


def dma_wait_read(maxi, fsm):
    return maxi.dma_wait_read(fsm)


def dma_wait_write(maxi, fsm):
    return maxi.dma_wait_write(fsm)


def dma_wait(maxi, fsm):
    return maxi.dma_wait(fsm)


def read_modify_write(m, fsm, maxi,
                      src_ram, dst_ram, laddr, gaddr):

    if (isinstance(src_ram, vthread.MultibankRAM) and
            isinstance(dst_ram, vthread.MultibankRAM)):
        if src_ram.datawidth != dst_ram.datawidth:
            raise ValueError('datawidth mismatch: %d != %d' % (src_ram.datawidth,
                                                               dst_ram.datawidth))
        if src_ram.numbanks != dst_ram.numbanks:
            raise ValueError('numbanks mismatch: %d != %d' % (src_ram.numbanks,
                                                              dst_ram.numbanks))

    elif (not isinstance(src_ram, vthread.MultibankRAM) and
          not isinstance(dst_ram, vthread.MultibankRAM)):
        if src_ram.datawidth != dst_ram.datawidth:
            raise ValueError('datawidth mismatch: %d != %d' % (src_ram.datawidth,
                                                               dst_ram.datawidth))
    else:
        raise TypeError('type mismatch: %s != %s' %
                        (type(src_ram), type(dst_ram)))

    # (read)
    if ((isinstance(src_ram, vthread.MultibankRAM) and
         src_ram.orig_datawidth == maxi.datawidth) or
        (not isinstance(src_ram, vthread.MultibankRAM) and
            src_ram.datawidth == maxi.datawidth)):
        bus_lock(maxi, fsm)
        maxi.dma_write(fsm, src_ram, laddr, gaddr, 1, port=1)
        bus_unlock(maxi, fsm)
        return

    bus_lock(maxi, fsm)
    maxi.dma_read(fsm, dst_ram, 0, gaddr, 1, port=1)
    bus_unlock(maxi, fsm)

    # (modify)
    write_value = src_ram.read(fsm, laddr)

    # (write)
    mem_width = int(math.log(to_byte(maxi.datawidth), 2))
    if isinstance(dst_ram, vthread.MultibankRAM):
        ram_width = int(math.log(to_byte(dst_ram.orig_datawidth), 2))
    else:
        ram_width = int(math.log(to_byte(dst_ram.datawidth), 2))
    pos_width = mem_width - ram_width
    if pos_width < 1:
        pos = 0
    else:
        pos = m.TmpWire(pos_width)
        pos.assign(gaddr >> ram_width)

    dst_ram.write(fsm, pos, write_value)

    bus_lock(maxi, fsm)
    maxi.dma_write(fsm, dst_ram, 0, gaddr, 1, port=1)
    bus_unlock(maxi, fsm)


def read_modify_write_single_bank(m, fsm, maxi, src_ram, dst_ram, laddr, gaddr):

    # (read)
    if maxi.datawidth != src_ram.datawidth:
        bus_lock(maxi, fsm)
        dma_read(maxi, fsm, dst_ram, 0, gaddr, 1, port=1)
        bus_unlock(maxi, fsm)

    # (modify)
    if maxi.datawidth != src_ram.datawidth:
        new_value = src_ram.read(fsm, laddr)
        orig_value = dst_ram.read(fsm, 0)

        write_pos_width = int(math.log(to_byte(maxi.datawidth), 2))
        write_pos = m.TmpWire(write_pos_width)
        write_pos.assign(gaddr)

        shift = m.TmpWire(write_pos_width + 3)
        shift.assign(optimize(write_pos * 8))

        or_mask = vg.Repeat(vg.Int(1, 1), src_ram.datawidth) << shift
        not_orig_value = vg.Unot(orig_value)
        masked_orig_value = vg.Unot(vg.Or(not_orig_value, or_mask))
        write_value = vg.Or(masked_orig_value, new_value << shift)

    else:
        write_value = src_ram.read(fsm, laddr)

    # (write)
    dst_ram.write(fsm, 0, write_value)

    bus_lock(maxi, fsm)
    dma_write(maxi, fsm, dst_ram, 0, gaddr, 1, port=1)
    bus_unlock(maxi, fsm)


def get_maxi_datawidth(obj):
    return obj.maxi.datawidth


def get_maxi_addrwidth(obj):
    return obj.maxi.addrwidth


def out_rcast(strm, v, width, point, signed):
    if v.bit_length() < width:
        v.width = width

    return strm.ReinterpretCast(v, width, point, signed)
