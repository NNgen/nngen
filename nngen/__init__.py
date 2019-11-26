"""
   NNgen: A Fully-Customizable Hardware Synthesis Compiler for Deep Neural Network

   Copyright 2017, Shinya Takamaeda-Yamazaki and Contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .version import __version__

from .storage import *
from .dtype_list import *
from .operator import *
from .util import *
from .eval import eval
from .onnx import from_onnx
from .quantizer import quantize

from .verilog import to_ipxact, to_verilog, to_veriloggen
from .verilog import header_reg
from .verilog import control_reg_start, control_reg_busy, control_reg_reset
from .verilog import control_reg_extern_send, control_reg_extern_recv
from .verilog import control_reg_global_offset
from .verilog import control_reg_global_addr
from .verilog import control_reg_load_global_addr_map, control_reg_busy_global_addr_map, control_reg_addr_global_addr_map

from . import verify
from . import sim
