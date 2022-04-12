from __future__ import absolute_import
from __future__ import print_function


def nngen_core(overlay, name):
    """
    Get a NNgenCore object from overlay

    Parameters
    ----------
    overlay : Overlay
        Target Overlay object

    name : str
        IP-core name

    Returns
    -------
    ip : NNgenCore
        NNgenCore Object
    """

    ip = getattr(overlay, name)
    ip = NNgenCore(ip)

    return ip


class NNgenCore(object):
    """
    NNgen IP-core Controller
    """

    WORDSIZE_REG = 4

    NUM_HEADERS = 4
    REG_HEADER = 0
    REG_START = REG_HEADER + NUM_HEADERS
    REG_BUSY = REG_START + 1
    REG_RESET = REG_BUSY + 1
    REG_EXTERN_OPCODE = REG_RESET + 1
    REG_EXTERN_RESUME = REG_EXTERN_OPCODE + 1

    # REG_GLOBAL_OFFSET must be same as control_reg_global_offset in "nngen/verilog.py".
    REG_GLOBAL_OFFSET = 32
    REG_MEM_TMP = REG_GLOBAL_OFFSET + 1
    REG_MEM_OBJ = REG_MEM_TMP + 1

    def __init__(self, base_ip):
        self.base_ip = base_ip

    # --------------------
    def run(self):
        reg = self.WORDSIZE_REG * self.REG_START
        self.base_ip.write(reg, 1)

    def wait(self):
        reg = self.WORDSIZE_REG * self.REG_BUSY
        busy = True
        while busy:
            busy = self.base_ip.read(reg)

    def wait_extern(self):
        reg = self.WORDSIZE_REG * self.REG_EXTERN_OPCODE
        code = 0
        while code == 0:
            code = self.base_ip.read(reg)
        return code

    def resume_extern(self):
        reg = self.WORDSIZE_REG * self.REG_EXTERN_RESUME
        self.base_ip.write(reg, 1)

    # --------------------
    def set_global_buffer(self, buf):
        """ Assign global buffer shared by all placeholders, variables, and temporal uses """
        addr = buf.physical_address
        self.write_global_offset(addr)

    def set_temporal_buffer(self, buf):
        addr = buf.physical_address
        self.write_temporal_address(addr)

    def set_buffer(self, index, buf):
        addr = buf.physical_address
        self.write_buffer_address(index, addr)

    # --------------------
    def write_global_offset(self, addr):
        reg = self.WORDSIZE_REG * self.REG_GLOBAL_OFFSET
        self.base_ip.write(reg, addr)

    def write_temporal_address(self, addr):
        reg = self.WORDSIZE_REG * self.REG_MEM_TMP
        self.base_ip.write(reg, addr)

    def write_buffer_address(self, index, addr):
        reg = self.WORDSIZE_REG * (self.REG_MEM_OBJ + index)
        self.base_ip.write(reg, addr)

    # --------------------
    def read_header(self, addr, index):
        reg = self.WORDSIZE_REG * (self.REG_HEADER + index)
        return self.base_ip.read(reg)

    def read_global_offset(self):
        reg = self.WORDSIZE_REG * self.REG_GLOBAL_OFFSET
        return self.base_ip.read(reg)

    def read_temporal_address(self):
        reg = self.WORDSIZE_REG * self.REG_MEM_TMP
        return self.base_ip.read(reg)

    def read_buffer_address(self, index):
        reg = self.WORDSIZE_REG * (self.REG_MEM_OBJ + index)
        return self.base_ip.read(reg)
