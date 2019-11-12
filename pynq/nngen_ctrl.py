from __future__ import absolute_import
from __future__ import print_function


def nngen_ip(overlay, name):
    """
    Get a NNgenIP object from overlay

    Parameters
    ----------
    overlay : Overlay
        Target Overlay object

    name : str
        IP-core name

    Returns
    -------
    ip : NNgenIP
        NNgenIP Object
    """

    ip = getattr(overlay, name)
    ip = NNgenIP(ip)

    return ip


class NNgenIP(object):
    """
    wrapper class for NNgen IP-core
    """

    WORDSIZE_REG = 4

    REG_START = 0
    REG_BUSY = 1
    REG_RESET = 2
    REG_EXTERN_OPCODE = 3
    REG_EXTERN_RESUME = 4
    REG_GLOBAL_OFFSET = 5
    REG_MEM_TMP = 6
    REG_MEM_OBJ = 7

    def __init__(self, base_ip, num_headers=4):
        self.num_headers = num_headers
        self.base_ip = base_ip

    def set_global_offset(self, buf):
        addr = buf.physical_address
        reg = self.WORDSIZE_REG * (self.REG_GLOBAL_OFFSET + self.num_headers)
        self.base_ip.write(reg, addr)

    def set_temporal_buffer(self, buf):
        addr = buf.physical_address
        reg = self.WORDSIZE_REG * (self.REG_MEM_TMP + self.num_headers)
        self.base_ip.write(reg, addr)

    def set_placeholder(self, index, buf):
        addr = buf.physical_address
        reg = self.WORDSIZE_REG * (self.REG_MEM_OBJ + self.num_headers + index)
        self.base_ip.write(reg, addr)

    def run(self):
        reg = self.WORDSIZE_REG * (self.REG_START + self.num_headers)
        self.base_ip.write(reg, 1)

    def wait(self):
        reg = self.WORDSIZE_REG * (self.REG_START + self.num_headers)
        busy = True
        while True:
            busy = self.base_ip.read(reg)

    def wait_extern(self):
        reg = self.WORDSIZE_REG * (self.REG_EXTERN_OPCODE + self.num_headers)
        code = 0
        while code == 0:
            code = self.base_ip.read(reg)
        return code

    def resume_extern(self):
        reg = self.WORDSIZE_REG * (self.REG_EXTERN_RESUME + self.num_headers)
        self.base_ip.write(reg, 1)
