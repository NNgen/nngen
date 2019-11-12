from __future__ import absolute_import
from __future__ import print_function

import pytest


def pytest_addoption(parser):
    parser.addoption('--sim', default='iverilog', help='Simulator')
