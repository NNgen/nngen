from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import re
import os


def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


VERSION = read("VERSION").splitlines()[0]
