from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


def eval(objs, **input_dict):
    memo = {}
    return [obj.eval(memo, input_dict) for obj in objs]
