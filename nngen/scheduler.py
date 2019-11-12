from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import defaultdict
import inspect

from . import basic_types as bt


class _Scheduler(object):
    """ ASAP scheduler """

    def __init__(self):
        self.current_stage = 0
        self.result = defaultdict(list)

    def schedule(self, objs):
        not_scheduled = set()
        not_scheduled.update(objs)

        while not_scheduled:
            candidates = []
            for obj in sorted(not_scheduled, key=lambda x: x.object_id, reverse=True):
                if (self.is_schedulable(obj, self.current_stage) and
                    self.check_constraints(obj) and
                    (obj.parallel_scheduling_allowed or
                     len(self.result[self.current_stage]) == 0)):
                    candidates.append(obj)

            if not candidates:
                self.next_stage()
                continue

            obj = self.select(*candidates)

            obj.set_stage(self.current_stage)
            not_scheduled.remove(obj)

            self.add_result(self.current_stage, obj)

            if not obj.parallel_scheduling_allowed:
                self.next_stage()
                continue

        return self.result

    def is_schedulable(self, obj, stage):
        return obj.is_schedulable(stage)

    def check_constraints(self, obj):
        methods = [method for name, method in inspect.getmembers(self)
                   if inspect.ismethod(method) and
                   name.startswith('constraint_')]

        for method in methods:
            ret = method(obj)
            if not ret:
                return False

        return True

    def next_stage(self):
        self.current_stage += 1

    def select(self, *objs):
        if not objs:
            raise ValueError('no candidate obj')

        return objs[0]

    def add_result(self, stage, obj):
        self.result[stage].append(obj)


class _ListScheduler(_Scheduler):
    """ List scheduler """

    def __init__(self, config):
        _Scheduler.__init__(self)
        self.config = config

    def select(self, *objs):
        if not objs:
            raise ValueError('no candidate')

        return self.select_highest_priority(*objs)

    def select_highest_priority(self, *objs):
        ret = objs[0]
        maxval = self.get_priority(ret)

        for obj in objs[1:]:
            p = self.get_priority(obj)
            if p > maxval:
                ret = obj
                maxval = p

        return ret

    def get_priority(self, obj):
        if bt.is_reduction_operator(obj):
            srcs = obj.collect_sources()
            return len(srcs) + 1

        if bt.is_elementwise_operator(obj):
            srcs = obj.collect_sources()
            return len(srcs) + 2

        return 0


class OperationScheduler(_ListScheduler):
    """ List scheduler """

    def constraint_max_parallel_ops(self, obj):
        if bt.is_storage(obj):
            return True

        if bt.is_output_chainable_operator(obj) and not obj.chain_head:
            return True

        max_parallel_ops = None
        if 'max_parallel_ops' in self.config:
            max_parallel_ops = self.config['max_parallel_ops']

        if max_parallel_ops is None:
            return True

        num_effective_ops = 0
        for op in self.result[self.current_stage]:
            if bt.is_output_chainable_operator(op) and not op.chain_head:
                continue
            num_effective_ops += 1

        if num_effective_ops >= max_parallel_ops:
            return False

        return True
