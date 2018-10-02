#coding:utf-8
import os, sys, time
import copy
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.dataset import concat_examples
from chainer.dataset import iterator as iterator_module
from chainer.dataset import concat_examples
from chainer import reporter

class EspcnEvaluator(chainer.training.extensions.Evaluator):
    def __init__(self, iterator, gen, num_of_calc=10, converter=concat_examples, device=None, eval_hook=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {"main":iterator}
        self._iterators = iterator
        self._targets = {"gen" : gen}
        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self._num_of_calc = num_of_calc

    def evaluate(self):
        iterator = self._iterators["main"]
        gen = self._targets["gen"]
        it = copy.copy(iterator)#shallow copy
        summary = reporter.DictSummary()
        count = 0
        for batch in it:
            observation ={}
            with reporter.report_scope(observation):
                x, t = self.converter(batch, self.device)
                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    y = gen(x)
                loss = F.mean_squared_error(y, t)
                observation["val/loss"] = loss
            summary.add(observation)
            count += 1
            if(count == self._num_of_calc):
                return summary.compute_mean()

        return summary.compute_mean()
