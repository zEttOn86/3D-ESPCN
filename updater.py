#coding:utf-8
import os, sys, time
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

class EspcnUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen = kwargs.pop('model')
        super(EspcnUpdater, self).__init__(*args, **kwargs)

    def mse_loss(self, y, t):
        loss = F.mean_squared_error(y, t)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')

        batch = self.get_iterator("main").next()#iterator
        x, t = self.converter(batch, self.device) #x: input, t: true

        gen = self.gen
        y = gen(x)

        loss_gen = self.mse_loss(y, t)
        gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()
        chainer.reporter.report({'loss' : loss_gen})
