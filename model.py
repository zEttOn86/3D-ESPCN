#coding:utf-8
"""
 20171018
 ESPCN
"""

#default
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
import sys, os, math, copy
#origin
from util.functions.PixelShuffler3D import pixelshuffler3d

class ESPCN(chainer.Chain):
    def __init__(self, in_ch=1, r=2):
        super(ESPCN, self).__init__()
        w = chainer.initializers.HeNormal()
        self.r = r

        with self.init_scope():
            self.conv1 = L.ConvolutionND(ndim=3, in_channels=1, out_channels=50, ksize=3, pad=1)
            self.conv2 = L.ConvolutionND(ndim=3, in_channels=50, out_channels=100, ksize=1, pad=0,initialW=w)
            self.conv3 = L.ConvolutionND(ndim=3, in_channels=100, out_channels=in_ch * (r**3), ksize=3, pad=1,initialW=w)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.conv3(h)
        h = pixelshuffler3d(h, self.r)
        return h
