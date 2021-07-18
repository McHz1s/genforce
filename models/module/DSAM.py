import torch
import torch.nn as nn
from ASPP import ASPP
from WindowAttention import WindowAttention


class AM(nn.modules):
    def init(self, in_channels=3,
             out_channels=64,
             k_size=7):
        super(AM, self).init()
        self.first_conv = nn.Conv2d(in_channels, out_channels, k_size)
        self.att = WindowAttention()


class SAM(AM):
    def init(self, *args, **kwarg):
        super(SAM, self).init(*args, **kwarg)


class DSAM(SAM):
    def init(self, *args, **kwarg):
        super(DSAM, self).init(*args, **kwarg)
