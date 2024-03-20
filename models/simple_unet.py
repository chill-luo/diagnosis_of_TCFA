""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
import torch.nn as nn


class simple_unet(nn.Module):
    def __init__(self, output_c = 1, bilinear=True):
        super(simple_unet, self).__init__()
        self.output_c = output_c
        self.bilinear = bilinear

        self.inc = DoubleConv(3, 16)
        # self.features.append(self.inc)
        self.down1 = Down(16, 32)
        # self.features.append(self.down1)
        factor = 2 if bilinear else 1
        self.down2 = Down(32, 64 // factor)
        # self.features.append(self.down4)
        self.up1 = Up(64, 32 // factor, bilinear)
        self.up2 = Up(32, 16, bilinear)
        # self.features.append(self.up1)
        self.outc = OutConv(16, output_c)
        print('.... Model Initialize....')
        #self._init_weights()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        output = self.outc(x)
        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)