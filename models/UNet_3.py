""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch.nn as nn
from .unet_parts import *


class UNet_3(nn.Module):
    def __init__(self, output_c = 1, bilinear=True):
        super(UNet_3, self).__init__()
        self.output_c = output_c
        self.bilinear = bilinear

        self.inc = DoubleConv(3, 64)
        # self.features.append(self.inc)
        self.down1 = Down(64, 128)
        # self.features.append(self.down1)
        factor = 2 if bilinear else 1
        self.down2 = Down(128, 256 // factor)
        # self.features.append(self.down4)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64, bilinear)
        # self.features.append(self.up1)
        self.outc = OutConv(64, output_c)
        self. init_weights()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        output = self.outc(x)
        return output


    def init_weights(self):
        print("[%s] Initialize weights..." % (self.__class__.__name__))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()