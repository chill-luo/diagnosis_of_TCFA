""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, output_c = 2, bilinear=True):
        super(UNet, self).__init__()
        self.output_c = output_c
        self.bilinear = bilinear

        self.inc = DoubleConv(3, 64)

        # self.features.append(self.inc)
        self.down1 = Down(64, 128)
        # self.features.append(self.down1)
        self.down2 = Down(128, 256)
        # self.features.append(self.down2)
        self.down3 = Down(256, 512)
        # self.features.append(self.down3)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        # self.features.append(self.down4)
        self.up1 = Up(1024, 512 // factor, bilinear)
        # self.features.append(self.up1)
        self.up2 = Up(512, 256 // factor, bilinear)
        # self.features.append(self.up2)
        self.up3 = Up(256, 128 // factor, bilinear)
        # self.features.append(self.up3)
        self.up4 = Up(128, 64, bilinear)
        # self.features.append(self.up4)
        self.outc = OutConv(64, output_c)
        self.init_weights()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1,True)
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
