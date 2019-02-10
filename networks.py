# -*- coding: utf-8 -*-

__author__ = "Rahul Bhalley"

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_CHANNELS

###################
# Neural Networks #
###################
# --------------- #
# ResidualBlock   #
# --------------- #
# ConvTranspose2d #
# --------------- #
# Generator       #
# --------------- #
# Critic          #
###################


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels,  out_channels, 3, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, 3, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Initialize the weights with Xavier Glorot technique
        self.params_init()

    def params_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        return x + self.main(x)  # skip connection


class ConvTranspose2d(nn.Module):
    """
    Odena, et al., 2016. Deconvolution and Checkerboard Artifacts. Distill.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, upsample=None, output_padding=1):
        super(ConvTranspose2d, self).__init__()
        self.upsample = upsample
        if upsample:
            self.scale_factor = 4

        self.upsample_layer = F.interpolate
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad2d(reflection_pad, value=0)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False)
        self.convtrans2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False)

        # Initialize the weights with Xavier Glorot technique
        self.params_init()

    def params_init(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        if self.upsample:
            return self.conv2d(self.reflection_pad(self.upsample_layer(x, scale_factor=self.scale_factor)))
        else:
            return self.convtrans2d(x)


class Generator(nn.Module):

    def __init__(self, dim=64, n_blocks=9, upsample=None):
        super(Generator, self).__init__()

        self.encoder_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(N_CHANNELS,   dim * 1, 7, 1,    bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),

            # Downsampling layers
            nn.Conv2d(dim * 1,      dim * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(dim * 2,      dim * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(inplace=True)
        )

        # Residual layers
        self.transform_block = nn.Sequential()
        for i in range(n_blocks):
            self.transform_block.add_module(str(i), ResidualBlock(dim * 4, dim * 4))

        # Upsampling layers
        self.decoder_block = nn.Sequential(
            ConvTranspose2d(dim * 4, dim * 2, upsample=upsample),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),

            ConvTranspose2d(dim * 2, dim * 1, upsample=upsample),
            nn.BatchNorm2d(dim * 1),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, N_CHANNELS, 7, 1),
            nn.Tanh()
        )

        # Initialize the weights with Xavier Glorot technique
        self.params_init()

    def params_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        x = self.encoder_block(x)
        x = self.transform_block(x)
        x = self.decoder_block(x)
        return x


class Critic(nn.Module):

    def __init__(self, dim=64):
        super(Critic, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(N_CHANNELS,   dim * 1, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # Increase number of filters with layers
            nn.Conv2d(dim * 1,      dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim * 2,      dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim * 4,      dim * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim * 8, 1, 4, 1, 1)
        )

        # Initialize the weights with Xavier Glorot technique
        self.params_init()

    def params_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
    
    def forward(self, x):
        return self.main(x)