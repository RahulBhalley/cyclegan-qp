# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from config import config

def get_norm_layer(channels):
    if config.USE_INSTANCE_NORM:
        return nn.InstanceNorm2d(channels, affine=True)
    else:
        return nn.BatchNorm2d(channels)

def get_activation():
    return nn.GELU()

class SelfAttention(nn.Module):
    """
    Self-attention layer for GANs.
    Follows standard implementation with Spectral Normalization and learnable gamma.
    """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        
        # Spectral normalization is key for stabilizing attention gradients
        self.query_conv = spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1))
        self.key_conv = spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1))
        self.value_conv = spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, bias=False),
            get_norm_layer(channels),
            get_activation(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, bias=False),
            get_norm_layer(channels)
        )
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        return x + self.res_scale * self.main(x)

class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, upsample=None, output_padding=1):
        super(ConvTranspose2d, self).__init__()
        self.upsample = upsample
        if upsample:
            self.scale_factor = 2
            self.main = nn.Sequential(
                nn.Upsample(scale_factor=self.scale_factor, mode='nearest'),
                nn.ReflectionPad2d(kernel_size // 2),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, bias=False)
            )
        else:
            self.main = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False)

    def forward(self, x):
        return self.main(x)

class Generator(nn.Module):
    def __init__(self, dim=64, n_blocks=9, upsample=None):
        super(Generator, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(config.N_CHANNELS, dim, 7, 1, bias=False),
            get_norm_layer(dim),
            get_activation(),

            # Downsampling
            nn.Conv2d(dim, dim * 2, 3, 2, 1, bias=False),
            get_norm_layer(dim * 2),
            get_activation(),

            nn.Conv2d(dim * 2, dim * 4, 3, 2, 1, bias=False),
            get_norm_layer(dim * 4),
            get_activation()
        )

        # Transformer with Attention
        blocks = []
        for i in range(n_blocks):
            blocks.append(ResidualBlock(dim * 4))
            if i == n_blocks // 2:
                blocks.append(SelfAttention(dim * 4))
        self.transformer = nn.Sequential(*blocks)

        # Decoder
        self.decoder = nn.Sequential(
            ConvTranspose2d(dim * 4, dim * 2, upsample=upsample),
            get_norm_layer(dim * 2),
            get_activation(),

            ConvTranspose2d(dim * 2, dim, upsample=upsample),
            get_norm_layer(dim),
            get_activation(),

            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, config.N_CHANNELS, 7, 1),
            nn.Tanh()
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)) and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        return self.decoder(self.transformer(self.encoder(x)))

class Critic(nn.Module):
    def __init__(self, dim=64):
        super(Critic, self).__init__()

        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(config.N_CHANNELS, dim, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=False)),
            get_norm_layer(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(dim * 2, dim * 4, 4, 2, 1, bias=False)),
            get_norm_layer(dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            SelfAttention(dim * 4),

            spectral_norm(nn.Conv2d(dim * 4, dim * 8, 4, 1, 1, bias=False)),
            get_norm_layer(dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim * 8, 1, 4, 1, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)) and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        return self.main(x)
