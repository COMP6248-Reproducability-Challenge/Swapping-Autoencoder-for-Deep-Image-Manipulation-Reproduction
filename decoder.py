import math

import torch
import torch.nn.functional as F
from torch import nn

from stylegan2_pytorch.stylegan2_model import ConvLayer, EqualLinear, ToRGB, StyledConv
# Initially implemented with two methods - as in taesungp version - but condensed into 1 -
#   I think they should both work the same
from taesung_data_loading.util import normalize

# Spatial Code refers to the Structure Code, and
# Global Code refers to the Texture Code of the paper.

'''
class ResPreservingResnet(torch.nn.Module):
    def __init__(self, in_ch, out_ch, texture_dim):
        super().__init__()
        self.conv1 = StyledConv(in_ch, out_ch, 3, texture_dim, upsample=False)
        self.conv2 = StyledConv(out_ch, out_ch, 3, texture_dim, upsample=False)

        if in_ch != out_ch:
            self.skip = ConvLayer(in_ch, out_ch, 1, activate=False, bias=False)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, texture):
        skip = self.skip(x)
        res = self.conv2(self.conv1(x, texture), texture)
        return (skip + res) / math.sqrt(2)

class UpsamplingResnet(torch.nn.Module):
    def __init__(self, in_ch, out_ch, texture_dim, blur_kernel=[1, 3, 3, 1], use_noise=False):
        super().__init__()
        self.in_ch, self.out_ch, self.texture_dim = in_ch, out_ch, texture_dim
        self.conv1 = StyledConv(in_ch, out_ch, 3, texture_dim, upsample=True, blur_kernel=blur_kernel, use_noise=use_noise)  # TODO don't get the parameters, just copying theirs
        self.conv2 = StyledConv(out_ch, out_ch, 3, texture_dim, upsample=False, use_noise=use_noise)

        if in_ch != out_ch:
            self.skip = ConvLayer(in_ch, out_ch, 1, activate=True, bias=True)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, texture):
        skip = F.interpolate(self.skip(x), scale_factor=2, mode='bilinear', align_corners=False) # TODO Don't really get this bit
        res = self.conv2(self.conv1(x, texture), texture)
        return (skip + res) / math.sqrt(2)
'''


class TextureResBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, texture_dim, upsample, blur_kernel=(1, 3, 3, 1)):
        super().__init__()

        self.upsample = upsample
        self.conv1 = StyledConv(in_ch, out_ch, 3, texture_dim, upsample=upsample,
                                blur_kernel=blur_kernel, use_noise=upsample)
        self.conv2 = StyledConv(out_ch, out_ch, 3, texture_dim)

        if in_ch == out_ch:
            self.skip = torch.nn.Identity()
        elif upsample:
            self.skip = ConvLayer(in_ch, out_ch, 1, activate=True, bias=True)
        else:
            self.skip = ConvLayer(in_ch, out_ch, 1, activate=False, bias=False)

    def forward(self, x, texture):
        if self.upsample:
            skip = F.interpolate(self.skip(x), scale_factor=2, mode='bilinear', align_corners=False)
        else:
            skip = self.skip(x)
        res = self.conv2(self.conv1(x, texture), texture)
        return (skip + res) / math.sqrt(2)


class GeneratorTexturalModulation(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.scale = EqualLinear(in_ch, out_ch)
        self.bias = EqualLinear(in_ch, out_ch)

    def forward(self, x, texture):
        if texture.ndimension() <= 2:
            return x * (1 * self.scale(texture)[:, :, None, None]) + self.bias(texture)[:, :, None, None]
        else:
            style = F.interpolate(texture, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            return x * (1 * self.scale(style)) + self.bias(style)


class Decoder(torch.nn.Module):

    def __init__(self,
                 channel=32,
                 structure_channels=8,
                 texture_channels=2048,
                 ch_multiplier=(4, 8, 12, 16, 16, 16, 8, 4),
                 upsample=(False, False, False, False, True, True, True, True, True),
                 blur_kernel=[1, 3, 3, 1],
                 use_noise=False):
        super().__init__()

        # Starting by modulating the structure code with the texture code
        self.TexturalModulation = GeneratorTexturalModulation(texture_channels, structure_channels)

        # Creating the next layers in a loop
        self.layers = nn.ModuleList()
        in_ch = structure_channels
        for ch_mul, up in zip(ch_multiplier, upsample):
            self.layers.append(TextureResBlock(in_ch, ch_mul * channel, texture_channels, up))
            in_ch = ch_mul * channel

        # Final layer to take it to RGB
        self.to_rgb = ToRGB(in_ch, texture_channels, blur_kernel=blur_kernel)

    def forward(self, structure_code, texture_code):
        structure_code = normalize(structure_code)
        texture_code = normalize(texture_code)

        # First modulation
        out = self.TexturalModulation(structure_code, texture_code)

        # Loop through all resnet layers (passing in texture code for modulation)
        for layer in self.layers:
            out = layer(out, texture_code)

        # To rgb
        out = self.to_rgb(out, texture_code)
        return out
